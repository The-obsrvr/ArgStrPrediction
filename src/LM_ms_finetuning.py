import json
import time
import copy
import logging

import numpy as np
import torch
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from transformers import (get_linear_schedule_with_warmup,
                          DataCollatorForTokenClassification, XLMRobertaTokenizerFast
                          )
from torch.optim import AdamW
from tqdm.auto import tqdm
# Local Imports
from system_utilities import parse_args, load_config, setup_experiment_dir, EarlyStopping
from LM_ms_utilities import prepare_data_for_AUE, prepare_data_for_RTC, ArgumentClassifier


def get_data_loaders(dataset, batch_size, indices=None):
    if indices is not None:
        subset = Subset(dataset, indices)
    else:
        subset = dataset
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights, is_long_model, desc):
    """
    Run an epoch of training.
    """
    model.train()
    total_loss = 0.0

    weights_tensor = torch.tensor(class_weights).float().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-100)

    loop = tqdm(dataloader, desc=desc or "Train", leave=False)

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tgt_mask = batch.get("tgt_mask")
        src_mask = batch.get("src_mask")
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
        if src_mask is not None:
            src_mask = src_mask.to(device)
        labels = batch["labels"].to(device)

        # Handle global attention for long models
        global_attention_mask = None
        if is_long_model:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
            global_attention_mask = global_attention_mask.to(device)

        model.zero_grad()

        logits = model(input_ids, attention_mask,
                       global_attention_mask=global_attention_mask,
                       tgt_mask=tgt_mask, src_mask=src_mask)

        # --- Handle token-level (AUE) vs sequence-level (RTC) ---
        if logits.dim() == 3 and labels.dim() == 2:
            # Token-level: [B, S, C] vs [B, S]
            loss = criterion(
                logits.view(-1, logits.size(-1)),  # [B*S, C]
                labels.view(-1),  # [B*S]
                )
        elif logits.dim() == 2 and labels.dim() == 1:
            # Sequence-level: [B, C] vs [B]
            loss = criterion(logits, labels)
        else:
            raise ValueError(
                f"Unexpected shapes in train_epoch: logits={logits.shape}, labels={labels.shape}"
                )

        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, class_weights, is_long_model):
    """
    Evaluate model on a validation set.
    Returns (avg_loss, macro_f1).
    """
    model.eval()
    total_loss = 0.0

    weights_tensor = torch.tensor(class_weights).float().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-100)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tgt_mask = batch.get("tgt_mask")
            src_mask = batch.get("src_mask")
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(device)
            if src_mask is not None:
                src_mask = src_mask.to(device)

            global_attention_mask = None
            if is_long_model:
                global_attention_mask = torch.zeros_like(input_ids)
                global_attention_mask[:, 0] = 1
                global_attention_mask = global_attention_mask.to(device)

            logits = model(input_ids, attention_mask,
                           global_attention_mask=global_attention_mask,
                           tgt_mask=tgt_mask, src_mask=src_mask)

            # --- Loss computation as in train_epoch ---
            if logits.dim() == 3 and labels.dim() == 2:
                # Token-level
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    )

                preds = logits.argmax(dim=-1)  # [B, S]

                # ignore positions with label = -100 (padding/subwords)
                valid_mask = labels != -100
                all_preds.extend(
                    preds[valid_mask].contiguous().view(-1).cpu().tolist()
                    )
                all_labels.extend(
                    labels[valid_mask].contiguous().view(-1).cpu().tolist()
                    )

            elif logits.dim() == 2 and labels.dim() == 1:
                # Sequence-level (RTC)
                loss = criterion(logits, labels)

                preds = logits.argmax(dim=-1)  # [B]
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            else:
                raise ValueError(
                    f"Unexpected shapes in evaluate: logits={logits.shape}, labels={labels.shape}"
                    )

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    if len(all_labels) == 0:
        f1 = 0.0
    else:
        f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, f1


def run_hpo(train_dataset, val_dataset, config, current_task: str, collator, device):
    """
    Runs HPO on the given dataset.
    - Optimizes validation loss (minimize).
    - Use Optuna pruning, but only AFTER a minimum number of epochs [2]
    - use EarlyStopping inside each trial.
    :param train_dataset:
    :param val_dataset:
    :param config:
    :param current_task:
    :param collator:
    :param device:
    """
    logging.info(f"Starting HPO for {current_task}")

    min_epochs_for_prune = 2  # run at least this many epochs before pruning
    # make less trials for RTC since it takes a lot longer to run
    if current_task == "RTC":
        max_epochs = 5
        n_trials = 5
    else:
        max_epochs = 5
        n_trials = 8

    # Less aggressive pruner: wait for a few startup trials + warmup steps
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2,  # first 2 trials run to completion
        n_warmup_steps=1,  # ignore pruning at epoch 0
        interval_steps=1,
        )

    def objective(trial):

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 9e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])

        # Setup Data
        if current_task == "AUE":
            train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup Model
        model = ArgumentClassifier(
            config[current_task]["model_name_or_path"],
            config[current_task]["num_labels"],
            is_long_model=config[current_task]["is_long_model"],
            task_type=current_task,
            )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * max_epochs,
            )

        early_stopping = EarlyStopping(patience=2, delta=0.001)
        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            epoch_desc = f"HPO-{current_task} T{trial.number} Ep{epoch + 1}/{max_epochs}"
            train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                config[current_task]["class_weights"],
                config[current_task]["is_long_model"],
                epoch_desc,
                )

            val_loss, val_f1 = evaluate(
                model,
                val_loader,
                device,
                config[current_task]["class_weights"],
                config[current_task]["is_long_model"],
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Report to Optuna
            trial.report(val_loss, step=epoch)

            # Allow pruning ONLY after min_epochs_for_prune
            if epoch + 1 >= min_epochs_for_prune:
                if trial.should_prune():
                    logging.info(
                        f"Trial {trial.number} pruned at epoch {epoch + 1} "
                        f"with val_loss={val_loss:.4f}"
                        )
                    raise optuna.exceptions.TrialPruned()

            # Early stopping inside the trial based on val_loss dynamics
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info(
                    f"Early stopping in trial {trial.number} at epoch {epoch + 1}"
                    )
                break

        return best_val_loss

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    # print(f"Best HPO: {study.best_params}")
    logging.info(f"Best HPO params for {current_task}: {study.best_params}")

    return study.best_params


def run_cv_and_save(
        dataset, best_params, config,
        current_task, collator,
        device, run_dir
        ):
    """

    :param collator:
    :param dataset:
    :param best_params:
    :param config:
    :param current_task:
    :param device:
    :param run_dir:
    :return:
    """

    logging.info(f"Starting CV for {current_task}")

    kf = KFold(n_splits=5, shuffle=True, random_state=config["experiment"]["seed"])
    fold_results = []

    best_model_state = None

    lr = best_params["learning_rate"]
    batch_size = best_params["batch_size"]
    epochs = 20
    if current_task == "RTC":
        epochs = 15

    best_model_path = run_dir / f"best_model_{current_task}.pt"

    # start the 5-fold search
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logging.info("--- Fold %d ---", fold + 1)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        if current_task == "AUE":

            train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collator, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collator, shuffle=False)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = ArgumentClassifier(config[current_task]["model_name_or_path"],
                                   config[current_task]["num_labels"],
                                   is_long_model=config[current_task]["is_long_model"],
                                   task_type=current_task
                                   )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * epochs)
        early_stopping = EarlyStopping(patience=3, delta=0.001)

        fold_best_f1 = -1.0
        best_overall_f1 = -1.0

        # print(f"Start Fold {fold + 1}/5")
        start_time = time.time()
        for epoch in range(epochs):
            epoch_desc = f"{current_task}-CV Fold{fold + 1}/5 Ep{epoch + 1}/{epochs}"
            train_epoch(model, train_loader, optimizer, scheduler, device,
                        config[current_task]["class_weights"], config[current_task]["is_long_model"], epoch_desc
                        )
            val_loss, val_f1 = evaluate(model, val_loader, device,
                                        config[current_task]["class_weights"], config[current_task]["is_long_model"]
                                        )
            # check if this is the best F1
            if val_f1 > fold_best_f1:
                fold_best_f1 = val_f1

                if val_f1 > best_overall_f1:
                    best_overall_f1 = val_f1
                    best_model_state = copy.deepcopy(model.state_dict())

            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info(f"Early stopping ends at {epoch + 1}/{epochs}")
                break

        time_taken = time.time() - start_time
        logging.info("Time taken by Fold " + str(fold + 1) + ": " + str(time_taken))
        fold_results.append([time_taken, fold_best_f1, best_overall_f1])

    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)
        logging.info(f"Saved best model to {best_model_path}")
        # print(f"Saved best model to {best_model_path}")

        with open(run_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
    else:
        logging.info(f"Warning: No best model to save")
        # print(f"Warning: No best model to save")

    return fold_results


def main():
    """

    :return:
    """

    args = parse_args()
    config = load_config(args)
    run_dir = setup_experiment_dir(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train AUE first and then RTC
    task_order = ["AUE", "RTC"]

    for task in task_order:
        logging.info("Task: " + task)
        # format data as per task.
        if task == "AUE":
            full_dataset, class_weights = prepare_data_for_AUE(config)

            config["AUE"]["class_weights"] = class_weights
        else:
            full_dataset = prepare_data_for_RTC(config)

        logging.info("Data loaded")
        indices = list(range(len(full_dataset)))
        # split training file into train and one hold-out validation set for HPO
        train_idx_hpo, val_idx_hpo = train_test_split(indices, test_size=0.3,
                                                      random_state=config["experiment"]["seed"]
                                                      )

        train_set_hpo = Subset(full_dataset, train_idx_hpo)
        val_set_hpo = Subset(full_dataset, val_idx_hpo)

        tokenizer = XLMRobertaTokenizerFast.from_pretrained(config[task]["model_name_or_path"])

        collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            max_length=config[task]["max_length"],
            )

        # Run HPO
        best_params = run_hpo(train_set_hpo, val_set_hpo, config, current_task=task, collator=collator, device=device)

        # Run CV and save the best model
        all_results = run_cv_and_save(full_dataset, best_params, config, current_task=task,
                                      collator=collator, device=device, run_dir=run_dir
                                      )

        time_results = all_results[0]
        logging.info(f"\nFINISHED {task}. Avg Time: {np.mean(time_results):.2f} sec")


if __name__ == "__main__":
    main()
