import logging
import time
import copy

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
from tqdm.auto import tqdm

from LM_ss_utilities import load_qt30_flair_multilabel_dataset, UnifiedAM_Conv, print_label_distribution, get_label_maps
from system_utilities import load_config, parse_args, setup_experiment_dir, EarlyStopping


def collate_fn(batch):
    # Sort by length (descending)
    batch.sort(key=lambda x: x[0].size(0), reverse=True)

    inputs = [item[0] for item in batch]  # (T, D)
    targets = [item[1] for item in batch]  # (T, 30) float 0/1
    lengths = torch.tensor([x.size(0) for x in inputs], dtype=torch.long)

    inputs_pad = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0)  # (B,T,D)

    # Pad targets with zeros (multi-label)
    targets_pad = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0.0)  # (B,T,30)

    # Mask: 1 where valid tokens, 0 where padding
    max_len = inputs_pad.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)  # (B,T) bool
    mask = mask.unsqueeze(-1).float()  # (B,T,1)

    return inputs_pad, lengths, targets_pad, mask


# --- 3. Training Utilities ---
def train_one_epoch(model, loader, optimizer, criterion, desc, device):
    """
    Train one epoch
    :param model:
    :param loader:
    :param optimizer:
    :param criterion:
    :param device:
    :return:
    """
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc=desc or "Train", leave=False)

    for batch in loop:
        inputs = batch[0].to(device)
        targets = batch[2].to(device)
        mask = batch[3].to(device)

        optimizer.zero_grad()
        logits, _, _ = model(inputs)

        # logits should be (B,T,29)
        loss_raw = criterion(logits, targets)  # (B,T,29) if reduction='none'
        loss = (loss_raw * mask).sum() / (mask.sum() * targets.size(-1) + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def evaluate(model, loader, criterion, desc, device):
    """

    :param model:
    :param loader:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc=desc or "Eval", leave=False)

        for batch in loop:
            inputs = batch[0].to(device)
            targets = batch[2].to(device)
            mask = batch[3].to(device)

            logits, _, _ = model(inputs)

            loss_raw = criterion(logits, targets)
            loss = (loss_raw * mask).sum() / (mask.sum() * targets.size(-1) + 1e-8)

            total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


# --- 4. Phase 1: HPO on Hold-out Set ---
def run_hpo(dataset, embedding_dim, device, n_labels=29):
    """

    :param dataset:
    :param embedding_dim:
    :param device:
    :param n_labels:
    :return:
    """
    logging.info("Starting  Hyperparameter Optimization")
    max_epochs = 10  # less epochs during HPO
    n_trials = 8

    def objective(trial):

        # hyperparameters to optimize
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 9e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])

        # Create Hold-out Split (80% Train, 20% Val)
        train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
        train_sub = torch.utils.data.Subset(dataset, train_indices)
        val_sub = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_sub, batch_size=batch_size, collate_fn=collate_fn)

        # load model and optimizer.
        model = UnifiedAM_Conv(input_dim=embedding_dim, num_labels=n_labels).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        es = EarlyStopping(patience=3, delta=0.001)

        best_val_loss = float('inf')

        # Short training loop for HPO (e.g., max 15 epochs)
        for epoch in range(max_epochs):
            epoch_desc = f"HPO- T{trial.number} Ep{epoch + 1}/{max_epochs}"

            _ = train_one_epoch(model, train_loader, optimizer, criterion, epoch_desc, device)
            val_loss = evaluate(model, val_loader, criterion, epoch_desc, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            trial.report(val_loss, step=epoch)

            if trial.should_prune():
                logging.info(f"Trial {trial.number} pruned at epoch {epoch + 1} "
                             f"with val_loss={val_loss:.4f}"
                             )
                raise optuna.exceptions.TrialPruned()

            es(val_loss)
            if es.early_stop:
                logging.info(f"Early stopping at epoch {epoch + 1} ")
                break

        return best_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # print(f"Best HPO: {study.best_params}")
    logging.info(f"Best HPO params: {study.best_params}")

    return study.best_params


# --- 5. Phase 2: CV and Final Model ---
def run_cv_and_save(dataset, config, embedding_dim, device, n_labels=29):
    """
    Cross validation and save the best model
    :param dataset:
    :param config:
    :param embedding_dim:
    :param device:
    :param n_labels:
    :return:
    """

    logging.info("Starting  Cross Validation to find best model")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_overall_loss = float('inf')
    best_model_state = None
    fold_metrics = []
    lr = config['learning_rate']
    batch_size = config['batch_size']
    max_epochs = 30

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        logging.info("--- Fold %d ---", fold + 1)

        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_sub, batch_size=batch_size, collate_fn=collate_fn)

        model = UnifiedAM_Conv(input_dim=embedding_dim, num_labels=n_labels).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        es = EarlyStopping(patience=3, delta=0.001)

        fold_best_loss = float('inf')
        fold_best_state = None

        start_time = time.time()
        for epoch in range(max_epochs):  # Full training epochs
            epoch_desc = f"Fold {fold + 1} Ep{epoch + 1}/{max_epochs}"
            _ = train_one_epoch(model, train_loader, optimizer, criterion, epoch_desc, device)
            val_loss = evaluate(model, val_loader, criterion, epoch_desc, device)

            # Track best within fold
            if val_loss < fold_best_loss:
                fold_best_loss = val_loss
                fold_best_state = copy.deepcopy(model.state_dict())

            es(val_loss)
            if es.early_stop:
                logging.info(f"Early Stopping at Epoch {epoch}")
                break

        logging.info(f"Fold {fold + 1} Best Validation Loss: {fold_best_loss:.4f}")
        fold_metrics.append(fold_best_loss)

        # Check if this fold produced the absolute best model across all CV
        if fold_best_loss < best_overall_loss:
            best_overall_loss = fold_best_loss
            best_model_state = fold_best_state

        time_elapsed = time.time() - start_time

        logging.info("Time taken by Fold %d is %f seconds", fold + 1, time_elapsed)
        fold_metrics.append([fold, time_elapsed, fold_best_loss])

    # Save the absolute best model
    if best_model_state:
        out_path = run_dir / "best_bilstm_er_model.pth"
        torch.save(best_model_state, out_path)
        logging.info("Best model saved to %s", out_path)
    else:
        logging.info("No Best model saved")

    return fold_metrics


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    run_dir = setup_experiment_dir(config, if_custom=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. Setup Flair Embeddings
    # Using word embeddings for general and bytepair for oov
    logging.info("Loading Flair Embeddings...")

    flair_embedding = StackedEmbeddings(
        [
            # WordEmbeddings('en'),
            # # Byte pair embeddings for English handling OOV
            # BytePairEmbeddings('en'),
            FlairEmbeddings('news-forward-fast'),
            FlairEmbeddings('news-backward-fast'),
            ]
        )
    embedding_dim = flair_embedding.embedding_length  # Should be 2048 or 4096 depending on fast/normal
    logging.info(f">> Flair Embedding Length: {embedding_dim}")

    csv_path = config["experiment"].get("data_path", "app/data/QT30_training.csv")
    dataset, conversation_ids = load_qt30_flair_multilabel_dataset(csv_path, flair_embedding)

    # print dataset label distribution if needed
    id_to_label, label_to_id = get_label_maps()
    print_label_distribution(dataset, label_to_id, id_to_label)

    logging.info("Loaded data set with flair embeddings")

    # # C. Run Phase 1: HPO
    # n_labels = config["unified"].get("num_labels", 29)
    # best_config = run_hpo(dataset, embedding_dim, device, n_labels=n_labels)
    #
    # # D. Run Phase 2: CV and Final Save
    # run_cv_and_save(dataset, best_config, embedding_dim, device)
