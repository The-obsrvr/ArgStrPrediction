import argparse
import random
import yaml
import datetime
import logging
from pathlib import Path
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a machine learning experiment.")

    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', '--learning_rate', type=float, default=None)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--num_examples', type=int, default=None)  # Changed to None
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--context_examples', type=str, default=None)
    parser.add_argument('--do_train', action='store_true', default=None)
    parser.add_argument('--do_inference', action='store_true', default=None)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--AUE_model_name_or_path', type=str, default=None)
    parser.add_argument('--RTC_model_name_or_path', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    return parser.parse_args()


def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Mapping Overrides: (arg_name, (config_section, config_key)) ---
    overrides = {
        'do_train': ('experiment', 'do_train'),
        'do_inference': ('experiment', 'do_inference'),
        'model_name_or_path': ('model', 'model_name_or_path'),
        'data': ('inference', 'data_path'),
        'context_examples': ('inference', 'context_examples_path'),
        'num_examples': ('inference', 'num_context_examples'),
        'epochs': ('finetuning', 'epochs'),
        'lr': ('finetuning', 'learning_rate'),
        }

    config.setdefault("experiment", {})
    config["experiment"].setdefault("do_inference", True)
    config["experiment"].setdefault("do_train", False)

    # Apply overrides ONLY if:
    #   (1) user provided a CLI value (val is not None)
    #   (2) the config file already contains that section and key
    for arg_key, (cfg_sec, cfg_key) in overrides.items():
        val = getattr(args, arg_key, None)
        if val is None:
            continue
        # Section must exist and be a dict
        if cfg_sec not in config or not isinstance(config.get(cfg_sec), dict):
            continue
        config[cfg_sec][cfg_key] = val

    # --- Handle Run Name Logic ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or config['experiment'].get('name', 'exp')
    seed = args.seed if args.seed is not None else config['experiment'].get('seed', 42)

    if args.run_name is None:
        config['experiment']['run_name'] = f"{name}_{seed}_{timestamp}"
    else:
        config['experiment']['run_name'] = args.run_name

    # fix model name pathing in multi-step setting
    config["inference"]["model_name_or_path"] = args.model_name_or_path
    config["experiment"]["data_path"] = args.data
    if args.AUE_model_name_or_path is not None:
        config['model']['AUE_model_name_or_path'] = args.AUE_model_name_or_path
        config['AUE']['model_name_or_path'] = args.AUE_model_name_or_path
        config['model']['RTC_model_name_or_path'] = args.RTC_model_name_or_path
        config['RTC']['model_name_or_path'] = args.RTC_model_name_or_path

    else:
        config["model"]["args_model_name_or_path"] = args.model_name_or_path
        config["model"]["rels_model_name_or_path"] = args.model_name_or_path

    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_experiment_dir(config, if_custom = True):
    """
    Sets up the experiment directory, logging and the experimental seed for the run.
    :param config:
    :return:
    """
    # Create the experiment directory
    if if_custom:
        run_dir = Path('experiments') / config['experiment']['run_name']
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path('experiments') / "lm_ss_123"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to the file and console
    log_file = run_dir / 'experiment.log'
    logging.basicConfig(
        level=config['experiment']['log_level'].upper(),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
            ],
        force=True
        )

    seed_everything(config['experiment']['seed'])

    logging.info(f"Experiment directory created at: {run_dir} with seed: {config['experiment']['seed']}")

    return run_dir


def count_tokens(text, tokenizer):
    """
    Returns the number of tokens in a text string.
    :param text:
    :param tokenizer:
    """

    if not text:
        return 0

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            )
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return len(encoded["input_ids"])
    except Exception:
        pass

    # Fallback to .encode
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.min_delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
