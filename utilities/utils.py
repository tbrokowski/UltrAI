"""
Utility functions.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, Optional
import random
import yaml
import scipy.stats
import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict
from rich.console import Console
from rich.table import Table
from termcolor import colored
import matplotlib


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_config(config):
    print("======== CONFIG ========")
    print(yaml.dump(config.to_dict()), end="")
    print("========================")


def show_splits_info(train_indices, test_indices, valid_indices, labels_dict, label_names):
    console = Console()

    table = Table(show_header=True)
    table.add_column("split")
    table.add_column("size", justify="right")
    for label in label_names:
        table.add_column(label, justify="right")
    train_labels = np.array([labels_dict[i] for i in train_indices])
    valid_labels = np.array([labels_dict[i] for i in valid_indices])
    test_labels = np.array([labels_dict[i] for i in test_indices])

    table.add_row("train", str(len(train_labels)),
                  f"{len(train_labels) - train_labels.sum()} ({int(np.round((len(train_labels) - train_labels.sum()) / len(train_labels) * 100, 0))}%)",
                  f"{train_labels.sum()} ({int(np.round((train_labels.sum()) / len(train_labels) * 100, 0))}%)"
                  )
    table.add_row("valid", str(len(valid_labels)),
                  f"{len(valid_labels) - valid_labels.sum()} ({int(np.round((len(valid_labels) - valid_labels.sum()) / len(valid_labels) * 100, 0))}%)",
                  f"{valid_labels.sum()} ({int(np.round((valid_labels.sum()) / len(valid_labels) * 100, 0))}%)"
                  )
    table.add_row("test", str(len(test_labels)),
                  f"{len(test_labels) - test_labels.sum()} ({int(np.round((len(test_labels) - test_labels.sum()) / len(test_labels) * 100, 0))}%)",
                  f"{test_labels.sum()} ({int(np.round((test_labels.sum()) / len(test_labels) * 100, 0))}%)"
                  )

    print("Split infos:")
    console.print(table)

def log_metrics(title: str, metrics: dict, color=None) -> None:
    try:
        print(colored(f"{title}:", color))
        for key, value in metrics.items():
            if isinstance(value, (int, float)):  # Check if the value is a number
                print(colored(f"{key}: {value:.3f}", color))
            elif isinstance(value, list):
                formatted_values = ", ".join(f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in value)
                print(colored(f"{key}: [{formatted_values}]", color))
            else:
                print(colored(f"{key}: {value}", color))
    except:
        print(colored(f"{title}:", color))
        print(metrics)

def prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def split_array_most_equaly(array, num_splits: int):
    """Split array in k arrays of similar sizes."""
    n = len(array)
    split_sizes = np.ones(num_splits, dtype=int) * (n // num_splits)
    split_sizes[: n % num_splits] += 1

    offset = 0
    splits = []
    for size in split_sizes:
        splits.append(array[offset: offset + size])
        offset += size

    return splits


def split_k_folds(indices, labels, k: int, random_state: int = 0):
    """Stratified K-fold of the indices array."""
    # split indices per label
    indices_by_label = defaultdict(lambda: [])
    for index, label in zip(indices, labels):
        indices_by_label[label].append(index)

    # shuffle each with a fixed random key
    np.random.seed(random_state)
    separate_indices = []
    for _, indices in indices_by_label.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        separate_indices.append(indices)

    # split each in k folds
    folds = [[] for _ in range(k)]
    for i, indices in enumerate(separate_indices):
        # Smallest fold first for a greedy strategy to balance the split sizes.
        folds = sorted(folds, key=lambda indices: sum(map(len, indices)))
        current_label_folds = split_array_most_equaly(indices, k)
        for j in range(k):
            folds[j].append(current_label_folds[j])

    folds = [np.concatenate(indices) for indices in folds]

    # Reshuffle
    for f in folds:
        np.random.shuffle(f)

    return folds


def override_config_dict(config: ConfigDict, overrides: Dict[str, Any]):
    for k, v in overrides.items():
        try:
            if "." in k:
                first = k.split(".")[0]
                rest = ".".join(k.split(".")[1:])
                override_config_dict(config[first], {rest: v})
            else:
                config.get_ref(k).set(v)
        except KeyError:
            raise KeyError(f"Cannot override configuration field '{k}'")


def get_label_names(labels_file):
    if "diagnosis" in labels_file:
        return ["negative", "positive"]

    elif "severity" in labels_file or "prognosis" in labels_file:
        # mild = hospital,
        # severe = hospital with O2 or intubated
        return ["mild", "severe"]

    return None


def exclusive_cumsum(t, dim=-1):
    shape = list(t.shape)
    shape[dim] = 1
    zeros = torch.zeros(shape, dtype=t.dtype, device=t.device)
    return torch.cat(
        [zeros, torch.cumsum(t, dim=dim).narrow(dim=dim, start=0, length=t.shape[dim] - 1)], dim=dim
    )


def pad_dim_with_zeros(t, dim, length):
    if t.shape[dim] == length:
        return t
    t_padded_shape = list(t.shape)
    t_padded_shape[dim] = length
    t_padded = torch.zeros(t_padded_shape, device=t.device, dtype=t.dtype)
    t_padded.narrow(dim=dim, start=0, length=t.shape[dim]).copy_(t)
    return t_padded


def try_parse_exact_bool(b):
    if isinstance(b, str):
        if b.lower() == "true":
            return True
        if b.lower() == "false":
            return False
    return b


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def nice_plot_settings(font_size=18, font_family='STIXGeneral', mathtext_fontset='stix', usetex=True):
    matplotlib.rcParams['mathtext.fontset'] = mathtext_fontset
    matplotlib.rcParams['font.family'] = font_family
    matplotlib.rcParams['text.usetex'] = usetex
    matplotlib.rcParams['font.size'] = font_size


def print_summary_results(metric_folds, title):
    console = Console()
    table = Table(title=title, show_header=True)
    table.add_column("Criterion")
    table.add_column("mean +/- std", justify="right")
    for metric in metric_folds[0].keys():
        if metric in ['false_positive_rate', 'true_positive_rate']:
            continue
        values = [i[metric] for i in metric_folds]
        avg, std = np.mean(values), np.std(values)
        table.add_row(metric, f"{np.round(avg, 2)} +/- {np.round(std, 2)}")
    console.print(table)

