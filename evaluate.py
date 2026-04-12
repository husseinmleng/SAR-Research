"""Evaluation utilities: metrics, JSON report saving, comparison tables."""

import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, class_names, unseen_label=None):
    """Compute full classification metrics.

    Args:
        y_true: list[int] ground-truth labels
        y_pred: list[int] predicted labels
        class_names: list[str] ordered class names (index -> name)
        unseen_label: int index of the unseen class (nway), or None

    Returns:
        dict with overall_accuracy, seen_accuracy, unseen_accuracy,
              per_class (dict name -> precision/recall/f1),
              confusion_matrix (list of lists)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = list(range(len(class_names)))

    overall_acc = accuracy_score(y_true, y_pred) * 100.0

    if unseen_label is not None:
        seen_mask   = y_true != unseen_label
        unseen_mask = y_true == unseen_label
        seen_acc   = accuracy_score(y_true[seen_mask],   y_pred[seen_mask])   * 100.0 if seen_mask.any()   else 0.0
        unseen_acc = accuracy_score(y_true[unseen_mask], y_pred[unseen_mask]) * 100.0 if unseen_mask.any() else 0.0
    else:
        seen_acc = overall_acc
        unseen_acc = 0.0

    per_class = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if not mask.any():
            per_class[name] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
            continue
        per_class[name] = {
            'precision': float(precision_score(y_true, y_pred, labels=[i], average='micro', zero_division=0) * 100),
            'recall':    float(recall_score(   y_true, y_pred, labels=[i], average='micro', zero_division=0) * 100),
            'f1':        float(f1_score(       y_true, y_pred, labels=[i], average='micro', zero_division=0) * 100),
            'support':   int(mask.sum()),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        'overall_accuracy': float(overall_acc),
        'seen_accuracy':    float(seen_acc),
        'unseen_accuracy':  float(unseen_acc),
        'per_class':        per_class,
        'confusion_matrix': cm,
    }


def save_report(metrics, config, output_dir, iteration):
    """Save metrics dict as JSON to output_dir/{config}_{iteration}.json.

    Args:
        metrics: dict from compute_metrics()
        config:  str — experiment identifier (e.g. '3way_20shot_gnn')
        output_dir: str — directory path
        iteration: int or str — training iteration or label

    Returns:
        str — path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f'{config}_{iteration}.json'
    fpath = os.path.join(output_dir, fname)
    with open(fpath, 'w') as f:
        json.dump({'config': config, 'iteration': iteration, **metrics}, f, indent=2)
    return fpath


def print_comparison_table(report_paths):
    """Load multiple JSON report files and print a Markdown comparison table.

    Args:
        report_paths: list[str] — paths to JSON files produced by save_report()

    Returns:
        str — the Markdown table string (also printed to stdout)
    """
    rows = []
    for path in report_paths:
        with open(path) as f:
            d = json.load(f)
        rows.append({
            'Config': d.get('config', os.path.basename(path)),
            'Iter':   d.get('iteration', ''),
            'Overall': f"{d.get('overall_accuracy', 0):.2f}%",
            'Seen':    f"{d.get('seen_accuracy', 0):.2f}%",
            'Unseen':  f"{d.get('unseen_accuracy', 0):.2f}%",
        })

    if not rows:
        return ''

    cols = ['Config', 'Iter', 'Overall', 'Seen', 'Unseen']
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}

    def fmt_row(r):
        return '| ' + ' | '.join(str(r[c]).ljust(widths[c]) for c in cols) + ' |'

    header = fmt_row({c: c for c in cols})
    sep    = '| ' + ' | '.join('-' * widths[c] for c in cols) + ' |'
    lines  = [header, sep] + [fmt_row(r) for r in rows]
    table  = '\n'.join(lines)
    print(table)
    return table
