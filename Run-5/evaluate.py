"""
evaluate.py - Run-5: Comprehensive Model Evaluation for Netra-Adapt

Run-5 change: result paths updated to /workspace/results_run5/
              GrayscaleToRGB removed — all models evaluated on full RGB
              colour inputs that they were trained on.

Evaluates all trained models on the labeled Chákṣu test set:
- Phase A: Source model (baseline from AIROGS)
- Phase B: Oracle model (upper bound with Chákṣu labels)
- Phase C: Netra-Adapt (source-free adapted model)

Metrics (Research Paper Standard):
- AUROC, Sensitivity, Specificity, Precision, F1-Score, Sens@95%Spec
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms
from training_logger import get_logger

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CSV_CHAKSU = "/workspace/data/processed_csvs/chaksu_test_labeled.csv"
TEST_CSV_AIROGS = "/workspace/data/processed_csvs/airogs_test.csv"
RESULTS_DIR     = "/workspace/results_run5/evaluation"

MODELS_CHAKSU = {
    "Pretrained → Chákṣu":     None,
    "AIROGS → Chákṣu":        "/workspace/results_run5/Source_AIROGS/model.pth",
    "Chákṣu → Chákṣu":        "/workspace/results_run5/Oracle_Chaksu/oracle_model.pth",
    "AIROGS+Adapt → Chákṣu":  "/workspace/results_run5/Netra_Adapt/adapted_model.pth",
}

MODELS_AIROGS = {
    "AIROGS → AIROGS": "/workspace/results_run5/Source_AIROGS/model.pth",
}

os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(model_path, name, test_csv, use_mean_pool=False):
    """
    Evaluate a single model on the test set with comprehensive metrics.

    use_mean_pool=True: classify via mean-pooled patch tokens [B, 1:, D].mean(1)
    use_mean_pool=False: classify via CLS token (standard forward)

    The adapted model (AIROGS+Adapt) requires use_mean_pool=True because
    its head was re-trained during adaptation on mean-pooled patch features,
    not CLS tokens.  All other models use CLS (consistent with their training).
    """
    model = NetraModel(num_classes=2).to(DEVICE)

    if model_path is None:
        print(f"  Using pretrained DINOv3 with random classifier head (no fine-tuning)")
    else:
        if not os.path.exists(model_path):
            print(f"  [SKIP] Model not found: {model_path}")
            return None
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()

    dataset = GlaucomaDataset(test_csv, transform=get_transforms(is_training=False))
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  Evaluating", leave=False):
            images = images.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                if use_mean_pool:
                    # Adapted model: head was trained on mean-pooled patch tokens
                    all_toks = model.extract_all_tokens(images)   # [B, N+1, D]
                    feats    = all_toks[:, 1:].mean(dim=1)        # [B, D]
                    logits   = model.head(feats)
                else:
                    # Source / Oracle: head was trained on CLS token
                    logits = model(images)
                probs  = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    valid_mask = all_labels >= 0
    all_labels = all_labels[valid_mask]
    all_probs  = all_probs[valid_mask]

    if len(all_labels) == 0:
        print(f"  [ERROR] No valid test samples")
        return None

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        print(f"  [ERROR] Cannot compute AUROC (possibly single class)")
        return None

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx       = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions       = (all_probs >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy    = accuracy_score(all_labels, predictions)

    valid_indices = np.where(fpr <= 0.05)[0]
    sens_at_95    = tpr[valid_indices[-1]] if len(valid_indices) > 0 else 0.0

    return {
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'sens_at_95': sens_at_95,
        'confusion_matrix': confusion_matrix(all_labels, predictions),
        'fpr': fpr,
        'tpr': tpr,
        'predictions': predictions,
        'labels': all_labels,
        'probs': all_probs
    }


def plot_roc_curves(all_results):
    plt.figure(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for (name, metrics), color in zip(all_results.items(), colors):
        if metrics and 'fpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'],
                     label=f"{name} (AUROC={metrics['auroc']:.3f})",
                     linewidth=2.5, color=color)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUROC=0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    plt.title('ROC Curves — Run-5 (Colour + Full Algorithm Fix)', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/roc_curves.pdf", bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved ROC curves to {RESULTS_DIR}/roc_curves.png")


def plot_confusion_matrices(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4))
    if len(all_results) == 1:
        axes = [axes]
    for idx, (name, metrics) in enumerate(all_results.items()):
        if metrics and 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Glaucoma'],
                        yticklabels=['Normal', 'Glaucoma'],
                        ax=axes[idx], cbar=True, square=True,
                        annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            axes[idx].set_title(name, fontsize=11, fontweight='bold', pad=10)
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrices to {RESULTS_DIR}/confusion_matrices.png")


def plot_metrics_comparison(all_results):
    metrics_to_plot = ['auroc', 'sensitivity', 'specificity', 'precision', 'f1', 'sens_at_95']
    metric_names    = ['AUROC', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'Sens@95%Spec']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes  = axes.flatten()
    colors = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71']
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax     = axes[idx]
        models = list(all_results.keys())
        values = [all_results[m][metric] if all_results[m] else 0 for m in models]
        bars   = ax.bar(range(len(models)), values, color=colors[:len(models)], alpha=0.8,
                        edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    plt.suptitle('Performance Metrics — Run-5 (Colour + Full Algorithm Fix)', fontsize=15,
                 fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/metrics_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics comparison to {RESULTS_DIR}/metrics_comparison.png")


def save_results_table(all_results):
    rows = []
    for name, metrics in all_results.items():
        if metrics:
            rows.append({
                'Model':        name,
                'AUROC':        f"{metrics['auroc']:.4f}",
                'Sensitivity':  f"{metrics['sensitivity']:.4f}",
                'Specificity':  f"{metrics['specificity']:.4f}",
                'Precision':    f"{metrics['precision']:.4f}",
                'F1-Score':     f"{metrics['f1']:.4f}",
                'Accuracy':     f"{metrics['accuracy']:.4f}",
                'Sens@95%Spec': f"{metrics['sens_at_95']:.4f}",
            })
    df = pd.DataFrame(rows)
    csv_path = f"{RESULTS_DIR}/results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results table to {csv_path}")
    latex_path = f"{RESULTS_DIR}/results_table.tex"
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False))
    print(f"✓ Saved LaTeX table to {latex_path}")


def main():
    exp_logger = get_logger()
    print("\n" + "="*70)
    print("   NETRA-ADAPT RUN-5: COMPREHENSIVE EVALUATION")
    print("   Colour inputs restored  +  Balanced training  +  Label matching fixed")
    print("="*70)

    # Sanity check on AIROGS
    print("\n[SANITY CHECK] Evaluating on AIROGS Test Set")
    print("-" * 70)
    if os.path.exists(TEST_CSV_AIROGS):
        for name, path in MODELS_AIROGS.items():
            print(f"\n{name}")
            metrics = evaluate(path, name, TEST_CSV_AIROGS)
            if metrics:
                print(f"  AUROC: {metrics['auroc']:.4f}  (should be >0.85 if training worked)")
    else:
        print(f"[SKIP] AIROGS test CSV not found: {TEST_CSV_AIROGS}")

    # Main experiments on Chákṣu
    print("\n\n[MAIN EXPERIMENTS] Evaluating on Chákṣu Test Set (Cross-Ethnic)")
    print("-" * 70)
    if not os.path.exists(TEST_CSV_CHAKSU):
        print(f"\n[ERROR] Chákṣu test CSV not found: {TEST_CSV_CHAKSU}")
        print("        Run prepare_data.py first!")
        return

    all_results = {}
    for name, path in MODELS_CHAKSU.items():
        print(f"\n{name}")
        print("-" * 70)
        # Adapted model's head was trained on mean-pooled patches during adaptation
        use_mp = (name == "AIROGS+Adapt \u2192 Ch\u00e1k\u1e63u")
        metrics = evaluate(path, name, TEST_CSV_CHAKSU, use_mean_pool=use_mp)
        if metrics:
            all_results[name] = metrics
            metrics_log = {k: float(v) for k, v in metrics.items()
                           if k not in ('confusion_matrix', 'fpr', 'tpr', 'predictions', 'labels', 'probs')}
            exp_logger.log_evaluation_metrics(name, metrics_log)
            print(f"  AUROC:        {metrics['auroc']:.4f}  ({metrics['auroc']*100:.1f}%)")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f}  ({metrics['sensitivity']*100:.1f}%)")
            print(f"  Specificity:  {metrics['specificity']:.4f}  ({metrics['specificity']*100:.1f}%)")
            print(f"  Precision:    {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
            print(f"  F1-Score:     {metrics['f1']:.4f}")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
            print(f"  Sens@95Spec:  {metrics['sens_at_95']:.4f}  ({metrics['sens_at_95']*100:.1f}%)")
        else:
            all_results[name] = None

    print("\n" + "="*70)
    print("   GENERATING VISUALIZATIONS...")
    plot_roc_curves(all_results)
    plot_confusion_matrices(all_results)
    plot_metrics_comparison(all_results)
    save_results_table(all_results)
    print("\n" + "="*70)
    print("   EVALUATION COMPLETE")
    print(f"   All results saved to: {RESULTS_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
