import os, glob, re, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime

# ========= user setting =========
GT_DATA_ROOT = "/data/volme/data/confusion/PTv1_volmeda/npy"  # GT NPY data path
PRED_DATA_ROOT = "/data/volme/data/confusion/PTv1_volmeda/npy"  # PRED NPY data path
IGNORE_INDEX = 16  # ex: neither 255 nor -1. or None
NUM_CLASSES = 16
CLASS_NAMES = [
    "Backpack", "Beanie", "Boots", "Cap", "Jeans", "Sweatshirt", "Shirt", "Knitwear",
    "JoggerPants", "Pants", "Slacks", "Skirt", "Loafers", "Sneakers", "Shoes", "Sleeve"
]
OUT_PDF = "confusion_matrix_ptv1_agu1.pdf"
OUT_CSV = "metrics_results_ptv1_aug1.csv"


def load_labels(path):
    """NPY 파일 로드"""
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=False)
    return arr.astype(np.int64).ravel()


def get_scene_id_from_folder(folder_name):
    """폴더명에서 scene_id 추출 (예: '1001.ply' -> '1001')"""
    if folder_name.endswith(".ply"):
        return folder_name[:-4]
    return folder_name


def calculate_metrics_from_confusion_matrix(cm):
    """
    Confusion matrix로부터 다양한 메트릭 계산
    Returns: dict with metrics for each class
    """
    n_classes = cm.shape[0]
    metrics = {
        'class_name': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'tn': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'iou': [],
        'accuracy': [],
        'support': []
    }

    for i in range(n_classes):
        # True Positive, False Positive, False Negative, True Negative
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # 다른 클래스를 i로 예측
        fn = cm[i, :].sum() - tp  # i를 다른 클래스로 예측
        tn = cm.sum() - tp - fp - fn

        # Precision, Recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # IoU (Intersection over Union)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        # Per-class accuracy
        accuracy = (tp + tn) / cm.sum() if cm.sum() > 0 else 0.0

        # Support (number of ground truth samples)
        support = cm[i, :].sum()

        metrics['class_name'].append(CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}")
        metrics['tp'].append(int(tp))
        metrics['fp'].append(int(fp))
        metrics['fn'].append(int(fn))
        metrics['tn'].append(int(tn))
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['iou'].append(iou)
        metrics['accuracy'].append(accuracy)
        metrics['support'].append(int(support))

    return metrics


def calculate_overall_metrics(cm, metrics_dict):
    """전체 메트릭 계산"""
    # Overall accuracy
    overall_acc = np.diag(cm).sum() / cm.sum() if cm.sum() > 0 else 0.0

    # Mean metrics (macro average)
    mean_precision = np.mean(metrics_dict['precision'])
    mean_recall = np.mean(metrics_dict['recall'])
    mean_f1 = np.mean(metrics_dict['f1_score'])
    mean_iou = np.mean(metrics_dict['iou'])

    # Weighted metrics
    supports = np.array(metrics_dict['support'])
    total_support = supports.sum()

    if total_support > 0:
        weighted_precision = np.sum(np.array(metrics_dict['precision']) * supports) / total_support
        weighted_recall = np.sum(np.array(metrics_dict['recall']) * supports) / total_support
        weighted_f1 = np.sum(np.array(metrics_dict['f1_score']) * supports) / total_support
        weighted_iou = np.sum(np.array(metrics_dict['iou']) * supports) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = weighted_iou = 0.0

    return {
        'overall_accuracy': overall_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1_score': mean_f1,
        'mean_iou': mean_iou,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1_score': weighted_f1,
        'weighted_iou': weighted_iou
    }


# [기존 코드 유지: GT와 PRED 폴더 수집 및 처리 부분]
# GT와 PRED 폴더 목록 수집
gt_scene_dirs = sorted(glob.glob(os.path.join(GT_DATA_ROOT, "*.ply")))
pred_scene_dirs = sorted(glob.glob(os.path.join(PRED_DATA_ROOT, "*.ply")))

if not gt_scene_dirs:
    print(f"[ERR] No GT scene folders found in: {GT_DATA_ROOT}", file=sys.stderr)
    sys.exit(1)

if not pred_scene_dirs:
    print(f"[ERR] No PRED scene folders found in: {PRED_DATA_ROOT}", file=sys.stderr)
    sys.exit(1)

# Scene ID 매핑 생성
gt_scene_map = {}
for gt_dir in gt_scene_dirs:
    if os.path.isdir(gt_dir):
        scene_id = get_scene_id_from_folder(os.path.basename(gt_dir))
        gt_scene_map[scene_id] = gt_dir

pred_scene_map = {}
for pred_dir in pred_scene_dirs:
    if os.path.isdir(pred_dir):
        scene_id = get_scene_id_from_folder(os.path.basename(pred_dir))
        pred_scene_map[scene_id] = pred_dir

# 공통 scene_id 찾기
common_scenes = set(gt_scene_map.keys()) & set(pred_scene_map.keys())
gt_only = set(gt_scene_map.keys()) - set(pred_scene_map.keys())
pred_only = set(pred_scene_map.keys()) - set(gt_scene_map.keys())

print(f"[INFO] Scene Analysis:")
print(f"  - GT scenes: {len(gt_scene_map)}")
print(f"  - PRED scenes: {len(pred_scene_map)}")
print(f"  - Common scenes: {len(common_scenes)}")
if gt_only:
    print(f"  - GT only: {len(gt_only)} scenes")
    print(f"    {sorted(gt_only)[:5]}{' ...' if len(gt_only) > 5 else ''}")
if pred_only:
    print(f"  - PRED only: {len(pred_only)} scenes")
    print(f"    {sorted(pred_only)[:5]}{' ...' if len(pred_only) > 5 else ''}")

if not common_scenes:
    print("[ERR] No matching scenes between GT and PRED", file=sys.stderr)
    sys.exit(1)

y_true_all, y_pred_all = [], []
missing_files = []
processed = []

for scene_id in sorted(common_scenes):
    gt_dir = gt_scene_map[scene_id]
    pred_dir = pred_scene_map[scene_id]

    # GT와 PRED 파일 경로
    gt_path = os.path.join(gt_dir, "segment.npy")
    pred_path = os.path.join(pred_dir, "pred.npy")

    # 파일 존재 확인
    if not os.path.exists(gt_path):
        missing_files.append(f"GT:{scene_id}/segment.npy")
        print(f"[WARN] Missing GT file: {gt_path}")
        continue

    if not os.path.exists(pred_path):
        missing_files.append(f"PRED:{scene_id}/pred.npy")
        print(f"[WARN] Missing PRED file: {pred_path}")
        continue

    # 라벨 로드
    gt = load_labels(gt_path)
    pr = load_labels(pred_path)

    if gt is None or pr is None:
        print(f"[WARN] Failed to load files for scene: {scene_id}")
        continue

    # 길이 체크
    if gt.shape != pr.shape:
        n = min(gt.size, pr.size)
        print(f"[WARN] Length mismatch: {scene_id}  gt={gt.size}, pred={pr.size} -> truncate to {n}")
        gt, pr = gt[:n], pr[:n]

    # Ignore index 처리
    if IGNORE_INDEX is not None:
        m = (gt != IGNORE_INDEX)
        gt, pr = gt[m], pr[m]

    y_true_all.append(gt)
    y_pred_all.append(pr)
    processed.append(scene_id)

# 처리 결과 출력
print(f"\n[INFO] Processing Summary:")
print(f"  - Successfully processed: {len(processed)}/{len(common_scenes)} scenes")
if missing_files:
    print(f"  - Missing files: {len(missing_files)}")
    for mf in missing_files[:10]:
        print(f"    {mf}")
    if len(missing_files) > 10:
        print(f"    ... and {len(missing_files) - 10} more")

if not y_true_all:
    print("[ERR] No valid scene pairs found", file=sys.stderr)
    sys.exit(1)

# 모든 데이터 연결
y_true = np.concatenate(y_true_all, axis=0)
y_pred = np.concatenate(y_pred_all, axis=0)

print(f"\n[INFO] Total points for evaluation: {len(y_true):,}")

# ========= Confusion Matrix 계산 =========
labels = list(range(NUM_CLASSES))
cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)

# ========= 메트릭 계산 =========
class_metrics = calculate_metrics_from_confusion_matrix(cm)
overall_metrics = calculate_overall_metrics(cm, class_metrics)

# ========= 결과 출력 =========
print("\n" + "=" * 80)
print("EVALUATION METRICS")
print("=" * 80)

print(f"\n[OVERALL METRICS]")
print(f"  Overall Accuracy:     {overall_metrics['overall_accuracy']:.4f}")
print(f"  Mean IoU (mIoU):      {overall_metrics['mean_iou']:.4f}")
print(f"  Mean Precision:       {overall_metrics['mean_precision']:.4f}")
print(f"  Mean Recall:          {overall_metrics['mean_recall']:.4f}")
print(f"  Mean F1-Score:        {overall_metrics['mean_f1_score']:.4f}")
print(f"\n[WEIGHTED METRICS]")
print(f"  Weighted IoU:         {overall_metrics['weighted_iou']:.4f}")
print(f"  Weighted Precision:   {overall_metrics['weighted_precision']:.4f}")
print(f"  Weighted Recall:      {overall_metrics['weighted_recall']:.4f}")
print(f"  Weighted F1-Score:    {overall_metrics['weighted_f1_score']:.4f}")

print(f"\n[PER-CLASS METRICS]")
print(f"{'Class':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print("-" * 70)
for i in range(len(class_metrics['class_name'])):
    print(f"{class_metrics['class_name'][i]:<15} "
          f"{class_metrics['iou'][i]:>8.4f} "
          f"{class_metrics['precision'][i]:>10.4f} "
          f"{class_metrics['recall'][i]:>8.4f} "
          f"{class_metrics['f1_score'][i]:>8.4f} "
          f"{class_metrics['support'][i]:>8}")

# ========= CSV 저장 =========
# Per-class metrics DataFrame
df_class = pd.DataFrame(class_metrics)
df_class = df_class.round(4)

# Overall metrics DataFrame
df_overall = pd.DataFrame([overall_metrics])
df_overall = df_overall.round(4)

# CSV 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"metrics_{timestamp}.csv"

with open(csv_filename, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("OVERALL METRICS\n")
    f.write("=" * 50 + "\n")
    df_overall.to_csv(f, index=False)
    f.write("\n")
    f.write("=" * 50 + "\n")
    f.write("PER-CLASS METRICS\n")
    f.write("=" * 50 + "\n")
    df_class.to_csv(f, index=False)

print(f"\n[OK] Metrics saved to: {csv_filename}")

# Excel 파일로도 저장 (옵션)
excel_filename = f"metrics_{timestamp}.xlsx"
with pd.ExcelWriter(excel_filename) as writer:
    df_overall.to_excel(writer, sheet_name='Overall', index=False)
    df_class.to_excel(writer, sheet_name='Per-Class', index=False)
print(f"[OK] Metrics saved to: {excel_filename}")

# ========= Confusion Matrix Visualization (기존 코드) =========
row_sum = cm.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
cm_norm = cm / row_sum

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(np.arange(NUM_CLASSES))
ax.set_yticks(np.arange(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=60, ha='right', fontsize=9)
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("Ground Truth Label", fontsize=11)
ax.set_title(f"Confusion Matrix (row-normalized)\n"
             f"OA: {overall_metrics['overall_accuracy']:.3f}, mIoU: {overall_metrics['mean_iou']:.3f}, "
             f"mF1: {overall_metrics['mean_f1_score']:.3f}",
             fontsize=12)

# 셀에 값 표시
th = cm_norm.max() * 0.5
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        v = cm_norm[i, j]
        if v > 0.01:
            ax.text(j, i, f"{v:.2f}",
                    va="center", ha="center", fontsize=7,
                    color=("white" if v > th else "black"))

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Confusion matrix saved: {OUT_PDF}")