import numpy as np
from scipy.stats import wilcoxon
import pandas as pd


def compute_per_scene_miou(predictions, ground_truth, scene_ids):
    """
    Compute mIoU for each scene separately
    """
    miou_per_scene = {}

    for scene_id in np.unique(scene_ids):
        # Select points belonging to this scene only
        mask = (scene_ids == scene_id)
        pred_scene = predictions[mask]
        gt_scene = ground_truth[mask]

        # Calculate mIoU (average of per-class IoU)
        num_classes = len(np.unique(gt_scene))
        ious = []

        for class_id in np.unique(gt_scene):
            if class_id == 0:  # Exclude background
                continue

            pred_mask = (pred_scene == class_id)
            gt_mask = (gt_scene == class_id)

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            if union > 0:
                iou = intersection / union
                ious.append(iou)

        miou_per_scene[scene_id] = np.mean(ious) if ious else 0.0

    return miou_per_scene


# Practical usage example
# 1. Run predictions with each method
predictions_smote3d = model_smote3d.predict(test_data)
predictions_mix3d = model_mix3d.predict(test_data)

# 2. Compute per-scene mIoU
miou_smote3d_dict = compute_per_scene_miou(
    predictions_smote3d,
    ground_truth,
    scene_ids
)
miou_mix3d_dict = compute_per_scene_miou(
    predictions_mix3d,
    ground_truth,
    scene_ids
)

# 3. Convert to lists (in the same order)
scenes = sorted(miou_smote3d_dict.keys())
miou_smote3d = [miou_smote3d_dict[s] for s in scenes]
miou_mix3d = [miou_mix3d_dict[s] for s in scenes]

# 4. Perform Wilcoxon test
statistic, p_value = wilcoxon(miou_smote3d, miou_mix3d)

print(f"Number of scenes: {len(scenes)}")
print(f"SMOTE3D mean mIoU: {np.mean(miou_smote3d):.4f}")
print(f"Mix3D mean mIoU: {np.mean(miou_mix3d):.4f}")
print(f"W statistic: {statistic}")
print(f"p-value: {p_value:.4f}")

# 5. Save results
results_df = pd.DataFrame({
    'scene_id': scenes,
    'miou_smote3d': miou_smote3d,
    'miou_mix3d': miou_mix3d,
    'difference': np.array(miou_smote3d) - np.array(miou_mix3d)
})

print("\nTop 5 scenes:")
print(results_df.head())

# 6. Proportion of improved scenes
improved = (results_df['difference'] > 0).sum()
print(f"\nScenes where SMOTE3D is better: {improved}/{len(scenes)} ({100 * improved / len(scenes):.1f}%)")