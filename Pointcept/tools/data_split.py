# -*- coding: utf-8 -*-
"""
Group-wise multilabel stratified split (70/20/10) for folders named:
  - "<base>.ply" (original) or "<base>_<aug>.ply" (augmentation)
Labels are derived automatically from segment npy files inside each sample folder.
Conversation in Korean; code comments in English per user's preference.
"""

import os
import re
import glob
import json
import random
import shutil
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# ========= User Config =========
DATA_ROOT = Path("/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_aug1_model/volmeda/data")     # root containing sample folders: "123.ply", "123_1.ply", ...
OUT_ROOT  = Path("/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_aug1_model/volmeda")      # where train/val/test will be created
RATIOS    = (0.7, 0.2, 0.1)                   # train, eval, test
SEED      = 42
COPY_MODE = "symlink"                          # "symlink" | "copy"
# If you already have a labels.csv per base_id, set USE_LABELS_CSV=True and point CSV path.
USE_LABELS_CSV = False
LABEL_CSV = Path("/path/to/labels.csv")        # columns: base_id,labels (labels separated by ';')

# Given label map (edit if needed)
LABEL_TO_NAMES = {
    0: 'Beanie', 1: 'Cap', 2: 'Sweatshirt', 3: 'Shirt', 4: 'Knitwear', 5: 'Jeans',
    6: 'JoggerPants', 7: 'Pants', 8: 'Slacks', 9: 'Skirt', 10: 'Loafers',
    11: 'Sneakers', 12: 'Shoes', 13: 'Backpack', 14: 'Sleeve', 15: 'Boots', 16: 'Mannequin'
}
# =================================

random.seed(SEED)
np.random.seed(SEED)

# ---- 1) Scan sample folders and group by base_id ----
pat = re.compile(r"^(\d+)(?:_(\d+))?\.ply$")  # matches "123.ply" or "123_7.ply"
all_items = []
for name in os.listdir(DATA_ROOT):
    p = DATA_ROOT / name
    if not p.is_dir():
        continue
    m = pat.match(name)
    if not m:
        continue
    base_id = m.group(1)
    all_items.append((base_id, name))

grouped = defaultdict(list)
for base_id, foldername in all_items:
    grouped[base_id].append(foldername)

if not grouped:
    raise SystemExit("No valid sample folders found. Check DATA_ROOT and folder naming pattern.")

# ---- 2) Build multilabel set per base_id ----
def find_segment_npy(sample_dir: Path):
    """
    Search for a segment npy inside the sample folder.
    Common patterns (customize if your repo differs):
      - segment.npy, _segment.npy
      - segment/*.npy
      - labels.npy
    Returns first hit or None.
    """
    candidates = []
    # direct files
    candidates += glob.glob(str(sample_dir / "segment.npy"))
    candidates += glob.glob(str(sample_dir / "_segment.npy"))
    # under segment/ subdir
    candidates += glob.glob(str(sample_dir / "segment" / "*.npy"))
    # any npy named with 'segment'
    candidates += glob.glob(str(sample_dir / "*segment*.npy"))
    candidates = [c for c in candidates if os.path.isfile(c)]
    # de-dup while keeping order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return Path(uniq[0]) if uniq else None

def labels_from_segment(sample_dir: Path):
    """
    Load unique label ids present in the segment array.
    Returns a set of valid class ids (keys of LABEL_TO_NAMES).
    """
    p = find_segment_npy(sample_dir)
    if p is None:
        return set()
    try:
        seg = np.load(p)
        # seg can be int array per-point; flatten to unique ids
        uniq = np.unique(seg).tolist()
        # keep only known labels
        valid = set([int(x) for x in uniq if int(x) in LABEL_TO_NAMES])
        return valid
    except Exception as e:
        print(f"[WARN] failed to read segment at {p}: {e}")
        return set()

base_to_labels = {}

if USE_LABELS_CSV:
    import csv
    with open(LABEL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_map = {row["base_id"].strip(): [s.strip() for s in row["labels"].split(";") if s.strip()]}
        # If label names are strings, map them back to ids if possible; else keep names.
    # For simplicity, assume CSV already lists numeric ids separated by ';'
    tmp = {}
    with open(LABEL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row["base_id"].strip()
            labs = [s.strip() for s in row["labels"].split(";") if s.strip()]
            # convert to ints if all digits; else keep as names
            labs_int = []
            for s in labs:
                if s.isdigit():
                    labs_int.append(int(s))
                else:
                    # map by name -> id if available
                    inv = {v:k for k,v in LABEL_TO_NAMES.items()}
                    if s in inv: labs_int.append(inv[s])
            base_to_labels[bid] = set([l for l in labs_int if l in LABEL_TO_NAMES])
else:
    # derive labels by union across all samples under each base_id (original + augmentations)
    for bid, folders in grouped.items():
        union_labels = set()
        for fname in folders:
            sdir = DATA_ROOT / fname
            union_labels |= labels_from_segment(sdir)
        if not union_labels:
            print(f"[WARN] base_id {bid} has no segment labels found. Excluding from split.")
            continue
        base_to_labels[bid] = union_labels

# keep only groups that have labels
base_ids = sorted([bid for bid in grouped.keys() if bid in base_to_labels and base_to_labels[bid]])
if not base_ids:
    raise SystemExit("No groups with labels. Check your segment npy discovery patterns.")

# ---- 3) Multilabel stratified split (group-wise) ----
Y = [sorted(list(base_to_labels[bid])) for bid in base_ids]

def split_iterative(base_ids, Y, ratios, seed=SEED):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except Exception as e:
        raise RuntimeError("iterstrat not installed. pip install iterstrat") from e

    import numpy as np
    X = np.arange(len(base_ids)).reshape(-1, 1)

    # First: Train vs Temp (Eval+Test)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1.0 - ratios[0]), random_state=seed)
    train_idx, temp_idx = next(msss1.split(X, Y))

    # Second: Temp -> Eval vs Test with proportional ratio
    eval_ratio = ratios[1] / (ratios[1] + ratios[2])
    X_temp = X[temp_idx]
    Y_temp = [Y[i] for i in temp_idx]
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1.0 - eval_ratio), random_state=seed)
    eval_rel, test_rel = next(msss2.split(X_temp, Y_temp))
    eval_idx = temp_idx[eval_rel]
    test_idx = temp_idx[test_rel]
    return train_idx.tolist(), eval_idx.tolist(), test_idx.tolist()

def split_greedy(base_ids, Y, ratios, seed=SEED):
    random.seed(seed)
    n = len(base_ids)
    target = [int(round(n * r)) for r in ratios]
    # fix rounding
    while sum(target) > n:
        target[0] -= 1
    while sum(target) < n:
        target[0] += 1

    buckets = [[], [], []]
    counters = [Counter(), Counter(), Counter()]

    def imbalance_score(cnt: Counter):
        # simple spread metric: max - min
        return (max(cnt.values()) - min(cnt.values())) if cnt else 0

    order = list(range(n))
    random.shuffle(order)
    for i in order:
        lbls = Y[i]
        # pick the bucket with room that yields smallest imbalance
        best_j, best_s = None, None
        for j in range(3):
            if len(buckets[j]) >= target[j]:
                continue
            tmp = counters[j].copy()
            tmp.update(lbls)
            s = imbalance_score(tmp)
            if (best_s is None) or (s < best_s):
                best_s, best_j = s, j
        if best_j is None:
            # all full? put into currently smallest imbalance
            best_j = min(range(3), key=lambda j: imbalance_score(counters[j] + Counter(lbls)))
        buckets[best_j].append(i)
        counters[best_j].update(lbls)
    return buckets[0], buckets[1], buckets[2]

try:
    idx_tr, idx_ev, idx_te = split_iterative(base_ids, Y, RATIOS, seed=SEED)
except RuntimeError as e:
    print(f"[INFO] {e}. Falling back to greedy balancing.")
    idx_tr, idx_ev, idx_te = split_greedy(base_ids, Y, RATIOS, seed=SEED)

BTR = [base_ids[i] for i in idx_tr]
BEV = [base_ids[i] for i in idx_ev]
BTE = [base_ids[i] for i in idx_te]

print(f"Groups => train:{len(BTR)} val:{len(BEV)} test:{len(BTE)} / total:{len(base_ids)}")

# ---- 4) Materialize split as symlinks or copies ----
for split in ["train", "val", "test"]:
    (OUT_ROOT / split).mkdir(parents=True, exist_ok=True)

def place_group(bid, split):
    for foldername in grouped[bid]:
        src = DATA_ROOT / foldername
        dst = OUT_ROOT / split / foldername
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if COPY_MODE == "copy":
            shutil.copytree(src, dst)
        else:
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError as e:
                print(f"[WARN] symlink failed ({e}); fallback to copy for {dst.name}")
                shutil.copytree(src, dst)

for bid in BTR: place_group(bid, "train")
for bid in BEV: place_group(bid, "val")
for bid in BTE: place_group(bid, "test")

# ---- 5) Report label distribution per split ----
def summarize(name, bids):
    cnt = Counter()
    for bid in bids:
        cnt.update(base_to_labels[bid])
    total_labels = sum(cnt.values())
    print(f"\n[{name}] groups={len(bids)} unique_labels={len(cnt)} total_label_counts={total_labels}")
    for lid, c in cnt.most_common():
        print(f"  {lid:>2} ({LABEL_TO_NAMES[lid]}): {c}")

summarize("train", BTR)
summarize("val",  BEV)
summarize("test",  BTE)

# Optionally, dump the assignment to JSON for reproducibility
assign = {
    "train": BTR,
    "val": BEV,
    "test": BTE,
    "ratios": RATIOS,
    "seed": SEED
}
(OUT_ROOT / "split_assignment.json").write_text(json.dumps(assign, indent=2), encoding="utf-8")
print("\nDone.")
