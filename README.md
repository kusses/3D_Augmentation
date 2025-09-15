# Fashion 3D Data Augmentation — Coord & Color

Tools for preprocessing and augmenting **3D point clouds (.ply)** for model training. The toolkit covers:

- PLY → NumPy conversion (coordinates, colors, semantic segment labels)
- Color transformations (channel swap, inversion, randomized channels)
- **SMOTE3D**-based coordinate augmentation for minority classes (optionally with color changes)
- Visual inspection of original vs. augmented point clouds
- Dataset packaging: moving/copying files, generating train/val/test splits, and creating folder layouts for downstream models (e.g., Pointcept)

---

## Table of Contents
1. [Features](#features)
2. [Pipeline at a Glance](#pipeline-at-a-glance)
3. [Repository Overview](#repository-overview)
4. [Installation & Requirements](#installation--requirements)
5. [Data Expectations](#data-expectations)
6. [Quick Start](#quick-start)
7. [Detailed Usage](#detailed-usage)
   - [1) PLY → NumPy](#1-ply--numpy)
   - [2) Augmentation](#2-augmentation)
   - [3) Visualization](#3-visualization)
   - [4) Packaging (Move/Split/Create)](#4-packaging-movesplitcreate)
8. [Configuration Notes & Best Practices](#configuration-notes--best-practices)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)
11. [License](#license)

---

## Features
- **Structured preprocessing**: cleanly separates **XYZ coordinates**, **RGB colors**, and **segment labels** into NumPy arrays.
- **Flexible augmentation modes**:
  - `color` — color-only transforms (no geometry changes)
  - `coords` — coordinate-only augmentation with **SMOTE3D** (no color changes)
  - `both` — coordinate augmentation **and** color changes
- **Class balancing**: SMOTE3D focuses on minority classes to mitigate label imbalance.
- **Round-trip conversion**: convert NumPy back to PLY for inspection or downstream tools.
- **Turnkey packaging**: move/copy augmented data to target layouts, generate split files, and create train/eval/test folders according to plan.

---

## Pipeline at a Glance
```
[ Raw .ply ]
     │
     ▼  Dataprep_ply_to_numpy.py
[ coords.npy | colors.npy | segments.npy ]
     │
     ├─► main_{file|folder|folder_plan}.py --mode color    (Color-only)
     ├─► main_{file|folder|folder_plan}.py --mode coords   (Coords-only via SMOTE3D)
     └─► main_{file|folder|folder_plan}.py --mode both     (Coords+Color)
     │
     ▼  visualization3d.py / main.py --operation both
[ Original vs. Augmented PLY Preview ]
     │
     ▼  Move_augfile.py → data_split_trainval.py / data_aug_split_trainval.py → Create_traineval_folder.py
[ Ready-to-train Dataset Layout ]
```

---

## Repository Overview

### `Dataprep_ply_to_numpy.py`
**Purpose**: Convert PLY to NumPy arrays (XYZ, RGB, segment labels).

**Key functions**
- `read_plymesh(filepath)`: Reads a PLY file and returns a NumPy array.
- `vertex_normal(coords, k=10)`: Computes per-vertex normals from coordinates.
- `scene_process(scene_id, split_name, mesh_path, output_path, parse_normals)`: Scene-level processing and saving.

> **Note**: This script has not been fully refactored for this project yet. Minor adjustments may be required.

---

### `ColorChange.py`
**Purpose**: Apply color transformations to point clouds.

**Key functions**
- `change_color(color)`: Changes a single color via predefined channel swapping/inversion.
- `change_color_class()`: Applies the color transform to an entire class.

---

### `SMOTE3D.py`
**Purpose**: Augment 3D data by generating new coordinates (and optionally colors) using a SMOTE-style approach.

**Key functions**
- `augment_coordinates(coords)`: Shifts/interpolates coordinates to create new points.
- `augment_colors(colors)`: Inverts and randomizes channels.
- `augment_minority_class()`: Applies SMOTE to the minority class.

---

### `visualization3d.py`
**Purpose**: Visualize original and augmented clouds; convert NumPy back to PLY.

**Key functions**
- `visualize_ply(file_path)`: Renders a point cloud from a PLY file.
- `numpy_to_ply(filename, coord_file, color_file, segment_file)`: Converts NumPy arrays back to PLY.

---

## Installation & Requirements
- **Python**: 3.9+ (3.10 recommended)
- **Python packages**:
  - Core: `numpy`, `scipy`, `tqdm`
  - PLY I/O: `plyfile` *or* `open3d`
  - Visualization (optional): `open3d`
  - (Optional) SMOTE helpers: `scikit-learn` if you plan to experiment with external SMOTE variants

```bash
# Example
pip install numpy scipy tqdm plyfile open3d scikit-learn
```

> If `requirements.txt` is provided, install via `pip install -r requirements.txt`.

---

## Data Expectations
- **Input**: one or more `.ply` files per scene.
- **Output (from dataprep)**: separate NumPy arrays, typically
  - `coords.npy` — shape `(N, 3)`, float XYZ
  - `colors.npy` — shape `(N, 3)`, uint8 RGB
  - `segments.npy` — shape `(N,)`, int class/segment id
- **Color channel order**: verify whether your pipeline expects **RGB** or a nonstandard order. The original source mentions extracting as **(G, R, B)**. Confirm and keep it consistent through augmentation and export.

---

## Quick Start

### 1) Convert PLY to NumPy
Extract coordinates, colors, and segment labels into separate arrays.

```bash
python Dataprep_ply_to_numpy.py   --dataset_root <path_to_ply_files>   --output_root  <output_folder>
```

### 2) Run Augmentation
There are three entry types and three modes:

- **Entry type** (`{type}`): `file`, `folder`, `folder_plan`
- **Mode** (`--mode`):
  - `color`  — color-only
  - `coords` — coordinates-only (SMOTE3D)
  - `both`   — coordinates + color

```bash
# Color only
python main_{type}.py --mode color

# Coordinates only (SMOTE3D); no color changes
python main_{type}.py --mode coords

# Both coordinates (SMOTE3D) and color
python main_{type}.py --mode both
```

### 3) Visualize
Compare original and augmented samples side-by-side.

```bash
python main.py --operation both
```

### 4) Package for Training
Move/copy augmented files, generate split files, and build train/eval/test folders.

```bash
# (A) Move augmented files into model-specific folder layouts
python Move_augfile.py

# (B) Generate split txt files
python data_split_trainval.py       # for original
python data_aug_split_trainval.py   # for augmented

# (C) Create train/eval/test folders according to your plan
python Create_traineval_folder.py
```

---

## Detailed Usage

### 1) PLY → NumPy
- Organize raw `.ply` files under `--dataset_root`.
- Run `Dataprep_ply_to_numpy.py` to produce `coords.npy`, `colors.npy`, and `segments.npy` under `--output_root` (per scene id).
- If normals are required, enable the relevant flag (see script’s `--help`).

**Example**
```bash
python Dataprep_ply_to_numpy.py   --dataset_root ./data/ply   --output_root  ./data/npy
```

---

### 2) Augmentation
**Entry types**
- `main_file.py` — process a single scene/file.
- `main_folder.py` — process every eligible file in a directory.
- `main_folder_plan.py` — process a curated list/plan (e.g., subset selection, pairing, or mapping). The exact plan format is defined in the script; adjust to your project’s convention.

**Modes**
- `color`: apply channel swaps/inversions; labels remain unchanged.
- `coords`: apply SMOTE3D coordinate synthesis/perturbation to minority classes.
- `both`: combine `coords` + `color`.

> **Tip**: For severe class imbalance, start with `coords` and verify spatial plausibility before enabling `both`.

**Examples**
```bash
# Folder-level color augmentation
python main_folder.py --mode color

# File-level coordinate augmentation only
python main_file.py --mode coords

# Folder-plan, both coordinates+color
python main_folder_plan.py --mode both
```

---

### 3) Visualization
Use `visualization3d.py` and/or the convenience entrypoint `main.py` to render PLY.

```bash
# Displays original + augmented
python main.py --operation both
```

Additionally, you can reconstruct PLY from NumPy for external viewers:
```python
# Inside visualization3d.py
numpy_to_ply(
  filename,         # output .ply path
  coord_file,       # path to coords.npy
  color_file,       # path to colors.npy
  segment_file      # path to segments.npy
)
```

---

### 4) Packaging (Move/Split/Create)
Some frameworks (e.g., **Pointcept**) expect a specific layout. Use the helper scripts to prepare datasets.

**Move augmented files**
Set paths inside `Move_augfile.py`:
```python
# Example
source_folder    = "/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_aug"
destination_root = "/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_all"
```
Run:
```bash
python Move_augfile.py
```

**Create split text files**
```bash
# For original data
python data_split_trainval.py

# For augmented data
python data_aug_split_trainval.py
```

**Create train/eval/test folders**
```bash
python Create_traineval_folder.py
```

> The “folder plan” determines how scenes are distributed into train/eval/test. Adjust the script to your desired policy (e.g., per-scene grouping, stratified per-class, etc.).

---

## Configuration Notes & Best Practices
- **Coordinate/label integrity**: After augmentation, verify that the **N** points in `coords` still align 1:1 with `colors` and `segments`.
- **Color ranges**: Keep color arrays in `uint8 [0, 255]` unless your pipeline expects normalized floats. Enforce dtype/range after transforms.
- **Class imbalance**: Prefer applying SMOTE3D to minority classes only; avoid exploding majority classes.
- **Normals**: If you compute normals, consider normalization and smoothing to stabilize downstream geometry features.
- **Reproducibility**: If available, set a **random seed** in your augmentation scripts to get deterministic results.
- **Performance**: Process large scenes in chunks or with multiprocessing. Avoid unnecessary copies of large arrays.
- **Validation**: Always preview a few scenes after each augmentation mode change.

---

## Troubleshooting
- **Colors look “off”**: Confirm channel order (RGB vs. GRB) and dtype. A common symptom is tinted outputs.
- **Misaligned labels**: Ensure that any point-wise filtering or reindexing updates all arrays consistently.
- **Excessive point drift** *(coords mode)*: Reduce step size or neighborhood `k` in SMOTE3D; constrain augmentation to local neighborhoods.
- **Viewer fails to open PLY**: Rebuild PLY via `numpy_to_ply` to ensure headers/format are correct for your viewer.
- **Folder layout mismatch**: Double-check `Move_augfile.py` destination structure and that `scene_id.ply` is placed as required by the target model.

---

## FAQ
**Q. Do I have to change colors when using SMOTE3D?**  
A. No. Use `--mode coords` if you only want geometric augmentation.

**Q. Can I visualize only the augmented set?**  
A. Yes—point the visualization entrypoints to augmented files only, or export augmented NumPy to PLY and open in your viewer.

**Q. What’s the recommended starting recipe for imbalance?**  
A. Begin with `--mode coords` for minority classes, validate geometry, then optionally enable `--mode both` to introduce color diversity.

---

## License
Specify your license here (e.g., MIT, Apache-2.0). If this is internal, add an INTERNAL USE ONLY notice.
