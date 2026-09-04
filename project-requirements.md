# Project: Synthetic-to-Real 3D Pose & Shape Estimation

## Goal
Build an end-to-end pipeline that (1) procedurally generates a synthetic 3D
dataset in Blender, (2) trains a PyTorch model to estimate object pose/depth/
mesh from rendered 2D images, and (3) demonstrates results with a rendered
comparison reel. This project is designed to showcase skills relevant to
computer graphics + computer vision + deep learning roles (e.g. Autodesk AI
Dev), specifically: 3D geometry, Blender scripting, PyTorch, and (optionally)
C++ performance code and Adobe Creative Cloud tooling.

## Target skills to demonstrate
- 3D geometry / computer graphics fundamentals (camera projection, rotations,
  quaternions, mesh representations)
- Blender Python API (`bpy`) for procedural scene/dataset generation
- PyTorch (preferred) model for a vision task (pose estimation, depth
  prediction, or mesh reconstruction)
- Strong Python engineering; optional C/C++ component for a performance-
  critical utility
- Optional: Houdini procedural variation, Adobe Substance 3D texturing,
  After Effects for the final demo reel

## Deliverables
1. A Blender Python script that procedurally renders a labeled synthetic
   dataset (images + ground-truth pose/depth/mesh annotations).
2. A PyTorch training pipeline that consumes this dataset and predicts pose
   (or depth, or mesh) from a single RGB image.
3. An evaluation script with quantitative metrics (e.g. rotation error,
   depth RMSE, or Chamfer distance) and qualitative visualizations
   (predicted vs. ground truth, rendered side-by-side).
4. A short README explaining the pipeline, design decisions, and results,
   suitable for a GitHub portfolio.
5. (Stretch) A small C++ or CUDA utility for a performance-sensitive step
   (e.g. fast mesh preprocessing or a custom PyTorch C++ extension).
6. (Stretch) A demo reel/video assembled with Adobe tools (Substance 3D for
   texturing synthetic assets, After Effects or Premiere for compositing
   results).

## Phase 1 — Synthetic Data Generation (Blender)
- Set up a Blender scene with 1-3 object categories (start simple: e.g.
  cubes/spheres/mugs, or download free CC0 3D models).
- Write a `bpy` Python script that, for each render:
  - Randomizes camera position/orientation around the object (on a sphere,
    varying elevation/azimuth/distance).
  - Randomizes lighting (HDRI environment maps or point lights) and, if
    time allows, materials/textures.
  - Renders an RGB image and saves ground-truth labels: camera intrinsics/
    extrinsics, object 6-DoF pose (rotation + translation), and/or a depth
    map (Blender's Z-pass).
  - Optionally exports the object mesh/point cloud for mesh-based tasks.
- Generate a dataset of at least 2,000-5,000 rendered images with paired
  labels. Split into train/val/test.
- Output format: images as PNG, labels as JSON or NPZ per sample.

## Phase 2 — Model Training (PyTorch)
- Choose ONE primary task to keep scope manageable:
  - **Pose estimation**: predict 6-DoF rotation + translation from a single
    image (classification/regression on rotation representation, e.g.
    quaternion or 6D rotation representation + L1/L2 translation loss).
  - **Depth estimation**: predict a per-pixel depth map from a single RGB
    image (encoder-decoder CNN, L1 loss on depth pixels).
  - **Mesh reconstruction**: use PyTorch3D to predict/deform a mesh from an
    image, supervised with Chamfer distance / mesh losses against the
    ground-truth mesh.
- Build a standard PyTorch pipeline: `Dataset`/`DataLoader`, model
  (ResNet/CNN backbone + task-specific head), training loop, checkpointing,
  logging (e.g. simple CSV or TensorBoard).
- Train on the synthetic dataset from Phase 1. Track train/val loss and the
  task-specific metric.
- If time allows, test generalization on a small set of real photos (even
  phone photos of a similar object) to demonstrate the "synthetic-to-real"
  angle — this doesn't need to work perfectly, just be discussed honestly
  in the README.

## Phase 3 — Evaluation & Visualization
- Compute quantitative metrics on the held-out test set (rotation error in
  degrees, depth RMSE, or Chamfer distance, depending on task).
- Generate qualitative comparison images: input image, predicted output
  overlaid or rendered next to ground truth.
- Re-render a few predictions back through Blender (e.g. render the object
  at the predicted pose vs. ground-truth pose side by side) to make results
  visually intuitive for a portfolio/demo.

## Phase 4 — Polish & Packaging
- Write a clear README: problem statement, pipeline diagram/description,
  dataset stats, model architecture, results (numbers + images), and
  honest discussion of limitations (sim-to-real gap, dataset size, etc.).
- Clean up repo structure:
  ```
  /blender_gen/      Blender Python scripts for dataset generation
  /data/             (gitignored) generated dataset
  /model/            PyTorch model, dataset loader, training/eval scripts
  /results/          sample outputs, metrics, comparison images
  README.md
  ```
- (Stretch) Record a short demo video/reel and edit with Adobe tools.
- (Stretch) Isolate one performance-sensitive function (e.g. point cloud
  nearest-neighbor search) and reimplement it as a C++/CUDA PyTorch
  extension for the "strong C/C++" qualification.

## Tech stack
- Blender 4.x with Python API (`bpy`)
- Python 3.10+, PyTorch (preferred) or TensorFlow
- Optional: PyTorch3D (mesh tasks), OpenCV, NumPy, Matplotlib
- Optional: C++/CUDA for a custom extension
- Optional: Houdini for procedural asset variation, Adobe Substance 3D /
  After Effects for texturing and final presentation

## Constraints / notes for implementation
- Prioritize getting a simple end-to-end version working first (small
  dataset, simple model, single task) before adding stretch goals.
- Keep the Blender generation script and the PyTorch training code
  decoupled — dataset generation should be runnable headless
  (`blender --background --python generate.py`).
- Document every design decision that trades off scope vs. time, since the
  README's honesty about limitations matters as much as the results for
  portfolio purposes.
