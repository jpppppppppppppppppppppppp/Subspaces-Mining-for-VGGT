#!/bin/bash

# Root directory for generated scenes and render outputs.
TARGET_PATH="./generated_scenes"
# For plane mode, set NUM_SCENES to match the number of files in ./imgs.
# NUM_SCENES=$(find ./imgs -maxdepth 1 -type f | wc -l)
NUM_SCENES=200
SEED=19700101

# Generate scene GLBs.
## Build plane-based scenes for face anti-spoofing.
python create_planes.py --project_dir "${TARGET_PATH}" --num_scenes "${NUM_SCENES}" --seed "${SEED}"

# Render the plane-based face anti-spoofing scenes.
COUNTER=1
# Iterate over each scene subdirectory.
for SUBDIR in "${TARGET_PATH}/scenes"/*/; do
  # Strip the trailing slash from the subdirectory path.
  SUBDIR_NAME=$(basename "${SUBDIR}")

  # Build the input scene path and output directory from the subdirectory name.
  SCENE_PATH="${TARGET_PATH}/scenes/${SUBDIR_NAME}/scenes.glb"
  OUTPUT_DIR="${TARGET_PATH}/rendering/${SUBDIR_NAME}"

  # Render the current scene with the configured arguments.
  python render_plane_rgbd.py --object_path "${SCENE_PATH}" --output_dir "${OUTPUT_DIR}" --seed "${COUNTER}" --micro --render_views 42 # --rotation_limit 0.00 --step_limit 0.0000
  COUNTER=$((COUNTER + 1))
done

# Generate the texture-variation dataset.
## Use different ID values to cache distinct texture variation sets.
## For implementation details, see create_shapes_texture.py:L1208-1236.
ID=0
python create_object_texture_para.py --project_dir "${TARGET_PATH}" --num_scenes "${NUM_SCENES}" --num_project "${ID}" --seed "${SEED}"

# Generate the geometry-variation dataset.
## Use different ID values to select different geometry-related configurations.
## For implementation details, see create_object_geometry.py:L966-981.
python create_object_geometry_para.py --project_dir "${TARGET_PATH}" --num_scenes "${NUM_SCENES}" --group_idx "${ID}" --seed "${SEED}"

# Render the texture- and geometry-variation datasets.
python render_scenes_rgbd_para.py --dir_path "${TARGET_PATH}" --seed "${SEED}"

# Render the camera-variation dataset.
## Use --camera_rand to enable randomized camera targets and a wider control-point distance range.
python render_scenes_rgbd_para.py --dir_path "${TARGET_PATH}" --seed "${SEED}" --camera_rand

# Render the lighting-variation dataset.
## `seed` controls randomized lighting parameters, while `np_seed` controls camera sampling and can be kept fixed.
python render_object_rgbd_light_para.py --dir_path "${TARGET_PATH}" --seed "${SEED}" --np_seed "42"
