#!/bin/bash

# Define the target path
TARGET_PATH="./test"
NUM_SCENES=2000


# create scene glb
time python create_object_center_para.py --project_dir "$TARGET_PATH" --num_scenes "$NUM_SCENES" --seed 1


# # render scenes
# COUNTER=1
# # Iterate over each subdirectory in the target path
# for SUBDIR in "$TARGET_PATH/scenes"/*/; do
#   # Remove trailing slash from the subdirectory path
#   SUBDIR_NAME=$(basename "$SUBDIR")

#   # Define the object path and output directory based on the subdirectory name
#   SCENE_PATH="$TARGET_PATH/scenes/$SUBDIR_NAME/scenes.glb"
#   OUTPUT_DIR="$TARGET_PATH/rendering/$SUBDIR_NAME"

#   # Run the Python script with the appropriate arguments
#   python render_scenes_rgbd_scan.py --object_path "$SCENE_PATH" --output_dir "$OUTPUT_DIR" --seed "$COUNTER" --scan

#   COUNTER=$((COUNTER + 1))
# done
