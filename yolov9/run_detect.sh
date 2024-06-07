#!/bin/bash

# Define the parent directory containing the subdirectories
PARENT_DIR="../32_33_AI_CUP_testdataset/32_33_AI_CUP_testdataset/AI_CUP_testdata/images"
WEIGHTS_PATH="runs/train/yolov9-c-small-lr/weights/best.pt"
DEVICE="0"


# Check if the parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
  echo "Error: Parent directory '$PARENT_DIR' does not exist."
  exit 1
fi

# Get a sorted list of subdirectories
SUBDIRS=($(ls -d "$PARENT_DIR"/* | sort))

# Iterate over each sorted subdirectory
for INDEX in "${!SUBDIRS[@]}"; do
  SUBDIR="${SUBDIRS[$INDEX]}"
  if [ -d "$SUBDIR" ]; then
    NAME=$(basename "$SUBDIR")
    # Determine image size based on index (odd or even)
    if (( INDEX % 2 == 0 )); then
      IMG_SIZE=1280
    else
      IMG_SIZE=640
    fi

    echo "Processing directory: $SUBDIR with name: $NAME and image size: $IMG_SIZE"

    # Run the Python detect.py script for each subdirectory
    python detect.py --source "$SUBDIR" --img "$IMG_SIZE" --device "$DEVICE" --weights "$WEIGHTS_PATH" --name "$NAME" --save-txt --save-conf --nosave
  fi
done