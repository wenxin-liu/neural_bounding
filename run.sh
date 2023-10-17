#!/usr/bin/env bash

# add the root directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# initialise objects and dimensions
objects=("bunny")
dims=(2 3 4)

# loop over objects and dimensions
for object in "${objects[@]}"; do
  for dim in "${dims[@]}"; do
    # set command line arguments for various queries
    point="--object_name ${object} --query point --dim ${dim}"
    ray="--object_name ${object} --query ray --dim ${dim}"
    plane="--object_name ${object} --query plane --dim ${dim}"
    box="--object_name ${object} --query box --dim ${dim}"

    # execute python script with the given arguments
    echo "${object} ${dim}d point"
    python3 src/main.py $point

    echo "${object} ${dim}d ray"
    python3 src/main.py $ray

    echo "${object} ${dim}d plane"
    python3 src/main.py $plane

    echo "${object} ${dim}d box"
    python3 src/main.py $box
  done
done