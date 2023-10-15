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
    # uncomment these lines to include plane and box queries
    # plane="--object_name ${object} --query plane --dim ${dim}"
    # box="--object_name ${object} --query box --dim ${dim}"

    # execute python script with the given arguments
    python3 src/main.py $point
    python3 src/main.py $ray
    # uncomment these lines to include plane and box queries
    # python3 src/main.py $plane
    # python3 src/main.py $box

  done
done