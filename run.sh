#!/usr/bin/env bash

# add the root directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# initialise objects for 2D spatial queries
objects=("bunny" "chair" "dragon1" "dragon2" "house" "lucy" "star1" "star2" "teapot")

# loop over objects and dimensions
for object in "${objects[@]}"; do
  # set command line arguments for various queries
  point="--object_name ${object} --query point --dim 2"
  ray="--object_name ${object} --query ray --dim 2"
  plane="--object_name ${object} --query plane --dim 2"
  box="--object_name ${object} --query box --dim 2"

  # execute python script with the given arguments
  python3 src/run_experiments.py $point
  python3 src/run_experiments.py $ray
  python3 src/run_experiments.py $plane
  python3 src/run_experiments.py $box
done

# initialise objects for 3D spatial queries
objects=("airplane" "armchair" "bunny" "camera" "car" "dragon" "sofa" "teapot" "teddy")

# loop over objects and dimensions
for object in "${objects[@]}"; do
  # set command line arguments for various queries
  point="--object_name ${object} --query point --dim 3"
  ray="--object_name ${object} --query ray --dim 3"
  plane="--object_name ${object} --query plane --dim 3"
  box="--object_name ${object} --query box --dim 3"

  # execute python script with the given arguments
  python3 src/run_experiments.py $point
  python3 src/run_experiments.py $ray
  python3 src/run_experiments.py $plane
  python3 src/run_experiments.py $box
done

# initialise objects for 4D spatial queries
objects=("bunny" "teapot" "dragon")

# loop over objects and dimensions
for object in "${objects[@]}"; do
  # set command line arguments for various queries
  point="--object_name ${object} --query point --dim 4"
  ray="--object_name ${object} --query ray --dim 4"
  plane="--object_name ${object} --query plane --dim 4"
  box="--object_name ${object} --query box --dim 4"

  # execute python script with the given arguments
  python3 src/run_experiments.py $point
  python3 src/run_experiments.py $ray
  python3 src/run_experiments.py $plane
  python3 src/run_experiments.py $box
done

# wait for all background jobs to complete
wait

# make Table 1 from the results of the experiments above
python3 src/make_table1.py