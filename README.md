# Neural Bounding

PyTorch implementation of [Neural Bounding](https://arxiv.org/abs/2310.06822).

## Run instructions

```agsl
conda env create -f environment.yml
conda activate neural_bounding
./run.sh
```

This makes all experiments results as absolute values, per task (indicator dimension and query), method and object.

Then it calculates Table 1 of the paper using the above experiment results.

All results are saved as CSV files in `<project_root>/results`.