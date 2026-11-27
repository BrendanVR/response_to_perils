# Mean classifier on the Adult census data

This example predicts whether a person's income is above $50K using the
[Adult dataset](https://www.openml.org/search?type=data&id=1590). It compares a
simple mean classifier, with and without a learned neural network feature map,
against logistic regression.

## Requirements

- Python 3.12 or newer (developed with 3.14)
- An internet connection for the first run, to download the data from OpenML
- Optional: an NVIDIA GPU. The script uses CUDA if available, otherwise the CPU,
  which will be much slower.

## Install

From the project folder:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The PyTorch download is large. For a CPU-only or specific CUDA build, follow the
[PyTorch install instructions](https://pytorch.org/get-started/locally/).

## Run

```bash
python mean_classifier_census.py
```

The file is also divided into `# %%` cells, so you can run it step by step in
VS Code's interactive window (this also needs `pip install ipykernel`).

## What it does

1. **Data:** loads Adult with `sklearn.datasets.fetch_openml`, one-hot encodes the
   categorical columns, makes a stratified train/test split (16,281 test rows),
   standardises the features and scales each row to unit L2 norm. Labels are `-1`
   for `>50K` and `+1` for `<=50K`.
2. **Constant classifier:** the better of always predicting `-1` or `+1`.
3. **Mean classifier:** predicts the sign of the inner product between a sample
   and the mean of the label-weighted training features.
4. **Logistic regression:** scikit-learn, `liblinear` solver, L2 penalty.
5. **Neural network features:** a two-block residual network with unit-norm
   output, trained with full-batch Adam to maximise the self-similarity score

   `‖(1/N) Σ wᵢ f(xᵢ)‖²`

   where `wᵢ` is the label weight and `f` is the network. This equals the average
   of `wᵢ wⱼ ⟨f(xᵢ), f(xⱼ)⟩` over all pairs, without building the N×N matrix.
6. **Evaluation on learned features:** the mean classifier and logistic regression
   are each applied to the network's output.

## Output

The script prints the self-similarity score every 500 training steps (higher is
better) and a test loss for each classifier (lower is better):

```
Constant Classifier Loss: ...
Mean Loss: ...
Logistic Regression Loss: ...
NN + Mean Loss: ...
NN + Logistic Regression Loss: ...
```

The loss is the cost-weighted error rate: `(false negatives + C × false positives) / N`,
with `+1` as the positive class. With `C = 1` it is the ordinary error rate.

## Settings

Edit these at the top of the script:

| Name | Meaning | Default |
|---|---|---|
| `SEED` | Seed for the data split and network initialisation | `42` |
| `C` | Cost of a false positive relative to a false negative | `1.0` |
| `STEPS` | Number of full-batch training steps | `5000` |
| `TEST_SIZE` | Number of test samples | `16281` |

`C` is not scikit-learn's regularisation parameter, which is also called `C` and is
left at its default of 1. The network size and learning rate are set where the
network and optimiser are created.

The split is random rather than the official UCI train/test split, so results are
not directly comparable with published figures that use the official one.
