# %%
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from torch import nn

# %%
SEED = 42
C = 1.0  # Cost parameter for false positives
STEPS = 5000
TEST_SIZE = 16281  # Size of the standard UCI Adult test split

torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def loss(y_true, y_pred, c=1.0):
    """Cost-weighted error rate for labels in {-1, 1}."""
    false_neg = ((y_true == 1) & (y_pred == -1)).sum()
    false_pos = ((y_true == -1) & (y_pred == 1)).sum()
    return ((false_neg + c * false_pos) / y_true.numel()).item()


def load_adult():
    """Adult census data from OpenML (cached by sklearn after the first download)."""
    X, income = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    # One-hot encode; binary `sex` becomes one column. Missing values are all zeros.
    X = pd.get_dummies(X, columns=["sex"], drop_first=True, dtype=np.float32)
    X = pd.get_dummies(X, dtype=np.float32).to_numpy(np.float32)
    y = np.where(income == ">50K", -1.0, 1.0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    X_train = normalize(scaler.fit_transform(X_train))  # unit L2 norm per row
    X_test = normalize(scaler.transform(X_test))
    return X_train, X_test, y_train, y_test


# %%
def learn_mean_weights(X, y, c=1.0):
    """Mean of the label-weighted feature vectors; negatives are weighted by c."""
    weights = y if c == 1.0 else torch.where(y == -1, -c, 1.0)
    return (X * weights.unsqueeze(1)).mean(0)


def mean_classifier(X, mean_weights):
    return torch.sign(X @ mean_weights).to(torch.int64)


def self_similarity_loss_linear(x, labels, c=1.0):
    # sum_ij w_i w_j <x_i, x_j> = ||sum_i w_i x_i||^2, avoiding the N x N matrix
    weights = labels if c == 1.0 else torch.where(labels == -1, -c, 1.0)
    return (weights @ x).square().sum()


class ResNN(nn.Module):
    """Residual block, initialised to the identity map."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def initialize(self):
        hidden = self.linear_stack[0]
        nn.init.kaiming_normal_(hidden.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(hidden.bias)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        return x + self.final_layer(self.linear_stack(x))


class NormalizedStackedResNN(nn.Module):
    """Stack of residual blocks with unit L2 norm outputs."""

    def __init__(self, n_nets, input_dim, hidden_dim):
        super().__init__()
        self.nets = nn.Sequential(
            *[ResNN(input_dim, hidden_dim) for _ in range(n_nets)]
        )
        # Initialise after building all blocks, so seeded runs match the old mc.py
        for net in self.nets:
            net.initialize()

    def forward(self, x):
        return F.normalize(self.nets(x), p=2, dim=1)


# %%
# Load data
X_train_np, X_test_np, y_train_np, y_test_np = load_adult()

X_train = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
X_test = torch.as_tensor(X_test_np, dtype=torch.float32, device=device)
y_train = torch.as_tensor(y_train_np, device=device)
y_test = torch.as_tensor(y_test_np, device=device)
n_features = X_train.shape[1]

# %%
# Baseline constant classifier
constant_loss = min(
    loss(y_test, torch.full_like(y_test, label), c=C) for label in (-1, 1)
)
print("Constant Classifier Loss:", constant_loss)

# %%
# Mean classifier
mean_weights = learn_mean_weights(X_train, y_train, c=C)
print("Mean Loss:", loss(y_test, mean_classifier(X_test, mean_weights), c=C))

# %%
# Logistic regression baseline (L2 penalty, C=1)
# liblinear solver
log_reg = LogisticRegression(solver="liblinear", max_iter=10000)
log_reg.fit(X_train_np, y_train_np)
prob_pos = log_reg.predict_proba(X_test_np)[:, 1]  # P(y = +1)
predictions = torch.as_tensor(np.where(prob_pos >= C / (1 + C), 1, -1), device=device)
print("Logistic Regression Loss:", loss(y_test, predictions, c=C))

# %%
# Neural network feature map
my_nn = NormalizedStackedResNN(2, n_features, 200).to(device)

opt = torch.optim.Adam(
    my_nn.parameters(), lr=0.001, maximize=True, fused=device.type == "cuda"
)


def objective_fn(inputs, labels):
    return self_similarity_loss_linear(my_nn(inputs), labels, c=C) / len(labels) ** 2


# %%
for step in range(1, STEPS + 1):
    opt.zero_grad()
    objective = objective_fn(X_train, y_train)
    objective.backward()
    opt.step()

    if step % 500 == 0:
        print(f"Step {step}: Self-Similarity Score: {objective.item():.6f}")

# %%
with torch.inference_mode():
    mean_weights = learn_mean_weights(my_nn(X_train), y_train, c=C)
    predictions = mean_classifier(my_nn(X_test), mean_weights)
    print("NN + Mean Loss:", loss(y_test, predictions, c=C))

# %%
# Logistic regression on the NN features (same settings as the baseline)
with torch.no_grad():
    Z_train = my_nn(X_train).cpu().numpy()
    Z_test = my_nn(X_test).cpu().numpy()

nn_log_reg = LogisticRegression(solver="liblinear", max_iter=10000)
nn_log_reg.fit(Z_train, y_train_np)
prob_pos = nn_log_reg.predict_proba(Z_test)[:, 1]  # P(y = +1)
predictions = torch.as_tensor(np.where(prob_pos >= C / (1 + C), 1, -1), device=device)
print("NN + Logistic Regression Loss:", loss(y_test, predictions, c=C))

# %%
