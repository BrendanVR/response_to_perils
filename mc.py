import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def learn_mean_weights(X, y, c=1.0):
    if c == 1.0:
        w = (X * y.unsqueeze(1)).mean(0)
        return w
    else:
        weights = torch.where(y == -1, -c, 1.0)
        weights = weights.unsqueeze(1)
        weighted_X = X * weights
        w = weighted_X.mean(0)
        return w


def mean_classifier(X, mean_weights):
    scores = torch.matmul(X, mean_weights)
    predictions = torch.sign(scores).to(torch.int64)
    return predictions


def gaussian_mean_classifier(X_train, y_train, X_test, c=1.0):
    K = torch.cdist(X_test, X_train, p=2).pow(2)
    K = torch.exp(-0.5 * K)  # Gaussian kernel
    if c == 1.0:
        scores = torch.matmul(K, y_train.float())
    else:
        weights = torch.where(y_train == -1, -c, 1.0)
        scores = torch.matmul(K, weights)
    predictions = torch.sign(scores).to(torch.int64)
    return predictions


@torch.compile
def self_similarity_loss_linear(x, labels, c=1.0):
    dot_products = torch.matmul(x, x.T)
    if c == 1.0:
        dot_products = dot_products * torch.outer(labels, labels)
        objective = dot_products.sum()
    else:
        weights = torch.where(labels == -1, -c, 1.0)
        dot_products = dot_products * torch.outer(weights, weights)
        objective = dot_products.sum()

    return objective


@torch.compile
def self_similarity_loss_gaussian(x, labels, c=1.0):
    K = torch.cdist(x, x, p=2).pow(2)
    K = torch.exp(-0.5 * K)  # Gaussian kernel

    if c == 1.0:
        K = K * torch.outer(labels, labels)
    else:
        weights = torch.where(labels == -1, -c, 1.0)
        K = K * torch.outer(weights, weights)

    objective = K.sum()
    return objective


def compute_gaussian_distance_matrix(X, y):
    K = torch.cdist(X, X, p=2).pow(2)
    K = torch.exp(-0.5 * K)
    Y = torch.outer(y, y).float()
    K = K * Y
    D = torch.sqrt(torch.clamp(2 - 2 * K, min=0.0))
    return D.cpu().numpy()


def farthest_first_traversal_epsilon_cover(distance_matrix, epsilon, delta):
    """
    Efficient farthest-first traversal for epsilon-cover.
    Returns a dict mapping each point index to its assigned center.
    """
    n = distance_matrix.shape[0]
    centers = [0]
    assigned_center = np.zeros(n, dtype=int)
    min_distances = distance_matrix[0, :]  # Distance from first center

    max_distance = float("inf")
    k = 1

    while 2 * max_distance * (1 / np.sqrt(k) + np.sqrt(-np.log(delta) / (k))) > epsilon:
        # Find the farthest point from any center
        farthest_point = np.argmax(min_distances)
        max_distance = min_distances[farthest_point]
        centers.append(farthest_point)
        # Update min_distances efficiently
        min_distances = np.minimum(min_distances, distance_matrix[farthest_point, :])
        k += 1
    # Assign each point to its closest center
    centers_arr = np.array(centers)
    closest_centers = np.argmin(distance_matrix[:, centers_arr], axis=1)
    assigned_center = {i: centers_arr[closest_centers[i]] for i in range(n)}
    return assigned_center


def weighted_cluster_sample(cluster, weights):
    cluster_dict = defaultdict(list)
    for idx, center in cluster.items():
        cluster_dict[center].append(idx)

    selected = []
    selected_weights = []
    n_samples = len(cluster_dict)

    for center, members in cluster_dict.items():
        member_weights = weights[members]
        total_weight = member_weights.sum()
        probabilities = member_weights / total_weight
        chosen = np.random.choice(
            members, size=int(np.ceil(n_samples * total_weight)), p=probabilities
        )
        selected.append(chosen)
        selected_weights.append([total_weight / len(chosen)] * len(chosen))

    return np.concatenate(selected), np.concatenate(selected_weights)


def margin_loss(score, y_true, margin):
    y_true = y_true.apply_(lambda x: 1 if x == 1 else -1)
    margins = y_true * score
    losses = margins <= margin
    return losses.float().mean()


class ResNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        output = self.final_layer(self.linear_stack(x))
        return output + x

    def initialize(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(layer.bias, 0.0)
        for m in [self.final_layer]:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)


class StackedResNN(nn.Module):
    def __init__(self, n_nets, input_dim, hidden_dim):
        super().__init__()
        self.n_nets = n_nets
        self.nets = nn.ModuleList([ResNN(input_dim, hidden_dim) for _ in range(n_nets)])
        for net in self.nets:
            net.initialize()

    def forward(self, x):
        for net in self.nets:
            x = net(x)
        return x


class StackedResNNLogReg(nn.Module):
    def __init__(self, n_nets, input_dim, hidden_dim, target_dim):
        super().__init__()
        self.n_nets = n_nets
        self.nets = nn.ModuleList([ResNN(input_dim, hidden_dim) for _ in range(n_nets)])
        for net in self.nets:
            net.initialize()
        self.classifier = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        for net in self.nets:
            x = net(x)
        return self.classifier(x)


class NormalizedStackedResNN(nn.Module):
    def __init__(self, n_nets, input_dim, hidden_dim):
        super().__init__()
        self.n_nets = n_nets
        self.nets = nn.ModuleList([ResNN(input_dim, hidden_dim) for _ in range(n_nets)])
        for net in self.nets:
            net.initialize()

    def forward(self, x):
        for net in self.nets:
            x = net(x)
        return F.normalize(x, p=2, dim=1)
