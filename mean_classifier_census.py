# %%
import torch
import pytorch_optimizer as optimizers
from river import linear_model, optim, stream
from adult import Adult
from sklearn.preprocessing import RobustScaler
import mc

# %%
# Cost parameter
c = 1.0  # Cost parameter for false positives


# %%
# Define loss function
def loss(y_true, y_pred, c=1):
    # y_true, y_pred: 1D tensors with values in {-1, 1}
    yt = y_true.view(-1)
    yp = y_pred.view(-1)
    false_neg = ((yt == 1) & (yp == -1)).sum().item() * 1
    false_pos = ((yt == -1) & (yp == 1)).sum().item() * c
    total = yt.numel()
    return (false_neg + false_pos) / total


# %%
# Load in Data
train_set = Adult(root=".\\data", download=True)
test_set = Adult(root=".\\data", download=True, train=False)

fm = RobustScaler()
train_set.data = torch.tensor(fm.fit_transform(train_set.data), dtype=torch.float32)
test_set.data = torch.tensor(fm.transform(test_set.data), dtype=torch.float32)

train_set.targets = torch.where(train_set.targets == 1, -1, 1)
test_set.targets = torch.where(test_set.targets == 1, -1, 1)

# %%
# Baseline Constant Classifier
constant_predictions_neg = torch.full_like(test_set.targets, -1)
constant_predictions_pos = torch.full_like(test_set.targets, 1)
print(
    "Constant Classifier Loss:",
    min(
        loss(test_set.targets, constant_predictions_neg, c=c),
        loss(test_set.targets, constant_predictions_pos, c=c),
    ),
)

# %%
## Mean Classifier
mean_weights = mc.learn_mean_weights(train_set.data, train_set.targets, c=c)
predictions = mc.mean_classifier(test_set.data, mean_weights)

print(
    "Mean Loss:",
    loss(test_set.targets, predictions, c=c),
)

# %%
# Logistic Regression Baseline
opt = optim.Averager(optim.AdaGrad(1.0), 0)
log_reg = linear_model.LogisticRegression(optimizer=opt, loss=optim.losses.Log())
for x, y in stream.iter_array(train_set.data.numpy(), train_set.targets.numpy()):
    log_reg.learn_one(x, y)

log_reg_avg = linear_model.LogisticRegression(optimizer=optim.SGD(2.0))
log_reg_avg.learn_one(dict(opt.avg_w), 1)

predictions = []
for x, _ in stream.iter_array(test_set.data.numpy(), test_set.targets.numpy()):
    predictions.append(log_reg_avg.predict_proba_one(x))
predictions = torch.tensor([p[True] for p in predictions])
predictions = torch.where(predictions >= c / (1 + c), 1, -1)

print(
    "Logistic Regression Loss:",
    loss(test_set.targets, predictions, c=c),
)

# %%
my_nn = mc.NormalizedStackedResNN(2, 104, 50).cuda()

base_opt = optimizers.Adam
opt = optimizers.SAM(my_nn.parameters(), base_opt, lr=0.001, maximize=True, rho=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, T_max=50)

X_train_gpu = train_set.data.cuda()
y_train_gpu = train_set.targets.cuda()
n_train = X_train_gpu.shape[0]
batch_size = 5000


# %%
epochs = 1000
average_loss = 0.0

for j in range(epochs):
    idx = torch.randperm(n_train, device=X_train_gpu.device)[:batch_size]
    inputs, labels = X_train_gpu[idx], y_train_gpu[idx]

    features = my_nn(inputs)
    objective = mc.self_similarity_loss_linear(features, labels, c=c) / (batch_size**2)
    objective.backward()
    opt.first_step(zero_grad=True)

    features = my_nn(inputs)
    objective = mc.self_similarity_loss_linear(features, labels, c=c) / (batch_size**2)
    objective.backward()
    opt.second_step(zero_grad=True)

    average_loss = 0.9 * average_loss + 0.1 * objective.detach()

    scheduler.step()

    if (j + 1) % 50 == 0:
        print(f"Epoch {j+1}: Loss: {average_loss:.6f}")

# %%
with torch.no_grad():
    X_train_nn = my_nn(train_set.data.cuda())
    X_test_nn = my_nn(test_set.data.cuda())

    mean_weights = mc.learn_mean_weights(X_train_nn, train_set.targets.cuda(), c=c)
    predictions = mc.mean_classifier(X_test_nn, mean_weights).cpu()

    print("NN + Mean Loss:", loss(test_set.targets, predictions, c=c))

# %%
