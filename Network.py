import numpy as np

# Initialise Network
def init_network(input_size, hidden_size, output_size):
    np.random.seed(42)  # reproducible

    # Xavier/Glorot for sigmoid: N(0, sqrt(1/fan_in))
    w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
    b1 = np.random.randn(1, hidden_size) * 0.01  # small random bias

    w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
    b2 = np.random.randn(1, output_size) * 0.01  # small random bias

    return w1, b1, w2, b2

# Activiation (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation
def forward(X, w1, b1, w2, b2):

    # Feed input data through the network:
    # X  -> input layer
    # w1,b1 -> hidden layer parameters
    # w2,b2 -> output layer parameters
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return a2

# Derivative of the sigmoid activation.
# Note: expects 'a' = sigmoid(z), not 'z or X' directly.
def sigmoid_derivative(a):
    return a * (1 - a)

# Mean Squared Error Loss
# Usefull for didactic purposes, for binary tasks also: BCE
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def bce_loss(y_true, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)  # numerical stability
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Steps per epoch:
# 1) Forward pass
# 2) Compute loss
# 3) Backward pass (gradients w.r.t. weights and biases)
# 4) Parameter update (gradient descent)
def train(X, y, w1, b1, w2, b2, lr=0.5, epochs=8000, verbose_every=500):
    n = X.shape[0]

    for epoch in range(1, epochs + 1):
        # ----- Forward -----
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)

        # ----- Loss (BCE) -----
        loss = bce_loss(y, a2)

        # ----- Backprop -----
        # Output layer: with BCE + sigmoid, gradient simplifies nicely:
        dLoss_dz2 = (a2 - y) / n                  # shape (n, 1)
        dLoss_dw2 = np.dot(a1.T, dLoss_dz2)       # (hidden, n)·(n,1) => (hidden,1)
        dLoss_db2 = np.sum(dLoss_dz2, axis=0, keepdims=True)

        # Hidden layer:
        dLoss_da1 = np.dot(dLoss_dz2, w2.T)       # (n,1)·(1,hidden) => (n,hidden)
        da1_dz1 = sigmoid_derivative(a1)          # expects a1 = sigmoid(z1)
        dLoss_dz1 = dLoss_da1 * da1_dz1

        dLoss_dw1 = np.dot(X.T, dLoss_dz1) / n    # (inputs,n)·(n,hidden) => (inputs,hidden)
        dLoss_db1 = np.sum(dLoss_dz1, axis=0, keepdims=True) / n

        w1 -= lr * dLoss_dw1
        b1 -= lr * dLoss_db1
        w2 -= lr * dLoss_dw2
        b2 -= lr * dLoss_db2

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
            print(f"Epoch {epoch:>5} | loss: {loss:.6f}")

    return w1, b1, w2, b2


def predict(X, w1, b1, w2, b2, threshold=0.5):
    probs = forward(X, w1, b1, w2, b2)
    labels = (probs >= threshold).astype(int)
    return probs, labels

# Testing
if __name__ == "__main__":
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Try 2-4-1 for XOR
    w1, b1, w2, b2 = init_network(2, 4, 1)

    w1, b1, w2, b2 = train(
        X, y, w1, b1, w2, b2,
        lr=0.5,
        epochs=8000,
        verbose_every=500
    )

    probs, labels = predict(X, w1, b1, w2, b2, threshold=0.5)
    print("\nPredicted probabilities:\n", probs.round(4))
    print("Predicted labels:\n", labels.reshape(-1))
    print("True labels:\n", y.reshape(-1))
    acc = np.mean(labels.reshape(-1) == y.reshape(-1))
    print(f"\nTraining accuracy: {acc * 100:.2f}%")


