import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class NeuralNetwork:
    """Multi-layer perceptron implemented from scratch in NumPy"""

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        optimizer: str = "sgd",
        activation: str = "relu",
    ):
        """
        Initialize neural network

        Args:
            layer_sizes: List of layer sizes, e.g., [784, 128, 64, 10]
            learning_rate: Learning rate for optimization
            optimizer: 'sgd', 'momentum', or 'adam'
            activation: 'relu', 'sigmoid', or 'tanh'
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.optimizer = optimizer
        self.activation_name = activation

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
                2.0 / layer_sizes[i]
            )
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

        # For momentum/adam
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # Adam timestep

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def activation(self, x):
        if self.activation_name == "relu":
            return self.relu(x)
        elif self.activation_name == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_name == "tanh":
            return self.tanh(x)

    def activation_derivative(self, x):
        if self.activation_name == "relu":
            return self.relu_derivative(x)
        elif self.activation_name == "sigmoid":
            return self.sigmoid_derivative(x)
        elif self.activation_name == "tanh":
            return self.tanh_derivative(x)

    def forward(self, X):
        """Forward pass through network"""
        self.z_values = []  # Pre-activation values
        self.a_values = [X]  # Post-activation values

        for i in range(len(self.weights)):
            z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            # Output layer uses softmax, hidden layers use chosen activation
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                a = self.activation(z)

            self.a_values.append(a)

        return self.a_values[-1]

    def backward(self, X, y):
        """Backpropagation to compute gradients"""
        m = X.shape[0]

        # Convert y to one-hot if needed
        if len(y.shape) == 1:
            y_onehot = np.zeros((m, self.layer_sizes[-1]))
            y_onehot[np.arange(m), y] = 1
            y = y_onehot

        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer gradient (softmax + cross-entropy)
        delta = self.a_values[-1] - y

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            dW[i] = np.dot(self.a_values[i].T, delta) / m
            db[i] = np.sum(delta, axis=0) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(
                    self.z_values[i - 1]
                )

        return dW, db

    def update_weights(self, dW, db):
        """Update weights using chosen optimizer"""
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * db[i]

        elif self.optimizer == "momentum":
            beta = 0.9
            for i in range(len(self.weights)):
                self.velocity_w[i] = beta * self.velocity_w[i] + (1 - beta) * dW[i]
                self.velocity_b[i] = beta * self.velocity_b[i] + (1 - beta) * db[i]
                self.weights[i] -= self.lr * self.velocity_w[i]
                self.biases[i] -= self.lr * self.velocity_b[i]

        elif self.optimizer == "adam":
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            self.t += 1

            for i in range(len(self.weights)):
                # Update biased first/second moment estimates
                self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dW[i]
                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db[i]
                self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dW[i] ** 2)
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db[i] ** 2)

                # Bias correction
                m_w_hat = self.m_w[i] / (1 - beta1**self.t)
                m_b_hat = self.m_b[i] / (1 - beta1**self.t)
                v_w_hat = self.v_w[i] / (1 - beta2**self.t)
                v_b_hat = self.v_b[i] / (1 - beta2**self.t)

                # Update weights
                self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
                self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def compute_loss(self, y_pred, y_true):
        """Cross-entropy loss"""
        m = y_true.shape[0]

        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_onehot = np.zeros((m, self.layer_sizes[-1]))
            y_onehot[np.arange(m), y_true] = 1
            y_true = y_onehot

        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def compute_accuracy(self, y_pred, y_true):
        """Compute classification accuracy"""
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        early_stopping_patience=10,
        verbose=True,
    ):
        """
        Train the neural network

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            early_stopping_patience: Epochs to wait before early stopping
            verbose: Print progress
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i : i + batch_size]
                y_batch = y_train_shuffled[i : i + batch_size]

                # Forward and backward pass
                y_pred = self.forward(X_batch)
                dW, db = self.backward(X_batch, y_batch)
                self.update_weights(dW, db)

            # Compute metrics on full training set
            y_pred_train = self.forward(X_train)
            train_loss = self.compute_loss(y_pred_train, y_train)
            train_acc = self.compute_accuracy(y_pred_train, y_train)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation metrics
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = self.compute_loss(y_pred_val, y_val)
                val_acc = self.compute_accuracy(y_pred_val, y_val)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                    )

    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def evaluate(self, X, y) -> Dict:
        """Comprehensive evaluation metrics"""
        y_pred = self.predict(X)

        # Confusion matrix
        n_classes = self.layer_sizes[-1]
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y, y_pred):
            confusion_matrix[true, pred] += 1

        # Per-class metrics
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = (
                2 * precision[i] * recall[i] / (precision[i] + recall[i])
                if (precision[i] + recall[i]) > 0
                else 0
            )

        return {
            "accuracy": np.mean(y_pred == y),
            "precision_macro": np.mean(precision),
            "recall_macro": np.mean(recall),
            "f1_macro": np.mean(f1),
            "confusion_matrix": confusion_matrix,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1,
        }
