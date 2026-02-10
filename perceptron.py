import numpy as np
import matplotlib.pyplot as plt


class SimplePerceptron:
    """Single-layer perceptron for binary classification"""

    def __init__(self, input_size, learning_rate=0.1):
        """
        Initialize perceptron

        Args:
            input_size: Number of input features
            learning_rate: Learning rate for weight updates
        """
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            Predictions between 0 and 1
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def train(self, X, y, epochs=1000):
        """
        Train the perceptron

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            epochs: Number of training iterations

        Returns:
            List of losses for each epoch
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            predictions = self.predict(X)

            # Calculate loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)

            # Backward pass (gradient descent)
            error = predictions - y

            # Update weights and bias
            self.weights -= self.lr * np.dot(X.T, error) / len(X)
            self.bias -= self.lr * np.mean(error)

            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses


# Test on logic gates
def test_logic_gate(gate_name, X, y):
    """Test perceptron on a logic gate"""
    print(f"\n{'='*50}")
    print(f"Testing {gate_name} Gate")
    print(f"{'='*50}")

    # Create and train perceptron
    model = SimplePerceptron(input_size=2, learning_rate=0.1)
    losses = model.train(X, y, epochs=1000)

    # Make predictions
    print("\nFinal Predictions:")
    predictions = model.predict(X)
    for i, (x, pred, actual) in enumerate(zip(X, predictions, y)):
        print(
            f"Input: {x}, Predicted: {pred:.3f}, Actual: {actual}, Correct: {abs(pred - actual) < 0.5}"
        )

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title(f"{gate_name} Gate - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(f"{gate_name.lower()}_loss.png")
    print(f"\nLoss curve saved as {gate_name.lower()}_loss.png")


if __name__ == "__main__":
    # AND Gate
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    test_logic_gate("AND", X_and, y_and)

    # OR Gate
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    test_logic_gate("OR", X_or, y_or)

    # NAND Gate
    X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_nand = np.array([1, 1, 1, 0])
    test_logic_gate("NAND", X_nand, y_nand)

    print("\n" + "=" * 50)
    print("All tests complete!")
    print("=" * 50)
