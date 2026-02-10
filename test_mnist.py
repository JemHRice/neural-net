import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from neural_network import NeuralNetwork
import time


def load_mnist():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    # For full MNIST (70,000 samples):
    print("Fetching full MNIST (this may take a minute on first run)...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X, y = mnist.data.values, mnist.target.values.astype(int)

    print(f"Full MNIST loaded: {X.shape} samples")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_custom_network(X_train, y_train, X_val, y_val, optimizer="adam"):
    """Train custom neural network"""
    print(f"\nTraining custom network with {optimizer} optimizer...")

    input_size = X_train.shape[1]
    hidden_sizes = [256, 128]  # Larger network for full MNIST
    output_size = len(np.unique(y_train))

    nn = NeuralNetwork(
        layer_sizes=[input_size] + hidden_sizes + [output_size],
        learning_rate=0.001,
        optimizer=optimizer,
        activation="relu",
    )

    start_time = time.time()
    nn.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,  # Fewer epochs for full dataset
        batch_size=128,  # Larger batch size for stability
        early_stopping_patience=5,
        verbose=True,
    )
    train_time = time.time() - start_time

    return nn, train_time


def train_sklearn_baseline(X_train, y_train, X_val, y_val):
    """Train sklearn MLP for comparison"""
    print("\nTraining sklearn baseline...")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),  # Larger network
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=50,  # Fewer iterations
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        verbose=False,
        batch_size=128,  # Larger batch size
    )

    start_time = time.time()
    mlp.fit(X_train, y_train)
    train_time = time.time() - start_time

    return mlp, train_time


def plot_training_curves(nn, save_path="training_curves.png"):
    """Plot loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(nn.history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, nn.history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, nn.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, nn.history["train_acc"], label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, nn.history["val_acc"], label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(confusion_matrix, save_path="confusion_matrix.png"):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")


def plot_sample_predictions(
    X_test, y_test, nn, n_samples=20, save_path="predictions.png"
):
    """Plot sample predictions"""
    predictions = nn.predict(X_test[:n_samples])

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for i, ax in enumerate(axes.flat):
        # Reshape to 8x8 for load_digits (or 28x28 for full MNIST)
        img_size = int(np.sqrt(X_test.shape[1]))
        ax.imshow(X_test[i].reshape(img_size, img_size), cmap="gray")

        pred = predictions[i]
        actual = y_test[i]
        color = "green" if pred == actual else "red"

        ax.set_title(f"Pred: {pred}, True: {actual}", color=color, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Sample Predictions (Green=Correct, Red=Wrong)", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Sample predictions saved to {save_path}")


def compare_optimizers(X_train, y_train, X_val, y_val, X_test, y_test):
    """Compare different optimizers"""
    optimizers = ["sgd", "momentum", "adam"]
    results = {}

    for opt in optimizers:
        print(f"\n{'='*60}")
        print(f"Testing {opt.upper()} optimizer")
        print("=" * 60)

        nn, train_time = train_custom_network(
            X_train, y_train, X_val, y_val, optimizer=opt
        )
        metrics = nn.evaluate(X_test, y_test)

        results[opt] = {
            "accuracy": metrics["accuracy"],
            "train_time": train_time,
            "final_loss": nn.history["val_loss"][-1],
        }

    # Print comparison table
    print(f"\n{'='*60}")
    print("OPTIMIZER COMPARISON")
    print("=" * 60)
    print(f"{'Optimizer':<12} {'Accuracy':<12} {'Train Time':<15} {'Final Loss':<12}")
    print("-" * 60)
    for opt, res in results.items():
        print(
            f"{opt.upper():<12} {res['accuracy']:<12.4f} {res['train_time']:<15.2f}s {res['final_loss']:<12.4f}"
        )

    return results


def main():
    """Main training and evaluation pipeline"""
    print("=" * 60)
    print("NEURAL NETWORK FROM SCRATCH - MNIST Classification")
    print("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist()

    # Train custom network
    nn, custom_train_time = train_custom_network(
        X_train, y_train, X_val, y_val, optimizer="adam"
    )

    # Train sklearn baseline
    sklearn_nn, sklearn_train_time = train_sklearn_baseline(
        X_train, y_train, X_val, y_val
    )

    # Evaluate custom network
    print("\n" + "=" * 60)
    print("CUSTOM NETWORK EVALUATION")
    print("=" * 60)
    custom_metrics = nn.evaluate(X_test, y_test)
    print(f"Accuracy: {custom_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {custom_metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {custom_metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro): {custom_metrics['f1_macro']:.4f}")
    print(f"Training Time: {custom_train_time:.2f}s")

    # Evaluate sklearn baseline
    print("\n" + "=" * 60)
    print("SKLEARN BASELINE EVALUATION")
    print("=" * 60)
    sklearn_acc = sklearn_nn.score(X_test, y_test)
    print(f"Accuracy: {sklearn_acc:.4f}")
    print(f"Training Time: {sklearn_train_time:.2f}s")

    # Comparison table
    print("\n" + "=" * 60)
    print("IMPLEMENTATION COMPARISON")
    print("=" * 60)
    print(f"{'Implementation':<20} {'Accuracy':<12} {'Train Time':<15}")
    print("-" * 60)
    print(
        f"{'Custom NumPy':<20} {custom_metrics['accuracy']:<12.4f} {custom_train_time:<15.2f}s"
    )
    print(f"{'sklearn MLP':<20} {sklearn_acc:<12.4f} {sklearn_train_time:<15.2f}s")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_training_curves(nn)
    plot_confusion_matrix(custom_metrics["confusion_matrix"])
    plot_sample_predictions(X_test, y_test, nn)

    # Compare optimizers (optional - takes longer)
    # compare_optimizers(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - predictions.png")


if __name__ == "__main__":
    main()
