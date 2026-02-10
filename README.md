# Neural Network from Scratch

A complete implementation of neural networks from pure NumPy, progressing from simple perceptrons to advanced multi-layer networks with multiple optimizers.

## Features

### Core Implementations

**SimplePerceptron (perceptron.py)**
- Single-layer binary classifier
- Sigmoid activation function
- Gradient descent optimization
- Logic gate demonstrations (AND, OR, NAND)

**Multi-Layer Network (neural_network.py)**
- Configurable architecture with multiple layers
- Multiple activation functions: ReLU, Sigmoid, Tanh, Softmax
- Multiple optimizers: SGD, Momentum, Adam
- Cross-entropy loss for multi-class classification
- Early stopping with validation monitoring
- He weight initialization

### Training Features
- Mini-batch gradient descent
- Backpropagation from scratch with chain rule
- Early stopping to prevent overfitting
- Train/validation/test splits
- Comprehensive metrics (accuracy, precision, recall, F1)

### Visualization & Evaluation
- Training curves (loss and accuracy)
- Confusion matrices
- Sample prediction visualization
- sklearn baseline comparison

## Results

### Full MNIST Dataset (70,000 samples)

| Implementation | Accuracy | Training Time |
|---|---|---|
| **Custom NumPy** | **97.28%** | **44.7s** |
| sklearn MLP | 96.96% | 96.6s |

**Key Achievement**: Custom implementation outperforms sklearn in both accuracy and speed! The efficiency comes from optimized NumPy vectorization.

## File Structure

```
neural-net/
├── perceptron.py              # Single-layer perceptron
├── neural_network.py          # Multi-layer network from scratch
├── app.py                     # Logic gates demonstration
├── test_mnist.py              # Full MNIST training & evaluation
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
└── venv/                      # Virtual environment
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JemHRice/neural-net.git
cd neural-net
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Test Simple Perceptron (Logic Gates)
```bash
python perceptron.py
```
Generates: `and_loss.png`, `or_loss.png`, `nand_loss.png`

### Train on Full MNIST
```bash
python test_mnist.py
```
Generates:
- `training_curves.png` - Loss and accuracy trends
- `confusion_matrix.png` - Misclassification patterns
- `predictions.png` - Sample predictions with correctness

## How It Works

### Forward Pass
```
Input → Linear(Wx+b) → Activation → ... → Output
```
- Each layer applies `z = Wx + b` followed by activation
- Hidden layers use ReLU, Sigmoid, or Tanh
- Output layer uses Softmax for probability distribution

### Backward Pass (Backpropagation)
```
Output Error → Propagate Backwards → Compute Gradients → Update Weights
```
- Output layer: `δ = prediction - target`
- Hidden layers: `δ = (W^T δ_next) ⊙ activation'(z)`
- Gradient: `dW = (a^T δ) / batch_size`

### Optimizers

**SGD (Stochastic Gradient Descent)**
```
w = w - lr * dw
```
Simple but can be slow and unstable

**Momentum**
```
v = β*v + (1-β)*dw
w = w - lr*v
```
Accumulates gradient history, smoother convergence

**Adam (Adaptive Moment Estimation)**
```
m = β1*m + (1-β1)*dw  # First moment
v = β2*v + (1-β2)*dw²  # Second moment
w = w - lr * m / (sqrt(v) + eps)  # Biased update
```
Combines momentum with adaptive per-parameter learning rates

## Architecture Example

```
Input (784 features)
    ↓
Dense(256, ReLU)
    ↓
Dense(128, ReLU)
    ↓
Dense(10, Softmax)
    ↓
Output (probabilities for 0-9)
```

**Why this architecture?**
- 784 input features: Flattened 28×28 MNIST images
- 256 & 128 hidden units: Sufficient capacity without overfitting
- ReLU activation: Prevents vanishing gradients
- Softmax output: Ensures valid probability distribution

## Key Concepts Explained

### Gradient Descent
Iteratively moving weights in the direction that reduces loss by following the negative gradient.

### Backpropagation
Computing gradients efficiently using the chain rule, propagating errors backwards through the network.

### Activation Functions
Non-linear transformations that allow networks to learn complex patterns:
- **ReLU**: `max(0, x)` - Fast, avoids vanishing gradients
- **Sigmoid**: `1/(1+e^-x)` - Smooth, but prone to vanishing gradients
- **Tanh**: `(e^x - e^-x)/(e^x + e^-x)` - Similar to sigmoid but centered at 0

### Early Stopping
Monitor validation loss and stop training when it stops improving to prevent overfitting.

## Comparison with Libraries

**Advantages of Custom Implementation:**
- Deep understanding of fundamentals
- Educational value for ML interviews
- Full control over every parameter
- Great for teaching others

**Advantages of TensorFlow/PyTorch:**
- Optimized C/GPU backends (~100x faster)
- Extensive features (layers, regularization, etc.)
- Production-ready and battle-tested
- Large community support

## Learning Path

1. Start with `app.py` - understand perceptrons on simple logic gates
2. Read `perceptron.py` - learn the gradient descent algorithm
3. Study `neural_network.py` - understand multi-layer networks
4. Run `test_mnist.py` - see it all working on real data

## Future Enhancements

- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] L2/L1 weight decay
- [ ] Learning rate scheduling
- [ ] Convolutional layers
- [ ] GPU support with CuPy
- [ ] Regression tasks
- [ ] Attention mechanisms

## References

- [Deep Learning Book](http://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [3Blue1Brown: Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Backpropagation Explained](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

## License

Open source - free to use for learning and development.

---

**Author**: Jem Rice  
**Created**: February 2026  
**Purpose**: Demonstrate deep understanding of neural network fundamentals
