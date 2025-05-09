# Fine-Tuning Demo: Non-linear Function Adaptation

This repository is an attempt to show the concept of fine-tuning through a simple yet illustrative example. It shows how a neural network can adapt to local changes in a non-linear function while preserving its original learning.

## Overview

The demo consists of three main phases:
1. **Initial Training**: Train a neural network to approximate a non-linear function across the full domain [0,4]
2. **Data Modification**: Introduce changes to a specific section of the input domain [2,3]
3. **Fine-Tuning**: Adapt the model to the modified section while maintaining performance on the original function

## Features

- **Interactive Visualization**:
  - Real-time plotting of original and modified functions
  - Color-coded data points (blue for base, red for fine-tuning)
  - Training and validation points with different markers
  - Loss curves for both training phases
  - Neural network architecture visualization

- **Training Controls**:
  - Adjustable learning rate (default: 0.001)
  - Multiple layer freezing options (6, 8, or 10 layers)
  - Configurable batch size (default: 512)
  - Training step control (default: 50 steps)
  - Model reset functionality

- **Data Generation**:
  - Base dataset: 10,000 samples
  - Fine-tuning dataset: 1,000 samples in [2,3] range
  - Automatic train/validation split
  - Two fine-tuning modes: combined or new data only

## Installation

1. Clone this repository:
```bash
git clone git@github.com:olmozavala/fine_tuning_demo.git
```

2. Install the required dependencies:
```bash
pip install torch numpy matplotlib plotly dash
```

## Usage

1. Run the demonstration:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8050
```

3. The interface will guide you through:
   - Initial model training on the original function
   - Visualization of the modified data section
   - Fine-tuning process with performance monitoring

## Implementation Details

### Model Architecture
- Feed-forward neural network with configurable layers (default: 11 layers)
- Adjustable hidden size (default: 20 units)
- ReLU activation functions
- Batch normalization between layers
- Parameter counting display for transparency

### Fine-Tuning Methods
1. **None**: All parameters frozen (baseline)
2. **Full**: All parameters trainable
3. **Freeze6**: First 6 layers frozen
4. **Freeze8**: First 8 layers frozen
5. **Freeze10**: First 10 layers frozen

### Data Structure
- **Base Function**: Continuous non-linear function over [0,4]
- **Modified Section**: Complex function in [2,3] combining:
  - Sinusoidal oscillations: sin(2x) * cos(3x)
  - High-frequency components: 0.2 * sin(16x)
  - Non-linear transformations: 0.2 * tanh(x²)
  - Random noise (std: 0.1)

### Training Process
1. **Initial Training**:
   - Learn the complete non-linear function
   - Automatic validation split
   - Loss monitoring for both training and validation
   - Model state saving after convergence

2. **Fine-Tuning Options**:
   - Combined Mode: Train on both original and new data
   - New Data Only Mode: Focus exclusively on modified section
   - Automatic learning rate adjustment
   - Selective layer freezing based on chosen method

## Best Practices Demonstrated

- Proper learning rate selection for fine-tuning
- Strategic layer freezing with multiple options
- Balanced dataset creation with validation splits
- Real-time loss monitoring
- Preventing catastrophic forgetting through:
  - Selective layer freezing
  - Combined dataset training option
  - Gradual adaptation strategies

## File Structure

```
├── app.py            # Dash application and main interface
├── model.py          # Neural network architecture
├── data.py           # Data generation utilities
├── training.py       # Training and fine-tuning logic
├── assets/           # Network architecture images
└── README.md
```

## Contributing

Contributions to improve the demonstration are welcome:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License

Copyright (c) 2024 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.