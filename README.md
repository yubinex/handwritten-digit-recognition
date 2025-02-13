# MNIST CNN Classifier

This repository contains a simple Convolutional Neural Network (CNN) implemented in PyTorch to classify handwritten digits from the MNIST dataset.

## Features
- Loads and preprocesses the MNIST dataset using `torchvision`.
- Implements a CNN with two convolutional layers and dropout.
- Trains the model using the Adam optimizer and cross-entropy loss.
- Evaluates the model on the test dataset.
- Visualizes predictions on sample images.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install torch torchvision matplotlib
```

## Usage

### Train the Model
Run the following command in a Jupyter Notebook or Python script to train the model:

```python
for epoch in range(1, 11):
    train(epoch)
    test()
```

### Test the Model
After training, you can evaluate the model using:

```python
test()
```

### Make Predictions
To visualize a sample prediction:

```python
import matplotlib.pyplot as plt

data, target = test_data[0]
data = data.unsqueeze(0).to(device)
output = model(data)
prediction = output.argmax(dim=1, keepdim=True).item()
print(f"Prediction: {prediction}")
plt.imshow(data.squeeze(0).squeeze(0).cpu().numpy(), cmap="gray")
plt.show()
```

## Model Architecture
- **Conv1**: 1 input channel → 10 filters (5x5 kernel)
- **Conv2**: 10 input channels → 20 filters (5x5 kernel) + Dropout
- **FC1**: 320 → 50 neurons
- **FC2**: 50 → 10 output classes
- **Activation**: ReLU and Softmax

## Device Compatibility
This script runs on both CPU and GPU. It automatically detects available hardware:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## License
This project is open-source and available under the MIT License.
