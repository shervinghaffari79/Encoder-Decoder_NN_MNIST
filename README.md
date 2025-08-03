# MNIST Autoencoder Project 🖼️

## Overview 📖
This project implements a convolutional autoencoder using PyTorch to compress and reconstruct images from the MNIST dataset. The autoencoder learns to encode high-dimensional input images into a lower-dimensional latent space and then decode them back to their original form, achieving effective image reconstruction with minimal loss.

## Project Structure 🗂️
- <code>project_3_3.ipynb</code>: Jupyter notebook containing the complete implementation, including data preprocessing, model architecture, training, and visualization. 📓
- <code>/data</code>: Directory where the MNIST dataset is automatically downloaded and stored. 📁
- <code>README.md</code>: This file, providing an overview and documentation of the project. 📄

## Table of Contents 📑
- [Methodology](#methodology) 🛠️
- [Achievements](#achievements) 🏆
- [Installation and Setup](#installation-and-setup) 🔧
- [Steps to Run](#steps-to-run) 🚀
- [Usage](#usage) ▶️
- [Future Improvements](#future-improvements) 🔮
- [License](#license) 📄
- [Acknowledgments](#acknowledgments) 🙏

## Methodology 🛠️
The project follows a structured approach to build, train, and evaluate a convolutional autoencoder for the MNIST dataset. Below is a detailed breakdown of the methodology:

### 1. Data Preparation 📊
- **Dataset**: The MNIST dataset, consisting of 60,000 grayscale handwritten digit images (28x28 pixels) for training, is used. ✏️
- **Preprocessing**: Images are transformed into PyTorch tensors using <code>transforms.ToTensor()</code> and normalized to the range [0, 1]. 🔄
- **Data Loading**: The dataset is loaded using <code>torchvision.datasets.MNIST</code> and batched with a <code>DataLoader</code> (batch size = 32, shuffled) for efficient training. 📥

### 2. Model Architecture 🏗️
The autoencoder consists of two main components: an Encoder and a Decoder, combined into an <code>EncoderDecoder</code> class.

#### Encoder 🔍
- **Input**: Grayscale images (1x28x28). 🖼️
- **Architecture**:
  - Three convolutional layers (<code>nn.Conv2d</code>) with ReLU activation:
    - Conv1: 1 input channel → 16 output channels, kernel size 3, stride 2, padding 1.
    - Conv2: 16 input channels → 32 output channels, kernel size 3, stride 2, padding 1.
    - Conv3: 32 input channels → 64 output channels, kernel size 3, stride 2, padding 1.
  - Adaptive average pooling (<code>nn.AdaptiveAvgPool2d</code>) to reduce the feature map to a 1x1 spatial dimension, producing a 64-dimensional latent representation. 📉
- **Purpose**: Compresses the input image into a compact latent space representation. 🤏

#### Decoder 🔄
- **Input**: 64-dimensional latent vector (reshaped to 64x1x1). 📐
- **Architecture**:
  - Four transposed convolutional layers (<code>nn.ConvTranspose2d</code>) with ReLU activation (except the final layer):
    - ConvTranspose1: 64 input channels → 32 output channels, kernel size 4, stride 1, padding 0 (outputs 4x4).
    - ConvTranspose2: 32 input channels → 16 output channels, kernel size 3, stride 2, padding 1 (outputs 7x7).
    - ConvTranspose3: 16 input channels → 8 output channels, kernel size 4, stride 2, padding 1 (outputs 14x14).
    - ConvTranspose4: 8 input channels → 1 output channel, kernel size 4, stride 2, padding 1 (outputs 28x28).
- **Purpose**: Reconstructs the original 28x28 image from the latent representation. 🖼️

#### EncoderDecoder 🔗
- Combines the Encoder and Decoder into a single model, passing the encoder's output directly to the decoder. ⚙️

### 3. Training 🏋️‍♂️
- **Loss Function**: Mean Squared Error (MSE) loss to measure the difference between the input and reconstructed images. 📏
- **Optimizer**: Adam optimizer with a learning rate of 0.001. ⚡
- **Training Loop**:
  - 10 epochs, processing batches of 32 images.
  - For each batch:
    - Zero the gradients.
    - Forward pass through the autoencoder.
    - Compute MSE loss.
    - Backpropagate and update model parameters.
  - Loss is printed for each epoch to monitor training progress. 📈

### 4. Evaluation and Visualization 📊
- **Visualization**: After training, the notebook visualizes 10 pairs of images (original vs. reconstructed) from the training set using Matplotlib. 👀
- **Metrics**: The final training loss after 10 epochs is approximately 0.0079, indicating good reconstruction quality. ✅

## Achievements 🏆
- **Effective Compression**: The autoencoder successfully compresses 28x28 MNIST images (784 pixels) into a 64-dimensional latent space, achieving a compression ratio of ~12:1. 📉
- **High Reconstruction Quality**: The model reconstructs images with minimal loss (final MSE ~0.0079), preserving key visual features of the handwritten digits. ✨
- **Robust Implementation**: The use of convolutional layers and adaptive pooling ensures efficient feature extraction and reconstruction, suitable for image data. 💪
- **Clear Visualization**: The project includes clear visualizations comparing original and reconstructed images, demonstrating the model's ability to capture essential digit characteristics. 🖼️

## Installation and Setup 🔧
To run this project, ensure you have the following dependencies installed:
```bash
pip install torch torchvision matplotlib
```

## Steps to Run 🚀
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```
2. **Navigate to the project directory**:
   ```bash
   cd <project-directory>
   ```
3. **Open and run the Jupyter notebook**:
   ```bash
   jupyter notebook project_3_3.ipynb
   ```
Ensure the <code>/data</code> directory is writable, as the MNIST dataset will be downloaded there automatically. 📂

## Usage ▶️
- **Training**: Execute the training cells in the notebook to train the autoencoder for 10 epochs. Modify <code>batch_size</code>, <code>lr</code>, or the number of epochs as needed. 🏋️‍♂️
- **Visualization**: Run the visualization cell to compare original and reconstructed images. 👁️
- **Customization**: Adjust the encoder/decoder architecture (e.g., number of layers, channels) or hyperparameters to experiment with different configurations. 🛠️

## Future Improvements 🔮
- **Latent Space Analysis**: Explore the latent space to understand learned representations or use them for tasks like clustering or generation. 🔍
- **Advanced Architectures**: Incorporate techniques like batch normalization, dropout, or variational autoencoders for improved performance. ⚙️
- **Extended Evaluation**: Add quantitative metrics (e.g., SSIM, PSNR) to evaluate reconstruction quality beyond MSE. 📊
- **Testing Set**: Include a test dataset to evaluate generalization performance. ✅

## License 📄
This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details. ⚖️

## Acknowledgments 🙏
- The MNIST dataset is provided by Yann LeCun and team. 📚
- Built using PyTorch and torchvision. 🔥

Happy coding and experimenting with autoencoders! 🎉
