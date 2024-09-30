# Variational Autoencoder (VAE) from Scratch in PyTorch

## Project Overview
This project implements a **Variational Autoencoder (VAE)** using PyTorch, applied to the **MNIST** dataset. A VAE is a generative model that encodes input data into a probabilistic latent space and then reconstructs it from this latent space by learning meaningful latent representations.

The model is trained on MNIST, a dataset of 28x28 grayscale images of handwritten digits, to learn compressed representations and generate new images by sampling from the latent space.

---

## VAE Architecture
### Encoder
- The encoder network takes the input image and encodes it into a latent space characterized by:
  - **Mean vector** (`mu`)
  - **Log-variance vector** (`logvar`)

These parameters define the distribution from which the latent code is sampled, helping to enforce the regularization necessary for a smooth latent space.

### Latent Sampling
- A latent vector `z` is sampled from a Gaussian distribution defined by the mean (`mu`) and log-variance (`logvar`) output by the encoder, using the **reparameterization trick** to ensure backpropagation through the stochastic process.

### Decoder
- The decoder network takes the sampled latent vector `z` and reconstructs the input image. The decoder learns to map the latent distribution back to the input space.

---

## Loss Function
The VAE is trained using a loss function that combines two terms:

1. **Reconstruction Loss**: Measures the difference between the original image and the reconstructed image (e.g., Binary Cross-Entropy loss for pixel-wise comparison).
2. **KL Divergence**: Regularizes the latent space by minimizing the difference between the learned latent distribution and the standard normal distribution.

The total loss is:
\[ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}} \]

where:
- \( \mathcal{L}_{\text{recon}} \) is the reconstruction loss.
- \( \mathcal{L}_{\text{KL}} \) is the KL divergence.
- \( \beta \) controls the weight of the KL divergence term.

---

## Dataset: MNIST
The MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
Each image is 28x28 pixels, with grayscale values between 0 and 1, representing handwritten digits from 0 to 9.

---

## Results
Once trained, the VAE is able to:
1. **Generate new handwritten digits** by sampling from the learned latent space.
2. **Reconstruct input images** with reasonable accuracy.

Here are sample results:
- **Generated Images**: The model can generate new images of digits that resemble the training data.
- **Reconstructed Images**: The input digits are reconstructed through the encoder-decoder pipeline, showing the model's ability to learn compressed representations.

---

## References
The implementation is inspired by the original Variational Autoencoder paper:

**Kingma, D. P., & Welling, M. (2013).** Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114. [Link to paper](https://arxiv.org/abs/1312.6114).

