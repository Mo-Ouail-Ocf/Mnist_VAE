import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pathlib import Path
from ignite.handlers import Checkpoint
from lib.model import VAE

def load_checkpoint(checkpoint_dir, device='cuda'):
    # Load the model checkpoint
    checkpoint_path = Path(checkpoint_dir)
    
    gen = VAE(28*28).to(device)
    
    checkpoint = torch.load(checkpoint_path / 'model_checkpoint_200.pt', map_location=device)
    Checkpoint.load_objects(to_load={'model': gen}, checkpoint=checkpoint)
    
    return gen

def get_latent_vectors(model, dataloader, device='cuda'):
    latent_vectors = []
    labels = []
    
    model.eval()
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.view(-1, 28*28).to(device)
            
            # Get latent vectors (z) from the encoder
            mu, log_var = model.encoder(images)
            z = model.reparameterize(mu, log_var)
            
            latent_vectors.append(z.cpu().numpy())
            labels.append(targets.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return latent_vectors, labels

def visualize_latent_space(latent_vectors, labels):
    # Apply PCA to reduce the latent space from 20D to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Plot the 2D latent space
    plt.figure(figsize=(8, 6))
    
    # Create a scatter plot where each point is a latent vector, colored by the corresponding digit
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    
    # Add a color bar to show which color corresponds to which digit
    plt.colorbar(scatter, label='Digit label')
    plt.title('Latent Space of MNIST (2D PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

def main(checkpoint_dir, device='cuda'):
    # Load the VAE model
    gen = load_checkpoint(checkpoint_dir, device=device)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
    
    # Get latent vectors and their corresponding labels
    latent_vectors, labels = get_latent_vectors(gen, test_loader, device=device)
    
    # Visualize the latent space using PCA
    visualize_latent_space(latent_vectors, labels)

if __name__ == "__main__":
    main(checkpoint_dir='./weights', device='cuda')
