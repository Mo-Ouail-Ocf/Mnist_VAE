import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,z_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, z_dim)
        self.sigma = nn.Linear(hidden_size, z_dim)

    def forward(self, x):
        x = self.hidden_layer(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size=200, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, z_dim)
        self.decoder = Decoder(z_dim, hidden_size, input_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
if __name__ == '__main__':
    model = VAE(784, 200, 20)
    x = torch.randn(32, 784)
    x_reconstructed, mu, sigma = model(x)
    print(x_reconstructed.shape, mu.shape, sigma.shape)
