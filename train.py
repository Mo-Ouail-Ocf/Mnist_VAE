from lib.model import VAE
from lib.utils import attach_checkpoint_handler,attach_ignite
from torchvision import datasets
from torchvision.transforms import transforms
import torch
from ignite.engine import Events,Engine
import torch.nn as nn
class Config:
    batch_size = 32
    epochs = 200
    lr = 1e-3
    device = 'cuda'
    z_dim = 20
    hidden_size = 200
    input_size = 784

# Load the data
train_ds = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)


model = VAE(Config.input_size, Config.hidden_size, Config.z_dim).to(Config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
reconstr_loss_fn = nn.BCELoss(reduction='sum')


def train_step(engine, batch):
    model.train()
    x, _ = batch
    x = x.view(-1, Config.input_size).to(Config.device)
    optimizer.zero_grad()
    x_reconstructed, mu, sigma = model(x)
    reconstr_loss = reconstr_loss_fn(x_reconstructed, x)
    kl_loss = -0.5 * torch.sum(1 +torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    loss = reconstr_loss + kl_loss
    loss.backward()
    optimizer.step()
    return {
            'loss':loss.item(),
            'reconstr_loss':reconstr_loss.item(), 
            'kl_loss':kl_loss.item()
            }

trainer = Engine(train_step)

if __name__ == '__main__':
    attach_checkpoint_handler(trainer, {'model': model}, {'optimizer': optimizer})
    attach_ignite(trainer, model)
    trainer.run(train_dl, max_epochs=Config.epochs)


