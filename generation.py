import torch
from pathlib import Path
from PIL import Image
import numpy as np
from ignite.handlers import Checkpoint
from ignite.engine import Engine
from lib.model import VAE

def load_checkpoint(checkpoint_dir, device='cuda'):
    # Load the model checkpoint
    checkpoint_path = Path(checkpoint_dir)
    
    gen = VAE(28*28).to(device)
    
    checkpoint = torch.load(checkpoint_path / 'model_checkpoint_200.pt', map_location=device)
    Checkpoint.load_objects(to_load={'model': gen}, checkpoint=checkpoint)
    
    return gen

def generate_image(model, device='cuda'):
    z = torch.randn(1, 20).to(device)
    
    with torch.no_grad():
        gen_img = model.decoder(z)
    
    gen_img = gen_img.view(28, 28).cpu().detach().numpy()
    
    gen_img = (gen_img * 255).astype(np.uint8)
    
    return gen_img

def save_image(image_array, output_path):
    img = Image.fromarray(image_array, mode='L')  # 'L' mode for grayscale
    
    img.save(output_path)
    print(f"Generated image saved to {output_path}")

def main( checkpoint_dir, output_path, device='cuda'):
    gen = load_checkpoint(checkpoint_dir, device=device)
    
    generated_img = generate_image(gen, device=device)
    
    save_image(generated_img, output_path)

if __name__ == "__main__":
    checkpoint_dir = './weights'  
    output_path = './generated_image.png' 
    
    main( checkpoint_dir, output_path, device='cuda')
