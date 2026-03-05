from pathlib import Path

import torch
from matplotlib import pyplot as plt

import vae

CHECKPOINTS_DIR = Path.cwd() / "vae_model_weights" / f"vae_model.pth"

# hyperparameters
IMAGE_SIZE = [64, 64]
Z_DIM = 256


vae_model = vae.VAE()
vae_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
vae_model.eval()

with torch.no_grad():
    noise = torch.randn(10, Z_DIM)
    image_samples = vae_model.generator(noise)

    plt.imshow(image_samples[1, 0], cmap="gray")
    plt.axis("off")
    plt.show()
