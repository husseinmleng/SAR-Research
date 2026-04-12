import torch
import random
import torchvision.transforms.functional as TF


class SpeckleNoise:
    """Multiplicative Rayleigh-distributed speckle noise.

    Applied channel-wise to a (C, H, W) tensor.
    noise ~ Rayleigh(sigma), multiplied into each pixel.
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, img):
        # Rayleigh noise: magnitude of 2-D complex Gaussian with std sigma
        noise = torch.sqrt(
            torch.randn_like(img) ** 2 + torch.randn_like(img) ** 2
        ) * self.sigma
        return img * (1.0 + noise)

    def __repr__(self):
        return f'SpeckleNoise(sigma={self.sigma})'


class RandomRotation360:
    """Uniform random rotation in [0, 360) degrees."""

    def __call__(self, img):
        angle = random.uniform(0, 360)
        # img may be a PIL Image or a tensor
        return TF.rotate(img, angle)

    def __repr__(self):
        return 'RandomRotation360()'
