"""Conditional DCGAN for SAR image synthesis.

Usage:
    python gan.py --data_root mstar_sampled_data --output_dir gan_output \
                  --epochs 200 --n_generate 100
"""

import os
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ConditionalGenerator(nn.Module):
    """z(100-D) + one-hot class(10-D) -> (1, 100, 100) grayscale image."""

    def __init__(self, noise_dim=100, n_classes=10, img_size=100):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        in_dim = noise_dim + n_classes

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512 * 6 * 6),
            nn.BatchNorm1d(512 * 6 * 6),
            nn.ReLU(True),
        )

        self.deconv = nn.Sequential(
            # (B, 512, 6, 6)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (B, 256, 12, 12)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (B, 128, 24, 24)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (B, 64, 48, 48)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1,
                               output_padding=0),
            nn.Tanh(),
            # (B, 1, 96, 96)  -> upsample to 100x100
        )
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear',
                                    align_corners=False)

    def forward(self, z, labels_onehot):
        x = torch.cat([z, labels_onehot], dim=1)
        x = self.fc(x).view(-1, 512, 6, 6)
        x = self.deconv(x)
        return self.upsample(x)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class ConditionalDiscriminator(nn.Module):
    """(1, 100, 100) image + spatially-broadcast class one-hot -> real/fake score."""

    def __init__(self, n_classes=10, img_size=100):
        super().__init__()
        self.n_classes = n_classes
        self.img_size  = img_size
        in_ch = 1 + n_classes   # image channels + class embedding channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 64, 50, 50)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 128, 25, 25)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 256, 12, 12)
            nn.Conv2d(256, 1, kernel_size=12),
            # (B, 1, 1, 1)
        )

    def forward(self, img, labels_onehot):
        B = img.size(0)
        # Spatially broadcast class label
        cls_map = labels_onehot.view(B, self.n_classes, 1, 1).expand(
            B, self.n_classes, self.img_size, self.img_size
        )
        x = torch.cat([img, cls_map], dim=1)
        out = self.conv(x)
        return out.view(B, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gan(args, device):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    n_classes = len(dataset.classes)
    class_names = dataset.classes
    print('Classes:', class_names)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    G = ConditionalGenerator(args.noise_dim, n_classes, args.img_size).to(device)
    D = ConditionalDiscriminator(n_classes, args.img_size).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.join('model', 'gan'), exist_ok=True)

    d_loss_history = []
    wgan_mode = False  # switch to WGAN-GP if D plateaus

    for epoch in range(1, args.epochs + 1):
        epoch_d_loss = 0.0
        n_batches = 0

        for real_imgs, real_labels in loader:
            B = real_imgs.size(0)
            real_imgs   = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # One-hot encode
            real_onehot = F.one_hot(real_labels, n_classes).float()

            # Sample noise and fake labels
            z = torch.randn(B, args.noise_dim, device=device)
            fake_labels = torch.randint(0, n_classes, (B,), device=device)
            fake_onehot = F.one_hot(fake_labels, n_classes).float()

            # ----- Train D -----
            opt_D.zero_grad()
            fake_imgs = G(z, fake_onehot).detach()

            real_score = D(real_imgs, real_onehot)
            fake_score = D(fake_imgs, fake_onehot)

            d_loss = bce(real_score, torch.ones_like(real_score)) + \
                     bce(fake_score, torch.zeros_like(fake_score))
            d_loss.backward()
            opt_D.step()
            epoch_d_loss += d_loss.item()

            # ----- Train G -----
            opt_G.zero_grad()
            z = torch.randn(B, args.noise_dim, device=device)
            fake_labels2 = torch.randint(0, n_classes, (B,), device=device)
            fake_onehot2 = F.one_hot(fake_labels2, n_classes).float()
            fake_imgs2 = G(z, fake_onehot2)
            g_score = D(fake_imgs2, fake_onehot2)
            g_loss = bce(g_score, torch.ones_like(g_score))
            g_loss.backward()
            opt_G.step()
            n_batches += 1

        avg_d = epoch_d_loss / max(n_batches, 1)
        d_loss_history.append(avg_d)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{args.epochs}]  D_loss: {avg_d:.4f}  G_loss: {g_loss.item():.4f}')

        # Check for D plateau (no improvement in 20 epochs)
        if not wgan_mode and len(d_loss_history) >= 20:
            recent = d_loss_history[-20:]
            if max(recent) - min(recent) < 0.01:
                print('[INFO] D loss plateau detected — consider switching to WGAN-GP')

        if epoch % 50 == 0:
            ckpt_path = os.path.join('model', 'gan', f'generator_ep{epoch}.pt')
            torch.save(G.state_dict(), ckpt_path)
            print(f'  Saved checkpoint: {ckpt_path}')

    final_path = os.path.join('model', 'gan', 'generator.pt')
    torch.save(G.state_dict(), final_path)
    print(f'Generator saved to {final_path}')

    return G, n_classes, class_names


# ---------------------------------------------------------------------------
# Image generation & quality check
# ---------------------------------------------------------------------------

def generate_images(G, n_classes, class_names, args, device):
    """Generate n_generate images per class and save to output_dir/<class>/."""
    G.eval()

    for cls_idx in range(n_classes):
        cls_dir = os.path.join(args.output_dir, class_names[cls_idx])
        os.makedirs(cls_dir, exist_ok=True)

        onehot = F.one_hot(
            torch.tensor([cls_idx] * args.n_generate, device=device),
            n_classes
        ).float()
        z = torch.randn(args.n_generate, args.noise_dim, device=device)

        with torch.no_grad():
            fake = G(z, onehot)   # (N, 1, H, W) in [-1, 1]

        # Denormalize to [0, 255]
        fake = ((fake + 1) / 2 * 255).clamp(0, 255).byte()
        fake = fake.cpu().numpy()

        for i in range(args.n_generate):
            img = fake[i, 0]   # (H, W) uint8
            pil = Image.fromarray(img, mode='L')
            pil.save(os.path.join(cls_dir, f'gen_{i:04d}.jpeg'))

    print(f'Generated {args.n_generate} images × {n_classes} classes -> {args.output_dir}/')


def classifier_quality_check(n_classes, class_names, args, device):
    """Run trained EmbeddingCNN+Linear_model on generated images; warn if <70%."""
    try:
        from cnn import EmbeddingCNN, Linear_model
    except ImportError:
        print('[WARNING] cnn.py not importable — skipping quality check')
        return

    best_cnn_path = None
    for root, _, files in os.walk('model'):
        for f in files:
            if f.endswith('.pth'):
                best_cnn_path = os.path.join(root, f)
                break
        if best_cnn_path:
            break

    if not best_cnn_path:
        print('[WARNING] No CNN checkpoint found — skipping quality check')
        return

    print(f'Quality check using checkpoint: {best_cnn_path}')

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    gen_dataset = datasets.ImageFolder(root=args.output_dir, transform=transform)
    gen_classes = gen_dataset.classes
    loader = torch.utils.data.DataLoader(gen_dataset, batch_size=64)

    cnn = EmbeddingCNN(100, 64, 32, 4).to(device)
    head = Linear_model(n_classes).to(device)

    try:
        state = torch.load(best_cnn_path, map_location=device)
        # The checkpoint may be a full gnnModel — try to load cnn sub-state
        cnn_state = {k.replace('cnn_feature.', ''): v
                     for k, v in state.items() if k.startswith('cnn_feature.')}
        if cnn_state:
            cnn.load_state_dict(cnn_state)
        else:
            print('[WARNING] No cnn_feature keys in checkpoint — quality check may be inaccurate')
    except Exception as e:
        print(f'[WARNING] Could not load CNN weights: {e}')
        return

    cnn.eval(); head.eval()
    correct = [0] * n_classes
    totals  = [0] * n_classes

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = cnn(imgs)
            logits = head(feats)
            preds = logits.argmax(1).cpu()
            for p, l in zip(preds, labels):
                totals[l.item()] += 1
                if p.item() == l.item():
                    correct[l.item()] += 1

    print('\n=== GAN Quality Check ===')
    for i, name in enumerate(gen_classes):
        acc = 100.0 * correct[i] / max(totals[i], 1)
        flag = '' if acc >= 70.0 else '  [WARNING: below 70%]'
        print(f'  {name}: {correct[i]}/{totals[i]} = {acc:.1f}%{flag}')
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Conditional DCGAN for SAR synthesis')
    p.add_argument('--data_root',   default='mstar_sampled_data')
    p.add_argument('--output_dir',  default='gan_output')
    p.add_argument('--epochs',      default=200, type=int)
    p.add_argument('--batch_size',  default=32,  type=int)
    p.add_argument('--noise_dim',   default=100, type=int)
    p.add_argument('--img_size',    default=100, type=int)
    p.add_argument('--lr_g',        default=0.0002, type=float)
    p.add_argument('--lr_d',        default=0.0002, type=float)
    p.add_argument('--n_generate',  default=100, type=int)
    p.add_argument('--seed',        default=42, type=int)
    p.add_argument('--gpu',         default='0')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    G, n_classes, class_names = train_gan(args, device)
    generate_images(G, n_classes, class_names, args, device)
    classifier_quality_check(n_classes, class_names, args, device)
