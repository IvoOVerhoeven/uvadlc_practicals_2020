import torch
import torch.nn.functional as F
import numpy as np
import mlp_encoder_decoder
import cnn_encoder_decoder
import utils
import scipy 
from scipy.stats import norm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from train_pl import VAE

import cnn_encoder_decoder
encoder = cnn_encoder_decoder.CNNEncoder()
decoder = cnn_encoder_decoder.CNNDecoder()

x = torch.randint(0, 2,(256,1,28,28)).float()
mean, log_std = encoder(x)
z = utils.sample_reparameterize(mean, log_std)
x_hat = torch.sigmoid(decoder(z))
torch.max(x_hat)

encoder = mlp_encoder_decoder.MLPEncoder()
decoder = mlp_encoder_decoder.MLPDecoder()
grid_size = 20

VAE = VAE.load_from_checkpoint('./VAE_logs/lightning_logs/version_70/checkpoints/epoch=39.ckpt')
visualize_manifold(VAE.decoder

# Compute the desired quantiles
d_grid=1/(grid_size+1)
ppfs = norm.ppf(torch.arange(0.5/(grid_size+1), 
                             (grid_size+0.5)/(grid_size+1), d_grid))
ppfs = torch.tensor(ppfs)

# Convert the quantiles to a [grid_size**2,2] latent input tensor
z1, z2 = torch.meshgrid(ppfs, ppfs)
z = torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=1).float()

with torch.no_grad():
    xhat = torch.sigmoid(VAE.decoder(z).detach())
imgs = torch.round(xhat)

plt.imshow(make_grid(xhat, nrow=grid_size).permute(1,2,0))

import cnn_encoder_decoder



(x==torch.round(x_hat)).sum().item()/torch.prod(torch.tensor(x.size())).item()



Lrec = torch.sum(F.binary_cross_entropy(x_hat, x, reduction='none'), dim=(1,2,3))
Lreg = utils.KLD(mean, log_std)
(Lrec-Lreg)*torch.log2(torch.exp(torch.tensor([1.0])))/torch.prod(torch.tensor(x.size()[1:]))

np.log2(np.exp(1))
utils.elbo_to_bpd(Lrec+Lreg, x.size())

decoder = mlp_encoder_decoder.MLPDecoder()

torch.min(torch.normal(0,1,size=(100,20)))

utils.elbo_to_bpd(Lrec+Lreg, x.size())

torch.normal(10, 0.1,(256,20))
u = torch.normal(0,1,(256,20))

torch.mean(torch.exp(log_std) * u + mean, dim=0)
torch.mean(mean, dim=0)
mean.size()

torch.exp(log_std) * u
mean



torch.tensor(x.size()[1:])

torch.randn_like(log_std).shape

out_shape = [1,28,28]
x = torch.rand((256,1*28*28))
x.reshape((x.shape[0],*out_shape)).shape
np.prod(out_shape)

decoder = mlp_encoder_decoder.MLPDecoder(hidden_dims=[100,10])
decoder
decoder(torch.rand((256,20))).shape

log_var = 2*log_std
torch.exp(log_var)
torch.sum((torch.exp(log_var) + mean ** 2 - 1 - log_var)/2, dim=1).shape



decoder = mlp_encoder_decoder.MLPDecoder(z_dim=2, hidden_dims=[100,500])
grid_size=20
d_grid=1/(grid_size+1)
ppfs = norm.ppf(torch.arange(0.5/(grid_size+1), (grid_size+0.5)/(grid_size+1), d_grid))
ppfs = torch.tensor(ppfs)
z1, z2 = torch.meshgrid(ppfs, ppfs)
z = torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=1).float()
imgs = decoder(z).detach()

plt.imshow(make_grid(imgs, nrow=20).permute(1,2,0))

from PIL import Image

z = torch.normal(0,1, size=(64, 20))
x_mean = torch.sigmoid(decoder(z))
x_samples = torch.round(x_mean)
img_grid = make_grid(x_samples, nrow=8)
img_grid = img_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
img_grid = img_grid.to('cpu', torch.uint8).numpy()

Image.fromarray(img_grid)
img_grid
