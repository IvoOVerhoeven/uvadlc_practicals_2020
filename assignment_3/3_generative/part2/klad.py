import torch
import models
import torch.nn.functional as F
import torchvision
from mnist import mnist
import train_pl


model = train_pl.GAN(hidden_dims_gen=[200],
                hidden_dims_disc=[200],
                dp_rate_gen=0.1,
                dp_rate_disc=0.1,
                z_dim=32,
                lr=0.02)

B = 2
imgs = model.interpolate(B,5).permute(1,0,2,3,4)
img_grid = torchvision.utils.make_grid(imgs.reshape(imgs.shape[0]*imgs.shape[1], *imgs.shape[-3:]), nrow=B)
torchvision.utils.save_image(img_grid, './GAN_logs/lightning_logs/test.png')

mnist

generator = models.GeneratorMLP()
discriminator = models.DiscriminatorMLP()

generator.parameters

x = torch.rand((256,32))
generator(x)
y_fake = discriminator(generator(x))
t_fake = torch.zeros((x.shape[0],1))

(torch.round(y_fake) == t_fake[:,None]).sum()

F.binary_cross_entropy(y_fake, t_fake)

interpolation_steps = 5
left = torch.rand((100,32))
right = torch.rand((100,32))

left = torch.normal(0,1,(100, 32))
right = torch.normal(0,1,(100, 32))

interpolated = []
for i in reversed(range(interpolation_steps+2)):
    mix = i/(interpolation_steps+1)
    mixed_latents = mix * left + (1-mix) * right
    interpolated.append(generator(mixed_latents))

x = torch.stack(interpolated, dim=1)
x.shape
torch.normal(0,1,(1000, 20))

F.binary_cross_entropy(generator(x), x)

torch.ones()