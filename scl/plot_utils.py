import numpy
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from utils import get_image_stats
import torchvision

# def _show_attention_old(self, batch):
#     mu, sigma = get_image_stats(self.hparams.dataset)
#     mu = torch.tensor(mu, device=self.device)[:, None, None]
#     sigma = torch.tensor(sigma, device=self.device)[:, None, None]
#     num_imgs = 2
#     x_i, xhat_i, x_j = batch
#     x_i, xhat_i, x_j = x_i[:num_imgs] * sigma + mu, xhat_i[:num_imgs] * sigma + mu, x_j[:num_imgs] * sigma + mu
#     out = self.model(x_i)
#     attn1 = self.model.encoder[-1].attn.weights[:,:,0,1:]  # (N,h,L,S) -> (N,h,S-1)
#     print(attn1.shape)
#     attn_size = int( attn1.shape[-1] ** 0.5)
#     attn1 = torch.mean(attn1, dim=1).view(num_imgs, 1, attn_size,attn_size)  # (N,h,S) -> (N,S) -> (N,1,S) -> (N,1,H,W)
#     attn1 = F.interpolate(attn1, size=self.hparams.imgsize)
#     attn1 = attn1.repeat(1, 3, 1, 1)
#     out = self.model(xhat_i)
#     attn2 = self.model.encoder[-1].attn.weights[:,:,0,1:]
#     attn2 = torch.mean(attn2, dim=1).view(num_imgs, 1, attn_size,attn_size)  # (N,h,S) -> (N,S) -> (N,1,S) -> (N,1,H,W)
#     attn2 = F.interpolate(attn2, size=self.hparams.imgsize)
#     attn2 = attn2.repeat(1, 3, 1, 1)
#     out = self.model(x_j)
#     attn3 = self.model.encoder[-1].attn.weights[:,:,0,1:]
#     attn3 = torch.mean(attn3, dim=1).view(num_imgs, 1, attn_size,attn_size)  # (N,h,S) -> (N,S) -> (N,1,S) -> (N,1,H,W)
#     attn3 = F.interpolate(attn3, size=self.hparams.imgsize)
#     attn3 = attn3.repeat(1, 3, 1, 1)
#     print(attn1.shape)
#     images = zip(x_i, attn1, xhat_i, attn2, x_j, attn3)
#     x_ = [image for tupl in images for image in tupl]
#     grid = torchvision.utils.make_grid(torch.stack(x_), nrow=6)
#     plt.figure(figsize=(50, 50))
#     fname = "plots/attn_{}.png".format(self.hparams.dataset)
#     plt.imshow(grid)
#     plt.savefig(fname)
#     plt.close()
#     #self.logger.experiment.add_image(self.hparams.dataset + "_attn", grid, self.global_step)


# def _show_attention(self, images, attn, num_imgs=2):
#     x_i, xhat_i, x_j = images
#     attn_i, attnhat_i, attn_j = attn
#     fig, ax = plt.subplots(len(images), len(images) + attn[0][0].shape[1] + 1, figsize=(50, 50))
#     for k in range(len(images)):
#         x_i, xhat_i, x_j = images[k]
#         attn_i, attnhat_i, attn_j = attn[k]
#         for l in range(len(attn)):
#             pass
#     attn_size = int(attn1.shape[-1] ** 0.5)
#     attn1 = torch.mean(attn1, dim=1).view(num_imgs, 1, attn_size,attn_size)  # (N,h,S) -> (N,S) -> (N,1,S) -> (N,1,H,W)
#     attn1 = F.interpolate(attn1, size=self.hparams.imgsize)
#     attn1 = attn1.repeat(1, 3, 1, 1)
#     out = self.model(xhat_i)
#     attn2 = self.model.encoder[-1].attn.weights[:,:,0,1:]
#     attn2 = torch.mean(attn2, dim=1).view(num_imgs, 1, attn_size,attn_size)  # (N,h,S) -> (N,S) -> (N,1,S) -> (N,1,H,W)
#     attn2 = F.interpolate(attn2, size=self.hparams.imgsize)
#     attn2 = attn2.repeat(1, 3, 1, 1)
#     out = self.model(x_j)
#     attn3 = self.model.encoder[-1].attn.weights[:,:,0,1:]
#     attn3 = torch.mean(attn3, dim=1).view(num_imgs, 1, attn_size,attn_size)  # (N,h,S) -> (N,S) -> (N,1,S) -> (N,1,H,W)
#     attn3 = F.interpolate(attn3, size=self.hparams.imgsize)
#     attn3 = attn3.repeat(1, 3, 1, 1)
#     print(attn1.shape)
#     images = zip(x_i, attn1, xhat_i, attn2, x_j, attn3)
#     x_ = [image for tupl in images for image in tupl]
#     grid = torchvision.utils.make_grid(torch.stack(x_), nrow=6)
#     self.logger.experiment.add_image(self.hparams.dataset + "_attn", grid, self.global_step)
