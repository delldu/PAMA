"""
Copyright 2021 Mahmoud Afifi.
 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN: 
 Controlling Colors of GAN-Generated and Real Images via Color Histograms." 
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via 
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class RGBuvHistBlock(nn.Module):
  def __init__(self, h=64, insz=150, sigma=0.02):
    """ Computes the RGB-uv histogram feature of a given image.
    """
    super(RGBuvHistBlock, self).__init__()
    self.h = h
    self.insz = insz
    self.sigma = sigma
    # h = 256
    # insz = 64
    # sigma = 0.02

  def forward(self, x):
    x = torch.clamp(x, 0, 1)
    if x.shape[2] > self.insz or x.shape[3] > self.insz:
      x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                              mode='bilinear', align_corners=False)
    else:
      x_sampled = x

    L = x_sampled.shape[0]  # size of mini-batch
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]

    X = torch.unbind(x_sampled, dim=0)
    hists = torch.zeros((x_sampled.shape[0], 3, self.h, self.h)).to(x.device)

    # np.linspace(-3, 3, 256).shape -- (256,)
    offset = torch.unsqueeze(torch.linspace(-3, 3, num=self.h), dim=0).to(x.device) # [1, 256]
    for l in range(L):
      I = torch.t(torch.reshape(X[l], (3, -1)))
      II = torch.pow(I, 2)
      Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + 1e-6), dim=1)

      Iu0 = torch.unsqueeze(torch.log(I[:, 0] + 1e-6) - torch.log(I[:, 1] + 1e-6), dim=1)
      Iv0 = torch.unsqueeze(torch.log(I[:, 0] + 1e-6) - torch.log(I[:, 2] + 1e-6), dim=1)
      diff_u0 = abs(Iu0 - offset)
      diff_v0 = abs(Iv0 - offset)

      diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)), 2) / self.sigma ** 2
      diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)), 2) / self.sigma ** 2
      diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
      diff_v0 = 1 / (1 + diff_v0)

      diff_u0 = diff_u0.type(torch.float32)
      diff_v0 = diff_v0.type(torch.float32)
      a = torch.t(Iy * diff_u0)
      hists[l, 0, :, :] = torch.mm(a, diff_v0)

      Iu1 = torch.unsqueeze(torch.log(I[:, 1] + 1e-6) - torch.log(I[:, 0] + 1e-6), dim=1)
      Iv1 = torch.unsqueeze(torch.log(I[:, 1] + 1e-6) - torch.log(I[:, 2] + 1e-6), dim=1)
      diff_u1 = abs(Iu1 - offset)
      diff_v1 = abs(Iv1 - offset)

      diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)), 2) / self.sigma ** 2
      diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)), 2) / self.sigma ** 2
      diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
      diff_v1 = 1 / (1 + diff_v1)

      diff_u1 = diff_u1.type(torch.float32)
      diff_v1 = diff_v1.type(torch.float32)
      a = torch.t(Iy * diff_u1)
      hists[l, 1, :, :] = torch.mm(a, diff_v1)

      Iu2 = torch.unsqueeze(torch.log(I[:, 2] + 1e-6) - torch.log(I[:, 0] + 1e-6), dim=1)
      Iv2 = torch.unsqueeze(torch.log(I[:, 2] + 1e-6) - torch.log(I[:, 1] + 1e-6), dim=1)
      diff_u2 = abs(Iu2 - offset)
      diff_v2 = abs(Iv2 - offset)

      diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)), 2) / self.sigma ** 2
      diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)), 2) / self.sigma ** 2
      diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
      diff_v2 = 1 / (1 + diff_v2)
      diff_u2 = diff_u2.type(torch.float32)
      diff_v2 = diff_v2.type(torch.float32)
      a = torch.t(Iy * diff_u2)
      hists[l, 2, :, :] = torch.mm(a, diff_v2)

    # normalization
    hists_normalized = hists / (
        ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + 1e-6)

    return hists_normalized
