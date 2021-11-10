import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import unet_encoder
from .decoders import unet_decoder


class Unet(nn.Module):
  def __init__(self, nbase, nout, kernel_size):
    """
    """
    super(Unet, self).__init__()
    self.nbase = nbase
    self.nout = nout
    self.kernel_size = kernel_size
    self.unet_encoder = unet_encoder(nbase, kernel_size)
    nbaseup = nbase[1:]
    nbaseup.append(nbase[-1])
    self.unet_decoder = unet_decoder(nbaseup, kernel_size)
    self.output = nn.Conv2d(nbase[1], self.nout, kernel_size,
                            padding=kernel_size//2)

  def forward(self, data):
    """
    
    """
    T0 = self.unet_encoder(data)
    T0 = self.unet_decoder(T0)
    T0 = self.output(T0)
    return T0

  def save_model(self, filename):
    """
    
    """
    torch.save(self.state_dict(), filename)

  def load_model(self, filename, cpu=False):
    """
    
    """
    if not cpu:
      self.load_state_dict(torch.load(filename))
    else:
      self.__init__(self.nbase,
                    self.nout,
                    self.kernel_size,
                    self.concatenation)

      self.load_state_dict(torch.load(filename,
                                      map_location=torch.device('cpu')))