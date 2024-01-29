import torch
import numpy as np
from torch import nn

def permutate(module: nn.Module, permutation_matrix: torch.Tensor):
  '''
  This function takes as arguments a layer and the permutation matrix for that layer. 
  It then permutates the layer swapping the rows. It is implemented as a matrix multiplication.
  '''
  module.weight.data = torch.matmul(module.weight.data, permutation_matrix)
  