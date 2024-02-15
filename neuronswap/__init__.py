import torch
import torch.nn as nn

from .modulexplore import get_layers_list, get_skipped_layers
from .optimizermatrixswap import swap as moswap
from .optimizerindexswap import swap as ioswap
from .matrixswap import swap as mswap
from .indexswap import swap as iswap

def permute(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor | list[int]], skip_connections: list[str] = []):
  '''This function takes as inputs the list of layers of a model, a dictionary containing the 
  permutation matrix associated to each layer or the indexes of the neurons to be moved at the top 
  of the layer, and an optional list of skip connections. It then calls one of the two 
  implementations of the swap method to swap the neurons. The last layer won't be permuted as it 
  will change the output of the network. If it receives a list  of skip connections, those 
  layers are not permuted as permutation of skip connection layers is not supported yet.'''
  type_check = None
  shape_check = 0
  for permutation in permutations.values():
    if type_check != None:
      if type(permutation) != type_check:
        raise TypeError(f'ERROR: values inside "permutations" must be of the same type, expected{type_check}, got instead {type(permutation)}!')
      if type(permutation) == torch.Tensor and len(permutation.shape) != shape_check:
        raise IndexError(f'ERROR: tensors must have the same number of dimensions, expected {shape_check}, got instead {len(permutation.shape)}!')
    else:
      type_check = type(permutation)
      if type(permutation) == torch.Tensor:
        shape_check = len(permutation.shape)
    
  permute_function = mswap if isinstance(list(permutations.values())[0], torch.Tensor) and len(list(permutations.values())[0].shape) > 1 else iswap
  return permute_function(layers_list, permutations, skip_connections)

def permute_optimizer(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer,skip_connections: list[str] = []):
  ''''''
  type_check = None
  shape_check = 0
  for permutation in permutations.values():
    print(type(permutation))
    if type_check != None:
      if type(permutation) != type_check:
        raise TypeError(f'ERROR: values inside "permutations" must be of the same type, expected{type_check}, got instead {type(permutation)}!')
      if type(permutation) == torch.Tensor and len(permutation.shape) != shape_check:
        raise IndexError(f'ERROR: tensors must have the same number of dimensions, expected {shape_check}, got instead {len(permutation.shape)}!')
    else:
      type_check = type(permutation)
      if type(permutation) == torch.Tensor:
        shape_check = len(permutation.shape)
  permute_function = moswap if isinstance(list(permutations.values())[0], torch.Tensor) and len(list(permutations.values())[0].shape) > 1 else ioswap
  return permute_function(layers_list, permutations, model, optimizer, skip_connections)
