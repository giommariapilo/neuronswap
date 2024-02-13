import torch
import torch.nn as nn

from .modulexplore import get_layers_list, get_skipped_layers
from .optimizerswap import optimizer_swap as oswap
from .matrixswap import swap as mswap
from .indexswap import swap as iswap

def permutate(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor | list[int]], skip_connections: list[str] = []):

  permutate_function = mswap if isinstance(list(permutations.values())[0], torch.Tensor) and len(list(permutations.values())[0].shape) > 1 else iswap
  return permutate_function(layers_list, permutations, skip_connections)

