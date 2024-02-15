import torch
import numpy as np
from torch import nn, optim
from .modulexplore import create_layer_indices_dict

def swap_layer(layer: nn.Module, permutation_indices: torch.Tensor | list[int], index: int, optimizer: optim.Optimizer):
  weights = optimizer.state_dict()['state'][index]['momentum_buffer']
  device = weights.get_device()
  weights = weights if device == -1 else weights.cpu()
  eq_weights = weights[permutation_indices,:] # slice containing just the eq neurons
  weights = torch.cat((eq_weights, torch.from_numpy(np.delete(weights.numpy(), permutation_indices, axis=0))), dim=0)
  optimizer.state_dict()['state'][index]['momentum_buffer'] = weights if device == -1 else weights.cuda()
  try:
    layer.get_parameter('bias')
  except:
    return
  else:
    bias = optimizer.state_dict()['state'][index+1]['momentum_buffer']
    bias = bias if device == -1 else bias.cpu()
    eq_bias = bias[permutation_indices]
    bias = torch.cat((eq_bias, torch.from_numpy(np.delete(bias.numpy(), permutation_indices, axis=0))))
    optimizer.state_dict()['state'][index+1]['momentum_buffer'] = bias if device == -1 else bias.cuda()

def swap_input_channels(permutation_indices: torch.Tensor | list[int], layer_index: int, previous_layer_index: int, optimizer: optim.Optimizer): 
  indexes = permutation_indices
  weights = optimizer.state_dict()['state'][layer_index]['momentum_buffer']
  device = weights.get_device()
  weights = weights if device == -1 else weights.cpu()
  previous_weights = optimizer.state_dict()['state'][previous_layer_index]['momentum_buffer']
  group_dimension = 1
  if previous_weights.shape[0] != weights.shape[1]:
    group_dimension = weights.shape[1] // previous_weights.shape[0] # integer division
    if weights.shape[1] % previous_weights.shape[0] != 0:
      raise ValueError(f"Incompatible layers: number of neurons of the first layer does not match number of input channels on the second layer\n{weights.shape[1]}%{previous_weights.shape[0]}={weights.shape[1] % previous_weights.shape[0]}")
    # using a list of ranges
    indexes = [range(index * group_dimension, index * group_dimension + group_dimension) for index in indexes]

  eq_weights = weights[:,indexes]# slice containing just the eq neurons 
  if group_dimension != 1:
    eq_weights = eq_weights.reshape(eq_weights.shape[0],-1) # reshape needed to eliminate dim created by using range
  for stp in reversed(indexes): # itll go out of range if the first range is passed first as it will be deleted 
    weights = torch.from_numpy(np.delete(weights.numpy(), stp, axis=1))
  weights = torch.cat((eq_weights, weights), dim=1)
  optimizer.state_dict()['state'][layer_index]['momentum_buffer'] = weights if device == -1 else weights.cuda()

def swap_bn_layer(permutation_indices: torch.Tensor | list[int], index: int, optimizer: optim.Optimizer):
  weights = optimizer.state_dict()['state'][index]['momentum_buffer']
  device = weights.get_device()
  weights = weights if device == -1 else weights.cpu()
  eq_weights = weights[permutation_indices] # slice containing just the eq neurons
  weights = torch.cat((eq_weights, torch.from_numpy(np.delete(weights.numpy(), permutation_indices, axis=0))))
  optimizer.state_dict()['state'][index]['momentum_buffer'] = weights if device == -1 else weights.cuda()
  
  bias = optimizer.state_dict()['state'][index + 1]['momentum_buffer']
  bias = bias if device == -1 else bias.cpu()
  eq_bias = bias[permutation_indices] # slice containing just the eq neurons
  bias = torch.cat((eq_bias, torch.from_numpy(np.delete(bias.numpy(), permutation_indices, axis=0))))
  optimizer.state_dict()['state'][index + 1]['momentum_buffer'] = bias if device == -1 else bias.cuda()

def swap(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor | list[int]], model: nn.Module, optimizer: optim.Optimizer,skip_connections: list[str] = []):
  ''''''
  layer_indices = create_layer_indices_dict(model)
  last_swapped_layer = ''
  for i in range(0,len(layers_list)):
    name, module = layers_list[i]
    if i != len(layers_list) - 1 and name not in skip_connections and name in permutations.keys():
      mask = permutations[name].long() if type(permutations[name]) == torch.Tensor else permutations[name]
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        index = layer_indices[name]
        swap_layer(module, mask, index, optimizer)
        next_name, next_module = layers_list[i + 1]
        last_swapped_layer = (name, module)
      elif not isinstance(module, nn.BatchNorm2d) and last_swapped_layer != '': 
        name, module = last_swapped_layer # this is important ... if the current layer is not linerar or convolutional,                                       
      if isinstance(next_module, (nn.Linear, nn.Conv2d)) and not isinstance(module, nn.BatchNorm2d):
        index = layer_indices[next_name]
        previous_index = layer_indices[module]
        swap_input_channels(mask, index, previous_index, optimizer)
      elif isinstance(next_module, (nn.BatchNorm2d)):
        index = layer_indices[next_name]
        swap_bn_layer(mask, index, optimizer)
        next_name, next_module = layers_list[i + 2]
        index = layer_indices[next_name]
        previous_index = layer_indices[name]
        swap_input_channels(mask, index, previous_index, optimizer)
  