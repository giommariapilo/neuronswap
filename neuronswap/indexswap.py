import torch
import numpy as np
from torch import nn

def swap_layer(layer: nn.Module, permutation_indices: torch.Tensor):
  weights = layer.weight.data
  eq_weights = weights[permutation_indices,:] # slice containing just the eq neurons
  weights = torch.cat((eq_weights, torch.from_numpy(np.delete(weights.numpy(), permutation_indices, axis=0))), dim=0)
  layer.weight.data = weights
  try:
    bias = layer.bias.data
  except:
    return
  eq_bias = bias[permutation_indices]
  bias = torch.cat((eq_bias, torch.from_numpy(np.delete(bias.numpy(), permutation_indices, axis=0))))
  layer.bias.data = bias

def swap_input_channels(layer: nn.Module, previous_layer: nn.Module, permutation_indices: torch.Tensor): # now supporting interface between conv2d and linear
  indexes = permutation_indices
  weights = layer.weight.data
  group_dimension = 1
  if previous_layer.weight.data.shape[0] != weights.shape[1]:
    group_dimension = weights.shape[1] // previous_layer.weight.data.shape[0] # integer division
    if weights.shape[1] % previous_layer.weight.data.shape[0] != 0:
      raise ValueError(f"Incompatible layers: number of neurons of the first layer does not match number of input channels on the second layer\n{weights.shape[1]}%{previous_layer.weight.data.shape[0]}={weights.shape[1] % previous_layer.weight.data.shape[0]}")
    # using a list of ranges
    indexes = [range(index * group_dimension, index * group_dimension + group_dimension) for index in indexes]

  eq_weights = weights[:,indexes]# slice containing just the eq neurons 
  if group_dimension != 1:
    eq_weights = eq_weights.reshape(eq_weights.shape[0],-1) # reshape needed to eliminate dim created by using range
  for stp in reversed(indexes): # itll go out of range if the first range is passed first as it will be deleted 
    weights = torch.from_numpy(np.delete(weights.numpy(), stp, axis=1))
  weights = torch.cat((eq_weights, weights), dim=1)
  layer.weight.data = weights

def swap_bn_layer(layer: nn.BatchNorm2d, permutation_indices: torch.Tensor):
  weights = layer.weight.data
  eq_weights = weights[permutation_indices] # slice containing just the eq neurons
  weights = torch.cat((eq_weights, torch.from_numpy(np.delete(weights.numpy(), permutation_indices, axis=0))))
  layer.weight.data = weights
  
  bias = layer.bias.data
  eq_bias = bias[permutation_indices] # slice containing just the eq neurons
  bias = torch.cat((eq_bias, torch.from_numpy(np.delete(bias.numpy(), permutation_indices, axis=0))))
  layer.bias.data = bias

  avg = layer.running_mean.data
  eq_avg = avg[permutation_indices] # slice containing just the eq neurons
  avg = torch.cat((eq_avg, torch.from_numpy(np.delete(avg.numpy(), permutation_indices, axis=0))))
  layer.running_mean.data = avg

  var = layer.running_var.data
  eq_var = var[permutation_indices] # slice containing just the eq neurons
  var = torch.cat((eq_var, torch.from_numpy(np.delete(var.numpy(), permutation_indices, axis=0))))
  layer.running_var.data = var

def swap(layers_list: list[nn.Module], ppermutations: dict[str, torch.Tensor | list[int]], skip_connections: list[str] = []):
  '''This function takes as inputs the list of layers in the model, a dictionary containing for each layer  
  the indexes of the neurons to be moved to the top of the layer, and an optional list of layers involved in 
  a skip connection. It then modifies each layer putting the weights of each of the specified neurons at the top 
  of the weight matrix. The last layer won't be permutated as it will change the output of the network.
  It then swaps the input channels of the next layer accordingly and performs the same 
  transformation for the biases and the parameters of any batch normalization layer related. If a list of the 
  skip connections is passed, the transformation is inhibited for those layers as it is not yet supported.'''
  
  last_swapped_layer = ''
  for i in range(0,len(layers_list)):
    name, module = layers_list[i]
    if i != len(layers_list) - 1 and name not in skip_connections and name in ppermutations.keys():
      mask = ppermutations[name].long() if type(ppermutations[name]) == torch.Tensor else ppermutations[name]
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        swap_layer(module, mask)
        _, next_module = layers_list[i + 1]
        last_swapped_layer = (name, module)
      elif not isinstance(module, nn.BatchNorm2d) and last_swapped_layer != '': 
        name, module = last_swapped_layer # this is important ... if the current layer is not linerar or convolutional,                                       
      if isinstance(next_module, (nn.Linear, nn.Conv2d)) and not isinstance(module, nn.BatchNorm2d):
        swap_input_channels(next_module, module, mask)
      elif isinstance(next_module, (nn.BatchNorm2d)):
        swap_bn_layer(next_module, mask)
        _, next_module = layers_list[i + 2]
        swap_input_channels(next_module, module, mask)

