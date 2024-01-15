import torch
import numpy as np
from torch import nn

def swap_lin_conv_layers(layer: nn.Module, equilibrium_neurons):
  weights = layer.weight.data
  eq_weights = weights[equilibrium_neurons,:] # slice containing just the eq neurons
  weights = torch.cat((eq_weights, torch.from_numpy(np.delete(weights.numpy(), equilibrium_neurons, axis=0))), dim=0)
  layer.weight.data = weights
  try:
    bias = layer.bias.data
  except:
    return
  eq_bias = bias[equilibrium_neurons]
  bias = torch.cat((eq_bias, torch.from_numpy(np.delete(bias.numpy(), equilibrium_neurons, axis=0))))
  layer.bias.data = bias

def swap_input_channels(module: nn.Module, previous_module: nn.Module, equilibrium_neurons): # now supporting interface between conv2d and linear
  indexes = equilibrium_neurons
  weights = module.weight.data
  group_dimension = 1
  if previous_module.weight.data.shape[0] != weights.shape[1]:
    group_dimension = weights.shape[1] // previous_module.weight.data.shape[0] # integer division
    if weights.shape[1] % previous_module.weight.data.shape[0] != 0:
      raise ValueError(f"Incompatible layers: number of neurons of the first layer does not match number of input channels on the second layer\n{weights.shape[1]}%{previous_module.weight.data.shape[0]}={weights.shape[1] % previous_module.weight.data.shape[0]}")
    # using a list of ranges
    indexes = [range(index * group_dimension, index * group_dimension + group_dimension) for index in indexes]

  eq_weights = weights[:,indexes]# slice containing just the eq neurons 
  if group_dimension != 1:
    eq_weights = eq_weights.reshape(eq_weights.shape[0],-1) # reshape needed to eliminate dim created by using range
  for stp in reversed(indexes): # itll go out of range if the first range is passed first as it will be deleted 
    weights = torch.from_numpy(np.delete(weights.numpy(), stp, axis=1))
  weights = torch.cat((eq_weights, weights), dim=1)
  module.weight.data = weights

def swap_bn_modules(module: nn.BatchNorm2d, equilibrium_neurons):
  weights = module.weight.data
  eq_weights = weights[equilibrium_neurons] # slice containing just the eq neurons
  weights = torch.cat((eq_weights, torch.from_numpy(np.delete(weights.numpy(), equilibrium_neurons, axis=0))))
  module.weight.data = weights
  
  bias = module.bias.data
  eq_bias = bias[equilibrium_neurons] # slice containing just the eq neurons
  bias = torch.cat((eq_bias, torch.from_numpy(np.delete(bias.numpy(), equilibrium_neurons, axis=0))))
  module.bias.data = bias

  avg = module.running_mean.data
  eq_avg = avg[equilibrium_neurons] # slice containing just the eq neurons
  avg = torch.cat((eq_avg, torch.from_numpy(np.delete(avg.numpy(), equilibrium_neurons, axis=0))))
  module.running_mean.data = avg

  var = module.running_var.data
  eq_var = var[equilibrium_neurons] # slice containing just the eq neurons
  var = torch.cat((eq_var, torch.from_numpy(np.delete(var.numpy(), equilibrium_neurons, axis=0))))
  module.running_var.data = var

def swap(module_list: list[nn.Module], equilibrium_mask: dict[str, torch.Tensor], skip_connections: list = []):
  last_swapped_layer = ''
  for i in range(0,len(module_list)):
    name, module = module_list[i]
    if i != len(module_list) - 1 and name not in skip_connections and name in equilibrium_mask.keys():
      mask = equilibrium_mask[name].int() if type(equilibrium_mask[name]) == torch.Tensor else equilibrium_mask[name]
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        swap_lin_conv_layers(module, mask)
        _, next_module = module_list[i + 1]
        last_swapped_layer = (name, module)
      elif not isinstance(module, nn.BatchNorm2d) and last_swapped_layer != '': 
        name, module = last_swapped_layer # this is important ... if the current layer is not linerar or convolutional,                                       
      if isinstance(next_module, (nn.Linear, nn.Conv2d)) and not isinstance(module, nn.BatchNorm2d):
        swap_input_channels(next_module, module, mask)
      elif isinstance(next_module, (nn.BatchNorm2d)):
        swap_bn_modules(next_module, mask)
        _, next_module = module_list[i + 2]
        swap_input_channels(next_module, module, mask)

  return