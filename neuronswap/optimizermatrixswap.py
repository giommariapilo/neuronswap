import torch
from torch import nn, optim
from .modulexplore import create_layer_indices_dict

def swap_layer(layer: nn.Module, permutation_matrix: torch.Tensor, index: int, optimizer: optim.Optimizer):
  ''''''
  for key in optimizer.state_dict()['state'][index].keys():   
    weight_shape = optimizer.state_dict()['state'][index][key].shape
    if len(weight_shape) != len(permutation_matrix.shape):
      reshaped_weight = torch.reshape(optimizer.state_dict()['state'][index][key], (weight_shape[0], -1))
      permuted_weight = torch.matmul(permutation_matrix, reshaped_weight)
      optimizer.state_dict()['state'][index][key] = torch.reshape(permuted_weight, weight_shape)
    else: 
      optimizer.state_dict()['state'][index][key] = torch.matmul(permutation_matrix, optimizer.state_dict()['state'][index][key])
  try:
    layer.get_parameter('bias')
  except:
    return
  else:
    for key in optimizer.state_dict()['state'][index].keys():
      optimizer.state_dict()['state'][index+1][key] = torch.matmul(permutation_matrix, optimizer.state_dict()['state'][index+1][key])

def swap_input_channels(permutation_matrix: torch.Tensor, layer_index: int, previous_layer_index: int, optimizer: optim.Optimizer):
  ''''''
  for key in optimizer.state_dict()['state'][layer_index].keys():
    matrix = permutation_matrix.transpose(0, 1) # important for asymmetrical permutation matrices
    weights = optimizer.state_dict()['state'][layer_index][key]
    previous_weights = optimizer.state_dict()['state'][previous_layer_index][key]
    group_dimension = 1
    # this is in order to take into account the conv into linear interface where oftentimes
    # you have outputs from conv connecting to multiple input channels in linear
    if previous_weights.shape[0] != weights.shape[1]:
      old_matrix = matrix
      group_dimension = weights.shape[1] // previous_weights.shape[0] # integer division
      if weights.shape[1] % previous_weights.shape[0] != 0:
        raise ValueError(f"Incompatible layers: number of neurons of the first layer does not match number of input channels on the second layer\n{weights.shape[1]}%{previous_weights.shape[0]}={weights.shape[1] % previous_weights.shape[0]}")
      # extending the permutation matrix to the dimension of the input layer
      matrix = torch.zeros(weights.shape[1], weights.shape[1])
      # obtaining the indices of the 1 inside the permutation matrix
      indexes = (old_matrix==1).nonzero(as_tuple=True)[1]
      # each of the elements of the permutation matrix is extended by the dimension of the group
      for j in range(len(indexes)):
        inde = indexes[j]
        for i in range(group_dimension):
          matrix[j * group_dimension + i, inde * group_dimension + i] = 1
    if len(optimizer.state_dict()['state'][layer_index][key]) != len(matrix.shape):
      weight_dim = weights.shape
      reshaped_weight = weights.reshape((weight_dim[0], weight_dim[1], -1))
      permuted_weights = torch.empty((reshaped_weight.shape))
      for i in range(reshaped_weight.shape[-1]):
        permuted_weights[:,:,i] = torch.matmul(reshaped_weight[:,:,i], matrix)

      optimizer.state_dict()['state'][layer_index][key] = permuted_weights.reshape(weight_dim)

    else:
      optimizer.state_dict()['state'][layer_index][key] = torch.matmul(weights, matrix)

def swap(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor], model: nn.Module, optimizer: optim.Optimizer,skip_connections: list[str] = []):
  ''''''
  layer_indices = create_layer_indices_dict(model)
  last_swapped_layer = ''
  for i in range(0,len(layers_list)):
    name, module = layers_list[i]
    if i != len(layers_list) - 1 and name not in skip_connections and name in permutations.keys():
      mask = permutations[name]
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        index = layer_indices[name]
        swap_layer(module, mask, index, optimizer)
        next_name, next_module = layers_list[i + 1]
        last_swapped_layer = (name, module)
      elif not isinstance(module, nn.BatchNorm2d) and last_swapped_layer != '': 
        name, module = last_swapped_layer # this is important ... if the current layer is not linerar or convolutional,                                       
      if isinstance(next_module, (nn.Linear, nn.Conv2d)) and not isinstance(module, nn.BatchNorm2d):
        index = layer_indices[next_name]
        previous_index = layer_indices[name]
        swap_input_channels(mask, index, previous_index, optimizer)
      elif isinstance(next_module, (nn.BatchNorm2d)):
        index = layer_indices[next_name]
        swap_layer(next_module, mask, index, optimizer)
        next_name, next_module = layers_list[i + 2]
        index = layer_indices[next_name]
        previous_index = layer_indices[name]
        swap_input_channels(mask, index, previous_index, optimizer)
  
def swap_inverted(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor | list[int]], model: nn.Module, optimizer: optim.Optimizer,skip_connections: list[str] = []):
  print('not implemented')
#   ''''''
#   layer_indices = create_layer_indices_dict(model)
#   last_swapped_layer = ''
#   for i in range(len(layers_list)-1,-1,-1):
#     name, module = layers_list[i]
#     if i != 0 and name not in skip_connections and name in permutations.keys():
#       mask = permutations[name]
#       if isinstance(module, (nn.Linear, nn.Conv2d)):
#         index = layer_indices[name]
#         previous_name, previous_module = layers_list[i - 1]
#         previous_index = layer_indices[previous_name]
#         swap_input_channels(mask, index, previous_index, optimizer)
#         last_swapped_layer = (name, module)
#       elif not isinstance(module, nn.BatchNorm2d) and last_swapped_layer != '': 
#         name, module = last_swapped_layer # this is important ... if the current layer is not linerar or convolutional,                                       
#       if isinstance(previous_module, (nn.Linear, nn.Conv2d)) and not isinstance(module, nn.BatchNorm2d):
#         swap_layer(previous_module, mask, previous_index, optimizer)
#       elif isinstance(previous_module, (nn.BatchNorm2d)):
#         previous_index = layer_indices[previous_name]
#         swap_layer(mask, previous_index, optimizer)
#         previous_name, previous_module = layers_list[i - 2]
#         previous_index = layer_indices[previous_name]
#         swap_layer(previous_module, mask, previous_index, optimizer)