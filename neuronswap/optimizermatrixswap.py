import torch
from torch import nn, optim
from .modulexplore import create_layer_indices_dict

def optimizer_swap_layer(layer: nn.Module, permutation_matrix: torch.Tensor, index: int, optimizer: optim.Optimizer):
  ''''''
  weight_shape = optimizer.state_dict()['state'][index]['momentum_buffer'].shape
  if len(weight_shape) != len(permutation_matrix.shape):
    reshaped_weight = torch.reshape(optimizer.state_dict()['state'][index]['momentum_buffer'], (weight_shape[0], -1))
    permutated_weight = torch.matmul(permutation_matrix, reshaped_weight)
    optimizer.state_dict()['state'][index]['momentum_buffer'] = torch.reshape(permutated_weight, weight_shape)
  else: 
    optimizer.state_dict()['state'][index]['momentum_buffer'] = torch.matmul(permutation_matrix, optimizer.state_dict()['state'][index]['momentum_buffer'])
  try:
    layer.get_parameter('bias')
  except:
    return
  else:
    optimizer.state_dict()['state'][index+1]['momentum_buffer'] = torch.matmul(optimizer.state_dict()['state'][index+1]['momentum_buffer'], permutation_matrix)

def optimizer_swap_input_channels(permutation_matrix: torch.Tensor, layer_index: int, previous_layer_index: int, optimizer: optim.Optimizer):
  ''''''
  matrix = permutation_matrix
  weights = optimizer.state_dict()['state'][layer_index]['momentum_buffer']
  previous_weights = optimizer.state_dict()['state'][previous_layer_index]['momentum_buffer']
  group_dimension = 1
  # this is in order to take into account the conv into linear interface where oftentimes
  # you have outputs from conv connecting to multiple input channels in linear
  if previous_weights.shape[0] != weights.shape[1]:
    group_dimension = weights.shape[1] // previous_weights.shape[0] # integer division
    if weights.shape[1] % previous_weights.shape[0] != 0:
      raise ValueError(f"Incompatible layers: number of neurons of the first layer does not match number of input channels on the second layer\n{weights.shape[1]}%{previous_weights.shape[0]}={weights.shape[1] % previous_weights.shape[0]}")
    # extending the permutation matrix to the dimension of the input layer
    matrix = torch.zeros(weights.shape[1], weights.shape[1])
    # obtaining the indices of the 1 inside the permutation matrix
    indexes = (permutation_matrix==1).nonzero(as_tuple=True)[1]
    # each of the elements of the permutation matrix is extended by the dimension of the group
    for j in range(len(indexes)):
      inde = indexes[j]
      for i in range(group_dimension):
        matrix[j * group_dimension + i, inde * group_dimension + i] = 1

    

  if len(optimizer.state_dict()['state'][layer_index]['momentum_buffer']) != len(matrix.shape):
    weight_dim = weights.shape
    reshaped_weight = weights.reshape((weight_dim[0], weight_dim[1], -1))
    permutated_weights = torch.empty((reshaped_weight.shape))
    for i in range(reshaped_weight.shape[-1]):
      permutated_weights[:,:,i] = torch.matmul(reshaped_weight[:,:,i], matrix)

    optimizer.state_dict()['state'][layer_index]['momentum_buffer'] = permutated_weights.reshape(weight_dim)

  else:
    optimizer.state_dict()['state'][layer_index]['momentum_buffer'] = torch.matmul(weights, matrix)

def optimizer_swap(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor], model: nn.Module, optimizer: optim.Optimizer,skip_connections: list[str] = []):
  ''''''
  layer_indices = create_layer_indices_dict(model)
  last_swapped_layer = ''
  for i in range(0,len(layers_list)):
    name, module = layers_list[i]
    if i != len(layers_list) - 1 and name not in skip_connections and name in permutations.keys():
      mask = permutations[name]
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        index = layer_indices[name]
        optimizer_swap_layer(module, mask, index, optimizer)
        next_name, next_module = layers_list[i + 1]
        last_swapped_layer = (name, module)
      elif not isinstance(module, nn.BatchNorm2d) and last_swapped_layer != '': 
        name, module = last_swapped_layer # this is important ... if the current layer is not linerar or convolutional,                                       
      if isinstance(next_module, (nn.Linear, nn.Conv2d)) and not isinstance(module, nn.BatchNorm2d):
        index = layer_indices[next_name]
        previous_index = layer_indices[name]
        optimizer_swap_input_channels(mask, index, previous_index, optimizer)
      elif isinstance(next_module, (nn.BatchNorm2d)):
        index = layer_indices[next_name]
        optimizer_swap_layer(next_module, mask, index, optimizer)
        next_name, next_module = layers_list[i + 2]
        index = layer_indices[next_name]
        previous_index = layer_indices[name]
        optimizer_swap_input_channels(mask, index, previous_index, optimizer)
  
