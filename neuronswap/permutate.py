import torch
import numpy as np
from torch import nn

def permutate(module: nn.Module, permutation_matrix: torch.Tensor):
  '''
  This function takes as arguments a layer and the permutation matrix for that layer. 
  It then permutates the layer swapping the rows. It is implemented as a matrix multiplication.
  In order to accept permutations of convolutional layers (or in general n-dimensional arrays)
  the weight matrix is reshaped to a 2d matrix, multiplied and then restored to its original 
  dimensions.
  '''
  if len(module.weight.data.shape) != len(permutation_matrix.shape):
    weight_shape = module.weight.data.shape
    reshaped_weight = torch.reshape(module.weight.data, (weight_shape[0], -1))
    permutated_weight = torch.matmul(permutation_matrix, reshaped_weight)
    module.weight.data = torch.reshape(permutated_weight, weight_shape)
  else:
    module.weight.data = torch.matmul(permutation_matrix, module.weight.data)
  try:
    module.bias.data = torch.matmul(module.bias.data, permutation_matrix)
  except:
    pass

def permutate_inputs(module: nn.Module, previous_module: nn.Module, permutation_matrix: torch.Tensor):
  '''
  This function swaps the input channels of a layer by means of a matrix multiplication
  in this case the second dimension is swapped, to represent the input channels.
  '''
  matrix = permutation_matrix
  weights = module.weight.data
  group_dimension = 1
  # this is in order to take into account the conv into linear interface where oftentimes
  # you have outputs from conv connecting to multiple input channels in linear
  if previous_module.weight.data.shape[0] != weights.shape[1]:
    group_dimension = weights.shape[1] // previous_module.weight.data.shape[0] # integer division
    if weights.shape[1] % previous_module.weight.data.shape[0] != 0:
      raise ValueError(f"Incompatible layers: number of neurons of the first layer does not match number of input channels on the second layer\n{weights.shape[1]}%{previous_module.weight.data.shape[0]}={weights.shape[1] % previous_module.weight.data.shape[0]}")
 
    matrix = torch.zeros(weights.shape[1], weights.shape[1])

    indexes = (permutation_matrix==1).nonzero(as_tuple=True)[1]

    for j in range(len(indexes)):
      index = indexes[j]
      for i in range(group_dimension):
        matrix[j * group_dimension + i, index * group_dimension + i] = 1


  if len(module.weight.data.shape) != len(permutation_matrix.shape):
    weight_dim = weights.shape
    reshaped_weight = weights.reshape((weight_dim[0], weight_dim[1], -1))
    permutated_weights = torch.empty((reshaped_weight.shape))
    for i in range(reshaped_weight.shape[-1]):
      permutated_weights[:,:,i] = torch.matmul(reshaped_weight[:,:,i], permutation_matrix)

    module.weight.data = permutated_weights.reshape(weight_dim)

  else:
    module.weight.data = torch.matmul(weights, matrix)

    

def permutate_bn(module: nn.BatchNorm2d, permutation_matrix: torch.Tensor):
  '''
  This function swaps the parameters in a BN layer according to the permutation layer
  '''
  module.weight.data = torch.matmul(module.weight.data, permutation_matrix)
  module.bias.data = torch.matmul(module.bias.data, permutation_matrix)
  module.running_mean.data = torch.matmul(module.running_mean.data, permutation_matrix)
  module.running_var.data = torch.matmul(module.running_var.data, permutation_matrix)
  
def model_permutation(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor], skip_connections: list[str] = []):
  '''
  This function receives the list of layers of a model and a dictionary containing the 
  permutation matrix associated to each layer. It then calls permutate sequentially on each 
  layer, also permuting the input channels. The last layer won't be permutated as it will 
  change the output of the network. If it receives a list  of skip connections, those 
  layers are not permuted as the permutation of skip connection layers is not supported yet.
  '''
  last_swapped_layer = ''
  for i in range(0,len(layers_list)):
    name, module = layers_list[i]
    if i != len(layers_list) - 1 and name not in skip_connections and name in permutations.keys():
      mask = permutations[name]
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        permutate(module, mask)
        _, next_module = layers_list[i + 1]
        last_swapped_layer = (name, module)
      elif not isinstance(module, nn.BatchNorm2d) and last_swapped_layer != '': 
        name, module = last_swapped_layer # this is important ... if the current layer is not linerar or convolutional,                                       
      if isinstance(next_module, (nn.Linear, nn.Conv2d)) and not isinstance(module, nn.BatchNorm2d):
        permutate_inputs(next_module, module, mask)
      elif isinstance(next_module, (nn.BatchNorm2d)):
        permutate_bn(next_module, mask)
        _, next_module = layers_list[i + 2]
        permutate_inputs(next_module, module, mask)