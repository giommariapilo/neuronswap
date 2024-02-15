import torch
from torch import nn

def swap_layer(layer: nn.Module, permutation_matrix: torch.Tensor):
  '''
  This function takes as arguments a layer and the permutation matrix for that layer. 
  It then permutates the layer swapping the rows. It is implemented as a matrix multiplication.
  In order to accept permutations of convolutional layers (or in general n-dimensional arrays)
  the weight matrix is reshaped to a 2d matrix, multiplied and then restored to its original 
  dimensions.
  '''
  if len(layer.weight.data.shape) != len(permutation_matrix.shape):
    weight_shape = layer.weight.data.shape
    reshaped_weight = torch.reshape(layer.weight.data, (weight_shape[0], -1))
    permutated_weight = torch.matmul(permutation_matrix, reshaped_weight)
    layer.weight.data = torch.reshape(permutated_weight, weight_shape)
  else:
    layer.weight.data = torch.matmul(permutation_matrix, layer.weight.data)
  try:
    layer.bias.data = torch.matmul(layer.bias.data, permutation_matrix)
  except:
    pass

def swap_input_channels(layer: nn.Module, previous_layer: nn.Module, permutation_matrix: torch.Tensor):
  '''
  This function swaps the input channels of a layer by means of a matrix multiplication
  in this case the second dimension is swapped, to represent the input channels.
  '''
  matrix = permutation_matrix.transpose(0, 1) # important for asymmetrical permutation matrices
  weights = layer.weight.data
  group_dimension = 1
  # this is in order to take into account the conv into linear interface where oftentimes
  # you have outputs from conv connecting to multiple input channels in linear
  if previous_layer.weight.data.shape[0] != weights.shape[1]:
    group_dimension = weights.shape[1] // previous_layer.weight.data.shape[0] # integer division
    if weights.shape[1] % previous_layer.weight.data.shape[0] != 0:
      raise ValueError(f"Incompatible layers: number of neurons of the first layer does not match number of input channels on the second layer\n{weights.shape[1]}%{previous_layer.weight.data.shape[0]}={weights.shape[1] % previous_layer.weight.data.shape[0]}")
 
    matrix = torch.zeros(weights.shape[1], weights.shape[1])

    indexes = (permutation_matrix==1).nonzero(as_tuple=True)[1]

    for j in range(len(indexes)):
      index = indexes[j]
      for i in range(group_dimension):
        matrix[j * group_dimension + i, index * group_dimension + i] = 1


  if len(layer.weight.data.shape) != len(permutation_matrix.shape):
    weight_dim = weights.shape
    reshaped_weight = weights.reshape((weight_dim[0], weight_dim[1], -1))
    permutated_weights = torch.empty((reshaped_weight.shape))
    for i in range(reshaped_weight.shape[-1]):
      permutated_weights[:,:,i] = torch.matmul(reshaped_weight[:,:,i], permutation_matrix)

    layer.weight.data = permutated_weights.reshape(weight_dim)

  else:
    layer.weight.data = torch.matmul(weights, matrix)

    

def swap_bn_layer(layer: nn.BatchNorm2d, permutation_matrix: torch.Tensor):
  '''
  This function swaps the parameters in a BN layer according to the permutation layer
  '''
  layer.weight.data = torch.matmul(layer.weight.data, permutation_matrix)
  layer.bias.data = torch.matmul(layer.bias.data, permutation_matrix)
  layer.running_mean.data = torch.matmul(layer.running_mean.data, permutation_matrix)
  layer.running_var.data = torch.matmul(layer.running_var.data, permutation_matrix)
  
def swap(layers_list: list[nn.Module], permutations: dict[str, torch.Tensor], skip_connections: list[str] = []):
  '''
  This function takes as inputs the list of layers of a model, a dictionary containing the 
  permutation matrix associated to each layer, and an optional list of skip connections. 
  It then sequentially permutates each layer, also permutating the input channels. The 
  last layer won't be permutated as it will change the output of the network. If it 
  receives a list of skip connections, those layers are not permutated as permutation 
  of skip connection layers is not supported yet.
  '''
  last_swapped_layer = ''
  for i in range(0,len(layers_list)):
    name, module = layers_list[i]
    if i != len(layers_list) - 1 and name not in skip_connections and name in permutations.keys():
      mask = permutations[name]
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