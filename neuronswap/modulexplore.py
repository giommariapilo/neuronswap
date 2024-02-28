from torch import fx, nn
from torch.fx import node

def get_layers_list(graph: fx.Graph, model: nn.Module):
  '''Returns a list of the layers of a module in the order 
  they appear in the graph obtained using torch.fx'''
  layers_dict = {name: layer for name, layer in model.named_modules()}
  layers_list = [('.'.join(node.name.split('_')), layers_dict['.'.join(node.name.split('_'))]) for node in graph.nodes if '.'.join(node.name.split('_')) in layers_dict.keys() and isinstance(layers_dict['.'.join(node.name.split('_'))], (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
  return layers_list 

def recursive_call(parent, parents, layers):
  for subparent in parent: # if the parent is not an instance of a linear or convolutional layer, keep searching
    if isinstance(subparent, tuple):
      recursive_call(subparent, parents, layers)
    elif isinstance(subparent, node.Node):
      # print(subparent)
      parent_search(parents, subparent, layers)

def parent_search(parents: list, node: node.Node, layers: list[tuple]):
  '''Recursive function to find the nearest linear or 
  convolutional parent node for a given node. Takes as 
  input an empty list which is populated by the function'''
  layers_dict = dict(layers)
  name = '.'.join(node.name.split('_')) # node names and layer names are the same but with different separators '_' for nodes and '.' for layers
  if name in layers_dict.keys() and isinstance(layers_dict[name], (nn.Linear, nn.Conv2d)):
    parents.append(name)
  else: 
    # for parent in node.args: # if the parent is not an instance of a linear or convolutional layer, keep searching
    #   parent_search(parents, parent, layers)
    recursive_call(node.args, parents, layers)
  return

def get_skipped_layers(graph: fx.Graph, layers: list[tuple]):
  '''This function takes as input a graph of a model and a list of its layers
  and returns a list of the layers involved in a skip connection that must be avoided 
  when swapping neurons'''
  residual_connections = []
  for node in graph.nodes:
    try:
      parents = [parent.name for parent in node.args]
    except:
      parents = []
    if len(parents) >= 2: residual_connections.append(node)

  skipped_layers = []
  for node in residual_connections:
    parent_search(skipped_layers, node, layers)

  skipped_layers = list(set(skipped_layers))
  return skipped_layers

def create_layer_indices_dict(model: nn.Module) -> dict[str, int]:
  ''''''
  layer_indices = {}
  index = 0
  for name, layer in model.named_children():
    if isinstance(layer, (nn.Linear, nn.BatchNorm2d, nn.Conv2d)):
      layer_indices[name] = index
      index += 1
      try:
        layer.get_parameter('bias')
      except:
        continue 
      else:
        index += 1
  return layer_indices