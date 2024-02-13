# Neuronswap

This library is used to permutate neurons inside layers of a model. It is intended to be uses in the NEq algorithm to move neurons at equilibrium to the top of the layer so that they can be excluded from gradient computations, but has also a more general method that accepts a permutation matrix to transform the layers in a model, maintaning the same output.

## Installation

Neuronswap has not been uploaded yet to PyPA so to use it, download the repository and include it in your code using `sys.path.append()`:

```python
import sys
sys.path.append('/path/to/folder/containing/library')
import neuronswap.nswap as ns
import neuronswap.modulexplore as modx
```

****

## Usage

### Main function
There are three main parts to the library.
- The first one, `get_layers_list()`, obtains an list of the layers ordered hierarchically from a graph of the model generated using `torch.fx.symbolic_trace(your_model).graph`. An optional list of skip connections can be obtained using `get_skip_connections()`
- The second one is `permutate()` which is the main function of the library. It takes as inputs the list of layer, and optionally the list of skip connections obtained previously, together with a dictionary containing for each layer name, the permutation that needs to be performed on the layer. This is in the form of a permutation matrix or a list of the indices of the neurons to be moved to the top of the matrix. Depending on which is passed, a different method is used to perform the transformation. The input channels of the following layer are swapped accordingly. In case a list of skip connections is passed, these layers are ignored as the permutation in these case is not supported yet. The function does not return any value as it changes the layers in the model directly.
- In case this method is used during training and an optimizer with momentum is used it is necessary to swap the parameters saved inside the state dictionary of the optimizer as well. This can be done using `permutate_optimizer()`. It takes as inputs the optimizer, the list of layer, and optionally the list of skip connections obtained previously, together with a dictionary containing for each layer name, the permutation that needs to be performed on the layer. Depending on which is passed, a different method is used to perform the transformation. The parameters corresponding to the input channels of the following layer are also swapped accordingly. In case a list of skip connections is passed, these layers are ignored as the permutation in these case is not supported yet. The function does not return any value as it changes the state dictionary in the optimizer directly.

#### Arguments

The expected arguments for `permutate` are:

- `layers_list (list[torch.nn.Module])`: List of the layers in hierarchical order of the model to which the swap is applyed. Obtained using `get_layers_list()`.
- `permutations (dict[str, torch.Tensor | list[int]])`: Dictionary where the keys are the layer names and the values are iterables containing the indexes of the neurons to be moved to the top of the layer or permutation matrices. It is important to chose just one of the two and not to mix them together.
- `skip_connections (list[str])`: List of the names of the layers which are connected to a skip connection layer. These will be prevented from switching. Default is an empty list.

#### Minimal working example

Tested on python 3.10.12

```python
import sys
sys.path.append('/home/gpilo/neuronswap')
import torch
import neuronswap as ns

class FCnet(torch.nn.Module):
  def __init__(self):
    super(FCnet, self).__init__()
    self.fc1 = torch.nn.Linear(5, 10)
    self.fc2 = torch.nn.Linear(10, 12)
    self.fc3 = torch.nn.Linear(12, 4)

  def forward(self, x):
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = FCnet()
eq_indexes = {"fc1": torch.tensor([4, 9], device='cpu'),
              "fc2": torch.tensor([3, 6], device='cpu'),
              "fc3": torch.tensor([1, 2], device='cpu')}

graph = torch.fx.symbolic_trace(model).graph
layers_list = ns.get_layers_list(graph, model)

input = torch.rand(5)

output_before = model(input)

ns.permutate(layers_list, eq_indexes)

output_after = model(input)

print(torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8))

```

#### Example with skip connections

```python
import sys
sys.path.append('/path/to/folder/containing/library')
import torch
from torchvision import models
import neuronswap as ns

model = models.resnet18()

eq_indexes = {'layer1.0.conv1': [2,3], 
              'layer1.1.conv1': [2,3], 
              'layer2.0.conv1': [2,3], 
              'layer2.1.conv1': [2,3], 
              'layer3.0.conv1': [2,3], 
              'layer3.1.conv1': [2,3], 
              'layer4.0.conv1': [2,3], 
              'layer4.1.conv1': [2,3], }

graph = torch.fx.symbolic_trace(model).graph
layers_list = ns.get_layers_list(graph, model)
skip_connections = ns.get_skipped_layers(graph, layers_list)

input = torch.rand([1,3,244,244])

model.train(False)

output_before = model(input)

ns.permutate(layers_list, eq_indexes, skip_connections)

output_after = model(input)

print(torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-6))

```

#### Permutation matrices example

Tested on python 3.10.12

```python
import sys
sys.path.append('/path/to/folder/containing/library')
import torch
import neuronswap as ns

class FCnet(torch.nn.Module):
  def __init__(self):
    super(FCnet, self).__init__()
    self.fc1 = torch.nn.Linear(5, 10)
    self.fc2 = torch.nn.Linear(10, 12)
    self.fc3 = torch.nn.Linear(12, 4)

  def forward(self, x):
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = FCnet()

permutation_matrix = {"fc1": torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                                          dtype = torch.float32),
                      "fc2": torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                                          dtype = torch.float32),
                      "fc3": torch.tensor([[0, 0, 1, 0],
                                            [0, 1, 0, 0],
                                            [1, 0, 0, 0],
                                            [0, 0, 0, 1]],
                                          dtype = torch.float32)} # this won't have any effect but it is here to verify it is ignored

graph = torch.fx.symbolic_trace(model).graph
layers_list = ns.get_layers_list(graph, model)

input = torch.rand(5)

output_before = model(input)

ns.permutate(layers_list, permutation_matrix)

output_after = model(input)

print(torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8))
```
