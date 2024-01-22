# Neuronswap

Simple library to swap neuron positions inside the layers of a model

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

The use of the library requires a graph of the model generated using `torch.fx.symbolic_trace(your_model).graph`. There are two submodules, `modulexplore` and `nswap`, which are used to analyze the proprieties of the model, and effectively swapping its neurons, respectively. The main function in `nswap` is `swap`, which does not return anything as it modifies the model itself. To use the function, a list of the layers of the model needs to be provided. This is obtained using the function `get_layers_list` in `modulexplore`. The function returns a list of the layers in the order they appear in the hierarchy.

#### Arguments

The expected arguments for `swap` are:

- `module_list (list[torch.nn.Module])`: List of the layers of the model to which the swap is applyed.
- `equilibrium_mask (dict[str, torch.Tensor])`: Dictionary where the keys are the layer names and the values are iterables containing the indexes of the neurons to be swapped.
- `skip_connections (list[str])`: List of the names of the layers which are connected to a skip connection layer. These will be prevented from switching. Default is an empty list.

#### Minimal working example

Tested on python 3.10.12

```python
import sys
sys.path.append('/path/to/folder/containing/library')
import torch
import neuronswap.nswap as ns
import neuronswap.modulexplore as modx

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
layers_list = modx.get_layers_list(graph, model)

input = torch.rand(5)

output_before = model(input)

ns.swap(layers_list, eq_indexes)

output_after = model(input)

print(torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8))

```

#### Example with skip connections

```python
import sys
sys.path.append('/path/to/folder/containing/library')
import torch
from torchvision import models
import neuronswap.nswap as ns
import neuronswap.modulexplore as modx

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
layers_list = modx.get_layers_list(graph, model)
skip_connections = modx.get_skipped_layers(graph, layers_list)

input = torch.rand([1,3,244,244])

model.train(False)

output_before = model(input)

ns.swap(layers_list, eq_indexes, skip_connections)

output_after = model(input)

print(torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-5))

```