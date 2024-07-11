import sys
sys.path.append('..')
import neuronswap.matrixswap as mswap
import neuronswap.modulexplore as modx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torchvision.models import resnet18

class FCnet(nn.Module):
  def __init__(self):
    super(FCnet, self).__init__()
    self.fc1 = nn.Linear(5, 10)
    self.fc2 = nn.Linear(10, 12)
    self.fc3 = nn.Linear(12, 4)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
class ConvNetwork(nn.Module):
  def __init__(self):
    super(ConvNetwork, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 4 * 4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class ConvBN(nn.Module):
  def __init__(self):
    super(ConvBN, self).__init__()
    self.conv1 = nn.Conv2d(4, 2, 3)
    self.bn1 = nn.BatchNorm2d(2)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(2, 3, 4)
    self.fc1 = nn.Linear(3 * 4 * 4, 8)
    self.fc3 = nn.Linear(8, 10)

  def forward(self, x):
    x = F.relu(self.pool(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.conv2(x)))
    #print(x.shape)
    x = x.view(-1, 3 * 4 * 4)
    x = F.relu(self.fc1(x))
    x = self.fc3(x)
    return x     

def test_permutation_fc_layer():
  module = nn.Linear(5, 10)
  permutation_matrix = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                                    dtype = torch.float32)
  
  module.weight.data = torch.tensor([[ 0.3755,  0.4431, -0.0478,  0.3921, -0.3728],
                                     [ 0.1616, -0.1324,  0.2722, -0.4187, -0.0524],
                                     [-0.2651, -0.3016,  0.2752,  0.1855, -0.2994],
                                     [ 0.1573,  0.0808,  0.2772,  0.2052, -0.0638],
                                     [ 0.0764,  0.2309,  0.1859, -0.2075,  0.3967],
                                     [ 0.1107,  0.3426,  0.1447,  0.1134, -0.4268],
                                     [ 0.1500,  0.3672,  0.3327,  0.4315, -0.4033],
                                     [ 0.0532,  0.3555,  0.0187,  0.2583,  0.0687],
                                     [-0.4306, -0.1518, -0.3142, -0.1896,  0.0766],
                                     [ 0.0536, -0.2405, -0.3818, -0.1850, -0.0953]])
  
  expected_weights = torch.tensor([[-0.2651, -0.3016,  0.2752,  0.1855, -0.2994],
                                   [ 0.1616, -0.1324,  0.2722, -0.4187, -0.0524],
                                   [ 0.3755,  0.4431, -0.0478,  0.3921, -0.3728],
                                   [ 0.1573,  0.0808,  0.2772,  0.2052, -0.0638],
                                   [ 0.0764,  0.2309,  0.1859, -0.2075,  0.3967],
                                   [ 0.1107,  0.3426,  0.1447,  0.1134, -0.4268],
                                   [ 0.1500,  0.3672,  0.3327,  0.4315, -0.4033],
                                   [ 0.0532,  0.3555,  0.0187,  0.2583,  0.0687],
                                   [-0.4306, -0.1518, -0.3142, -0.1896,  0.0766],
                                   [ 0.0536, -0.2405, -0.3818, -0.1850, -0.0953]])

  mswap.swap_layer(module, permutation_matrix)

  assert torch.equal(expected_weights, module.weight.data), 'Permutation was not performed correctly'

def test_permutation_conv_layer():
  module = nn.Conv2d(3, 3, 2)
  permutation_matrix = torch.tensor([[0, 0, 1],
                                     [0, 1, 0],
                                     [1, 0, 0]],
                                    dtype = torch.float32)

  module.weight.data = torch.tensor([[[[ 0.0287,  0.1400],
                                       [-0.0062,  0.2209]],
                                      [[-0.1105, -0.1511],
                                       [ 0.2192,  0.2149]],
                                      [[-0.0857,  0.1002],
                                       [-0.1481,  0.2322]]],

                                     [[[ 0.0184,  0.0994],
                                       [ 0.1431, -0.0547]],
                                      [[ 0.0721, -0.0202],
                                       [ 0.0227, -0.1591]],
                                      [[ 0.2512, -0.0538],
                                       [-0.0302,  0.1101]]],

                                     [[[-0.1709, -0.2350],
                                       [ 0.2849, -0.0315]],
                                      [[-0.2852,  0.2399],
                                       [ 0.1293, -0.2778]],
                                      [[ 0.0035, -0.1421],
                                       [-0.0445,  0.2405]]]])

  expected_weights = torch.tensor([[[[-0.1709, -0.2350],
                                     [ 0.2849, -0.0315]],
                                    [[-0.2852,  0.2399],
                                     [ 0.1293, -0.2778]],
                                    [[ 0.0035, -0.1421],
                                     [-0.0445,  0.2405]]],

                                   [[[ 0.0184,  0.0994],
                                     [ 0.1431, -0.0547]],
                                    [[ 0.0721, -0.0202],
                                     [ 0.0227, -0.1591]],
                                    [[ 0.2512, -0.0538],
                                     [-0.0302,  0.1101]]],

                                   [[[ 0.0287,  0.1400],
                                     [-0.0062,  0.2209]],
                                    [[-0.1105, -0.1511],
                                     [ 0.2192,  0.2149]],
                                    [[-0.0857,  0.1002],
                                     [-0.1481,  0.2322]]]])

  mswap.swap_layer(module, permutation_matrix)

  assert torch.equal(expected_weights, module.weight.data), 'Permutation was not performed correctly'

def test_permutate_inputs_fc():
  prev_module = nn.Linear(4, 5)
  module = nn.Linear(5, 10)
  permutation_matrix = torch.tensor([[0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1]],
                                    dtype = torch.float32)
  
  module.weight.data = torch.tensor([[ 0.3755,  0.4431, -0.0478,  0.3921, -0.3728],
                                     [ 0.1616, -0.1324,  0.2722, -0.4187, -0.0524],
                                     [-0.2651, -0.3016,  0.2752,  0.1855, -0.2994],
                                     [ 0.1573,  0.0808,  0.2772,  0.2052, -0.0638],
                                     [ 0.0764,  0.2309,  0.1859, -0.2075,  0.3967],
                                     [ 0.1107,  0.3426,  0.1447,  0.1134, -0.4268],
                                     [ 0.1500,  0.3672,  0.3327,  0.4315, -0.4033],
                                     [ 0.0532,  0.3555,  0.0187,  0.2583,  0.0687],
                                     [-0.4306, -0.1518, -0.3142, -0.1896,  0.0766],
                                     [ 0.0536, -0.2405, -0.3818, -0.1850, -0.0953]])
  
  expected_weights = torch.tensor([[-0.0478,  0.4431,  0.3755,  0.3921, -0.3728],
                                   [ 0.2722, -0.1324,  0.1616, -0.4187, -0.0524],
                                   [ 0.2752, -0.3016, -0.2651,  0.1855, -0.2994],
                                   [ 0.2772,  0.0808,  0.1573,  0.2052, -0.0638],
                                   [ 0.1859,  0.2309,  0.0764, -0.2075,  0.3967],
                                   [ 0.1447,  0.3426,  0.1107,  0.1134, -0.4268],
                                   [ 0.3327,  0.3672,  0.1500,  0.4315, -0.4033],
                                   [ 0.0187,  0.3555,  0.0532,  0.2583,  0.0687],
                                   [-0.3142, -0.1518, -0.4306, -0.1896,  0.0766],
                                   [-0.3818, -0.2405,  0.0536, -0.1850, -0.0953]])

  mswap.swap_input_channels(module, prev_module, permutation_matrix)

  assert torch.equal(expected_weights, module.weight.data), 'Permutation was not performed correctly'


def test_permutate_inputs_conv():
  prev_module = nn.Linear(4, 3)
  module = nn.Conv2d(3, 3, 2)
  permutation_matrix = torch.tensor([[0, 0, 1],
                                     [0, 1, 0],
                                     [1, 0, 0]],
                                    dtype = torch.float32)

  module.weight.data = torch.tensor([[[[ 0.0287,  0.1400],
                                       [-0.0062,  0.2209]],
                                      [[-0.1105, -0.1511],
                                       [ 0.2192,  0.2149]],
                                      [[-0.0857,  0.1002],
                                       [-0.1481,  0.2322]]],

                                     [[[ 0.0184,  0.0994],
                                       [ 0.1431, -0.0547]],
                                      [[ 0.0721, -0.0202],
                                       [ 0.0227, -0.1591]],
                                      [[ 0.2512, -0.0538],
                                       [-0.0302,  0.1101]]],

                                     [[[-0.1709, -0.2350],
                                       [ 0.2849, -0.0315]],
                                      [[-0.2852,  0.2399],
                                       [ 0.1293, -0.2778]],
                                      [[ 0.0035, -0.1421],
                                       [-0.0445,  0.2405]]]])

  expected_weights = torch.tensor([[[[-0.0857,  0.1002],
                                     [-0.1481,  0.2322]],
                                    [[-0.1105, -0.1511],
                                     [ 0.2192,  0.2149]],
                                    [[ 0.0287,  0.1400],
                                     [-0.0062,  0.2209]]],

                                   [[[ 0.2512, -0.0538],
                                     [-0.0302,  0.1101]],
                                    [[ 0.0721, -0.0202],
                                     [ 0.0227, -0.1591]],
                                    [[ 0.0184,  0.0994],
                                     [ 0.1431, -0.0547]]],

                                   [[[ 0.0035, -0.1421],
                                     [-0.0445,  0.2405]],
                                    [[-0.2852,  0.2399],
                                     [ 0.1293, -0.2778]],
                                    [[-0.1709, -0.2350],
                                     [ 0.2849, -0.0315]]]])

  mswap.swap_input_channels(module, prev_module, permutation_matrix)

  assert torch.equal(expected_weights, module.weight.data), 'Permutation was not performed correctly'

def test_permutate_bn_layer():
  module = nn.BatchNorm2d(5)

  permutation_matrix = torch.tensor([[0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1]],
                                    dtype = torch.float32)

  module.weight.data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
  module.bias.data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
  module.running_var.data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
  module.running_mean.data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

  expected = torch.tensor([3, 2, 1, 4, 5], dtype=torch.float32)
  mswap.swap_bn_layer(module, permutation_matrix)

  assert torch.equal(module.weight.data, expected)
  assert torch.equal(module.bias.data, expected)
  assert torch.equal(module.running_mean.data, expected)
  assert torch.equal(module.running_var.data, expected)
  
def test_permutate_inputs_conv_fc():
  conv = nn.Conv2d(3,3,2)
  lin = nn.Linear(12, 3)

  permutation_matrix = torch.tensor([[0, 0, 1],
                                     [0, 1, 0],
                                     [1, 0, 0]],
                                    dtype = torch.float32)

  lin.weight.data = torch.tensor([[-0.1278,  0.0898, -0.2036,  0.0865,  0.2089,  0.0134,  0.2746, -0.2613,
                                    0.1419,  0.1712, -0.0689, -0.0997],
                                  [-0.1243, -0.2744, -0.0487, -0.1254, -0.1331, -0.0780, -0.1378, -0.2418,
                                  -0.1816,  0.0072,  0.0932,  0.0596],
                                  [ 0.2785, -0.1574,  0.2641, -0.1644,  0.1377, -0.1832, -0.0208, -0.0665,
                                    0.1544, -0.0328, -0.0416, -0.0471]])
  expected_weights = torch.tensor([[ 0.1419,  0.1712, -0.0689, -0.0997,  0.2089,  0.0134,  0.2746, -0.2613,
                                    -0.1278,  0.0898, -0.2036,  0.0865],
                                   [-0.1816,  0.0072,  0.0932,  0.0596, -0.1331, -0.0780, -0.1378, -0.2418,
                                    -0.1243, -0.2744, -0.0487, -0.1254],
                                   [ 0.1544, -0.0328, -0.0416, -0.0471,  0.1377, -0.1832, -0.0208, -0.0665,
                                     0.2785, -0.1574,  0.2641, -0.1644]])

  mswap.swap_input_channels(lin, conv, permutation_matrix)

  assert torch.equal(lin.weight.data, expected_weights)

def test_permutate_fc_net():
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

  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)

  input = torch.rand(5)

  output_before = model(input)

  mswap.swap(layers_list, permutation_matrix)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8)


def test_permutate_conv_net():
  model = ConvNetwork()

  permutation_matrix = {}

  # i create the permutation matrix automatically, the same for each layer
  for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
      matrix = torch.zeros(layer.weight.data.shape[0], layer.weight.data.shape[0], dtype = torch.float32)
      for i in range(layer.weight.data.shape[0]):
        if 0 == i:
          matrix[i, 2] = 1
        elif 2 == i:
          matrix[i, 0] = 1
        else:
          matrix[i, i] = 1
      if matrix.shape == torch.tensor([2, 2]):
        matrix = torch.tensor([[0, 1],
                               [1, 0]],
                              dtype = torch.float32)
      permutation_matrix[name] = matrix

  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)

  input = torch.rand([1, 28, 28])

  output_before = model(input)

  mswap.swap(layers_list, permutation_matrix)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8)

def test_permutate_convBN_net():
  model = ConvBN()

  permutation_matrix = {"conv1": torch.tensor([[0, 1],
                                               [1, 0]],
                                              dtype = torch.float32),
                        "conv2": torch.tensor([[0, 0, 1],
                                               [0, 1, 0],
                                               [1, 0, 0]],
                                              dtype = torch.float32),    
                        "fc1": torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 1]],
                                            dtype = torch.float32),
                        "fc3": torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                                            dtype = torch.float32)} # this won't have any effect but it is here to verify it is ignored

  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)

  input = torch.rand([1, 4, 24, 24])

  output_before = model(input)

  mswap.swap(layers_list, permutation_matrix)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8)


def test_permutate_resnet18():
  model = resnet18()

  permutation_matrix = {}
  # i create the permutation matrix automatically, the same for each layer
  for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
      matrix = torch.zeros(layer.weight.data.shape[0], layer.weight.data.shape[0], dtype = torch.float32)
      for i in range(layer.weight.data.shape[0]):
        if 0 == i:
          matrix[i, 2] = 1
        elif 2 == i:
          matrix[i, 0] = 1
        else:
          matrix[i, i] = 1
      if matrix.shape == torch.tensor([2, 2]):
        matrix = torch.tensor([[0, 1],
                               [1, 0]],
                              dtype = torch.float32)
      permutation_matrix[name] = matrix
  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)
  skip_connections = modx.get_skipped_layers(graph, layers_list)
  
  input = torch.rand([1,3,244,244])

  output_before = model(input)

  mswap.swap(layers_list, permutation_matrix, skip_connections)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-5)

# def test_permutate_resnet18_inverted():
#   model = resnet18()

#   permutation_matrix = {}
#   # i create the permutation matrix automatically, the same for each layer
#   for name, layer in model.named_modules():
#     if isinstance(layer, (nn.Conv2d, nn.Linear)):
#       matrix = torch.zeros(layer.weight.data.shape[0], layer.weight.data.shape[0], dtype = torch.float32)
#       for i in range(layer.weight.data.shape[0]):
#         if 0 == i:
#           matrix[i, 2] = 1
#         elif 2 == i:
#           matrix[i, 0] = 1
#         else:
#           matrix[i, i] = 1
#       if matrix.shape == torch.tensor([2, 2]):
#         matrix = torch.tensor([[0, 1],
#                                [1, 0]],
#                               dtype = torch.float32)
#       permutation_matrix[name] = matrix
#   graph = fx.symbolic_trace(model).graph
#   layers_list = modx.get_layers_list(graph, model)
#   skip_connections = modx.get_skipped_layers_inverted(graph, layers_list)
  
#   input = torch.rand([1,3,244,244])

#   output_before = model(input)

#   mswap.swap_inverted(layers_list, permutation_matrix, skip_connections)

#   output_after = model(input)

#   assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-5)