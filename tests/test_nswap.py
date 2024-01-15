import sys
sys.path.append('..')
import neuronswap.nswap as ns
import neuronswap.modulexplore as modx
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
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
  
def test_swap_FC_layer():
  linear_layer = nn.Linear(3, 4)
  # get a known value for weight
  linear_layer.weight.data = torch.Tensor([[ 0.15,  0.23, -0.67], 
                                           [-0.98,  0.56,  0.34], 
                                           [-0.05,  0.01,  0.07], 
                                           [ 0.15, -0.56,  0.18]])

  linear_layer.bias.data = torch.Tensor([0.435, -0.543, 0.242, 0.673])

  ns.swap_lin_conv_layers(linear_layer, [1, 2])
  # compare with expected result
  expected_weights = torch.Tensor([[-0.98,  0.56,  0.34], 
                                   [-0.05,  0.01,  0.07], 
                                   [ 0.15,  0.23, -0.67], 
                                   [ 0.15, -0.56,  0.18]])
  
  expected_bias = torch.Tensor([-0.543, 0.242, 0.435, 0.673])

  assert torch.equal(linear_layer.weight.data, expected_weights), f'Incorrect weight swap: expecting {expected_weights}\ngot instead {linear_layer.weight.data}'
  assert torch.equal(linear_layer.bias.data, expected_bias), f'Incorrect bias swap: expecting {expected_bias}\ngot instead {linear_layer.bias.data}'

def test_swap_conv_layer():
  conv_layer = nn.Conv2d(3, 4, 2)
  # get a known value for weight
  conv_layer.weight.data = torch.Tensor([[[[-0.1508,  0.2335],
                                           [-0.0520,  0.2136]],
                                          [[-0.2786,  0.2326],
                                           [-0.1977, -0.0169]],
                                          [[-0.0533,  0.1771],
                                           [-0.0153, -0.1908]]],

                                         [[[-0.1698,  0.2072],
                                           [-0.1241, -0.0714]],
                                          [[ 0.1051, -0.0489],
                                           [ 0.1667, -0.1183]],
                                          [[-0.1565, -0.1056],
                                           [-0.2035,  0.2017]]],

                                         [[[-0.0479, -0.0728],
                                           [-0.2783,  0.1412]],
                                          [[ 0.0313,  0.0433],
                                           [ 0.2048,  0.1866]],
                                          [[ 0.0477,  0.2821],
                                           [-0.2720,  0.1080]]],

                                         [[[ 0.0658, -0.0500],
                                           [ 0.1675, -0.2215]],
                                          [[ 0.2166, -0.0258],
                                           [ 0.1032, -0.1255]],
                                          [[-0.2366,  0.1706],
                                           [ 0.1604,  0.1617]]]])

  conv_layer.bias.data = torch.Tensor([-0.0591, 0.2348, 0.2842, 0.1032])

  ns.swap_lin_conv_layers(conv_layer, [1, 2])
  # compare with expected result
  expected_weights = torch.Tensor([[[[-0.1698,  0.2072],
                                     [-0.1241, -0.0714]],
                                    [[ 0.1051, -0.0489],
                                     [ 0.1667, -0.1183]],
                                    [[-0.1565, -0.1056],
                                     [-0.2035,  0.2017]]],

                                   [[[-0.0479, -0.0728],
                                     [-0.2783,  0.1412]],
                                    [[ 0.0313,  0.0433],
                                     [ 0.2048,  0.1866]],
                                    [[ 0.0477,  0.2821],
                                     [-0.2720,  0.1080]]],

                                   [[[-0.1508,  0.2335],
                                     [-0.0520,  0.2136]],
                                    [[-0.2786,  0.2326],
                                     [-0.1977, -0.0169]],
                                    [[-0.0533,  0.1771],
                                     [-0.0153, -0.1908]]],

                                   [[[ 0.0658, -0.0500],
                                     [ 0.1675, -0.2215]],
                                    [[ 0.2166, -0.0258],
                                     [ 0.1032, -0.1255]],
                                    [[-0.2366,  0.1706],
                                     [ 0.1604,  0.1617]]]])
  
  expected_bias = torch.Tensor([0.2348, 0.2842, -0.0591, 0.1032])

  assert torch.equal(conv_layer.weight.data, expected_weights), f'Incorrect weight swap: expecting {expected_weights}\ngot instead {conv_layer.weight.data}'
  assert torch.equal(conv_layer.bias.data, expected_bias), f'Incorrect bias swap: expecting {expected_bias}\ngot instead {conv_layer.bias.data}'


def test_swap_input_ch():
  linear_layer = nn.Linear(4, 3)
  conv_layer = nn.Conv2d(3, 4, 2)
  linear_layer2 = nn.Linear(4 * 4 * 4, 3)
  
  conv_layer.weight.data = torch.Tensor([[[[-0.1174, -0.1749],
                                           [ 0.1261,  0.1615]],
                                          [[-0.1227,  0.1047],
                                           [ 0.0219, -0.0687]],
                                          [[ 0.0389, -0.2334],
                                           [-0.0555,  0.0880]]],

                                         [[[ 0.2188,  0.0427],
                                           [ 0.1169, -0.0729]],
                                          [[-0.1758,  0.2083],
                                           [ 0.1382,  0.2704]],
                                          [[-0.1103, -0.0255],
                                           [ 0.1005, -0.1327]]],

                                         [[[ 0.0200,  0.2130],
                                           [-0.0108,  0.1380]],
                                          [[ 0.0066,  0.2665],
                                           [-0.1093, -0.1395]],
                                          [[-0.1100, -0.0686],
                                           [-0.1565,  0.0420]]],

                                         [[[ 0.0918,  0.1825],
                                           [-0.1903,  0.1788]],
                                          [[-0.2867, -0.1817],
                                           [-0.1381, -0.1312]],
                                          [[-0.2264,  0.2446],
                                           [-0.1248,  0.0423]]]])
  
  linear_layer2.weight.data = torch.Tensor([[ 0.0572, -0.0070,  0.0352, -0.0337,  0.0676, -0.0399, -0.1211,  0.0260,
                                             -0.1205,  0.0446,  0.0982,  0.0404, -0.0097,  0.0356,  0.0370, -0.0776,
                                             -0.0175,  0.1090, -0.0675, -0.0374,  0.1087, -0.0021,  0.0836,  0.1199,
                                              0.0399, -0.0503, -0.0387, -0.0073,  0.0730, -0.0100,  0.1004,  0.0102,
                                              0.0011,  0.1002,  0.0274, -0.0886,  0.0765,  0.0257,  0.0937, -0.1228,
                                              0.0276,  0.0135,  0.0062,  0.1214, -0.1206, -0.0226,  0.1146, -0.0878,
                                             -0.0437, -0.0562,  0.0009,  0.0959,  0.0101,  0.0677,  0.0863, -0.0880,
                                              0.0703, -0.0215,  0.0746, -0.0708, -0.0759,  0.0399,  0.0418, -0.0443],
                                            
                                            [ 0.0033, -0.0655,  0.0512,  0.0838,  0.0089, -0.0061, -0.1234, -0.0833,
                                             -0.0477, -0.0127, -0.1064, -0.0606,  0.0928,  0.0769,  0.0925, -0.0902,
                                             -0.1232, -0.0244, -0.0882, -0.0614,  0.1186,  0.0046,  0.1204,  0.0593,
                                             -0.0811,  0.0869, -0.1226, -0.0863,  0.0761, -0.1164,  0.0790, -0.0495,
                                              0.1179,  0.0189, -0.0123,  0.0202, -0.0031,  0.0494, -0.0908, -0.0908,
                                             -0.0797,  0.0505,  0.0219,  0.0727, -0.1028,  0.0940, -0.0938, -0.0403,
                                              0.0032, -0.0427, -0.1064, -0.0196, -0.0273,  0.0472, -0.0564, -0.0839,
                                              0.1095, -0.0564, -0.0674, -0.1093,  0.0220, -0.0858, -0.1164,  0.0285],
                                            
                                            [ 0.0080,  0.1081, -0.1150,  0.1225, -0.0302,  0.0536, -0.0165,  0.0924,
                                              0.0191,  0.0874, -0.0570, -0.0876, -0.1161, -0.1037, -0.0591,  0.0121,
                                              0.0936, -0.0514, -0.0152,  0.0186, -0.0833,  0.0069, -0.1193,  0.0800,
                                             -0.1080, -0.0827, -0.0055, -0.0379,  0.0229,  0.0208, -0.0127,  0.0338,
                                              0.0284,  0.1117,  0.0548,  0.0429, -0.0145, -0.0301, -0.0007,  0.0775,
                                              0.0890, -0.1129,  0.0293, -0.1230, -0.0389, -0.0149, -0.0743,  0.1045,
                                              0.0746, -0.1218,  0.0764, -0.1010,  0.0984, -0.0977, -0.0398, -0.0161,
                                             -0.1042, -0.1140,  0.1210,  0.0813, -0.1096, -0.0840,  0.0410,  0.0692]])

  expected_conv = torch.Tensor([[[[-0.1227,  0.1047],
                                  [ 0.0219, -0.0687]],
                                 [[ 0.0389, -0.2334],
                                  [-0.0555,  0.0880]],
                                 [[-0.1174, -0.1749],
                                  [ 0.1261,  0.1615]]],

                                [[[-0.1758,  0.2083],
                                  [ 0.1382,  0.2704]],
                                 [[-0.1103, -0.0255],
                                  [ 0.1005, -0.1327]],
                                 [[ 0.2188,  0.0427],
                                  [ 0.1169, -0.0729]]],

                                [[[ 0.0066,  0.2665],
                                  [-0.1093, -0.1395]],
                                 [[-0.1100, -0.0686],
                                  [-0.1565,  0.0420]],
                                 [[ 0.0200,  0.2130],
                                  [-0.0108,  0.1380]]],

                                [[[-0.2867, -0.1817],
                                  [-0.1381, -0.1312]],
                                 [[-0.2264,  0.2446],
                                  [-0.1248,  0.0423]],
                                 [[ 0.0918,  0.1825],
                                  [-0.1903,  0.1788]]]])
  
  expected_lin2 = torch.Tensor([[-0.0175,  0.1090, -0.0675, -0.0374,  0.1087, -0.0021,  0.0836,  0.1199,
                                  0.0399, -0.0503, -0.0387, -0.0073,  0.0730, -0.0100,  0.1004,  0.0102,
                                  0.0011,  0.1002,  0.0274, -0.0886,  0.0765,  0.0257,  0.0937, -0.1228,
                                  0.0276,  0.0135,  0.0062,  0.1214, -0.1206, -0.0226,  0.1146, -0.0878,
                                  0.0572, -0.0070,  0.0352, -0.0337,  0.0676, -0.0399, -0.1211,  0.0260,
                                 -0.1205,  0.0446,  0.0982,  0.0404, -0.0097,  0.0356,  0.0370, -0.0776,
                                 -0.0437, -0.0562,  0.0009,  0.0959,  0.0101,  0.0677,  0.0863, -0.0880,
                                  0.0703, -0.0215,  0.0746, -0.0708, -0.0759,  0.0399,  0.0418, -0.0443],

                                [-0.1232, -0.0244, -0.0882, -0.0614,  0.1186,  0.0046,  0.1204,  0.0593,
                                 -0.0811,  0.0869, -0.1226, -0.0863,  0.0761, -0.1164,  0.0790, -0.0495,
                                  0.1179,  0.0189, -0.0123,  0.0202, -0.0031,  0.0494, -0.0908, -0.0908,
                                 -0.0797,  0.0505,  0.0219,  0.0727, -0.1028,  0.0940, -0.0938, -0.0403,
                                  0.0033, -0.0655,  0.0512,  0.0838,  0.0089, -0.0061, -0.1234, -0.0833,
                                 -0.0477, -0.0127, -0.1064, -0.0606,  0.0928,  0.0769,  0.0925, -0.0902,
                                  0.0032, -0.0427, -0.1064, -0.0196, -0.0273,  0.0472, -0.0564, -0.0839,
                                  0.1095, -0.0564, -0.0674, -0.1093,  0.0220, -0.0858, -0.1164,  0.0285],
                                  
                                [ 0.0936, -0.0514, -0.0152,  0.0186, -0.0833,  0.0069, -0.1193,  0.0800,
                                 -0.1080, -0.0827, -0.0055, -0.0379,  0.0229,  0.0208, -0.0127,  0.0338,
                                  0.0284,  0.1117,  0.0548,  0.0429, -0.0145, -0.0301, -0.0007,  0.0775,
                                  0.0890, -0.1129,  0.0293, -0.1230, -0.0389, -0.0149, -0.0743,  0.1045,
                                  0.0080,  0.1081, -0.1150,  0.1225, -0.0302,  0.0536, -0.0165,  0.0924,
                                  0.0191,  0.0874, -0.0570, -0.0876, -0.1161, -0.1037, -0.0591,  0.0121,
                                  0.0746, -0.1218,  0.0764, -0.1010,  0.0984, -0.0977, -0.0398, -0.0161,
                                 -0.1042, -0.1140,  0.1210,  0.0813, -0.1096, -0.0840,  0.0410,  0.0692]])

  ns.swap_input_channels(conv_layer, linear_layer, [1, 2])
  ns.swap_input_channels(linear_layer2,conv_layer, [1, 2])

  assert torch.equal(conv_layer.weight.data, expected_conv), f'Incorrect swap of input channel in convolution layer'
  assert torch.equal(linear_layer2.weight.data, expected_lin2), f'Incorrect swap of input channel in linear layer'

def test_swap_bn_layer():
  bn_layer = nn.BatchNorm2d(5)

  bn_layer.weight.data = torch.Tensor([0.4137, 0.6493, 0.5521, 0.3699, 0.2557])
  bn_layer.bias.data = torch.Tensor([0.0846, 0.2871, 0.9143, 0.6967, 0.9102])
  bn_layer.running_mean.data = torch.Tensor([0.8453, 0.5144, 0.8725, 0.2729, 0.4508])
  bn_layer.running_var.data = torch.Tensor([0.5508, 0.9586, 0.4615, 0.9354, 0.3631])

  ns.swap_bn_modules(bn_layer, [1, 3])

  exp_weight = torch.Tensor([0.6493, 0.3699, 0.4137, 0.5521, 0.2557])
  exp_bias = torch.Tensor([0.2871, 0.6967, 0.0846, 0.9143, 0.9102])
  exp_mean = torch.Tensor([0.5144, 0.2729, 0.8453, 0.8725, 0.4508])
  exp_var = torch.Tensor([0.9586, 0.9354, 0.5508, 0.4615, 0.3631])
  
  assert torch.equal(bn_layer.weight.data, exp_weight), f'Incorrect weight swap: expecting {exp_weight}\ngot instead {bn_layer.weight.data}'
  assert torch.equal(bn_layer.bias.data, exp_bias), f'Incorrect bias swap: expecting {exp_bias}\ngot instead {bn_layer.bias.data}'
  assert torch.equal(bn_layer.running_mean.data, exp_mean), f'Incorrect mean swap: expecting {exp_mean}\ngot instead {bn_layer.running_mean.data}'
  assert torch.equal(bn_layer.running_var.data, exp_var), f'Incorrect var swap: expecting {exp_var}\ngot instead {bn_layer.running_var.data}'
  
def test_swap_fc():
  model = FCnet()
  eq_indexes = {"fc1": torch.tensor([4, 9], device='cpu'),
                "fc2": torch.tensor([3, 6], device='cpu'),
                "fc3": torch.tensor([1, 2], device='cpu')}

  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)

  input = torch.rand(5)

  output_before = model(input)

  ns.swap(layers_list, eq_indexes)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8)

def test_swap_conv():
  model = ConvNetwork()

  eq_indexes = {"conv1": torch.Tensor([2, 4], device='cpu'),
                "conv2": torch.Tensor([5, 10], device='cpu'),
                "fc1": torch.tensor([4, 9], device='cpu'),
                "fc2": torch.tensor([3, 6], device='cpu'),
                "fc3": torch.tensor([1, 2], device='cpu')}

  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)

  input = torch.rand([1, 28, 28])

  output_before = model(input)

  ns.swap(layers_list, eq_indexes)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8)

def test_swap_bn():
  model = ConvBN()

  eq_indexes = {"conv1": torch.Tensor([1], device='cpu'),
                "conv2": torch.Tensor([0, 2], device='cpu'),
                "fc1": torch.tensor([4, 7], device='cpu'),
                "fc3": torch.tensor([1, 2], device='cpu')}

  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)

  input = torch.rand([1, 4, 24, 24])

  model.train(False)

  output_before = model(input)

  ns.swap(layers_list, eq_indexes)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8)

def test_swap_resnet():
  model = resnet18()

  eq_indexes = {'layer1.0.conv1': [2,3], 
                'layer1.1.conv1': [2,3], 
                'layer2.0.conv1': [2,3], 
                'layer2.1.conv1': [2,3], 
                'layer3.0.conv1': [2,3], 
                'layer3.1.conv1': [2,3], 
                'layer4.0.conv1': [2,3], 
                'layer4.1.conv1': [2,3], }

  graph = fx.symbolic_trace(model).graph
  layers_list = modx.get_layers_list(graph, model)
  skip_connections = modx.get_skipped_layers(graph, layers_list)

  input = torch.rand([1,3,244,244])

  model.train(False)

  output_before = model(input)

  ns.swap(layers_list, eq_indexes, skip_connections)

  output_after = model(input)

  assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8)

