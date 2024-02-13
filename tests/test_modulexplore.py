import sys
sys.path.append('..')
import neuronswap.modulexplore as modx
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

def test_get_fully_connected():
  model = FCnet()

  model.train(False)

  graph = fx.symbolic_trace(model).graph

  layers_list = modx.get_layers_list(graph, model)

  # check against the known good order
  known_good_list = [('fc1', model.fc1), ('fc2', model.fc2), 
                      ('fc3', model.fc3)]
  assert layers_list == known_good_list, "The two lists are not in the same order"

def test_get_conv():
  model = ConvNetwork()

  model.train(False)

  graph = fx.symbolic_trace(model).graph

  layers_list = modx.get_layers_list(graph, model)

  # check against the known good order
  known_good_list = [('conv1', model.conv1), ('conv2', model.conv2), 
                      ('fc1', model.fc1), ('fc2', model.fc2), 
                      ('fc3', model.fc3)]
  assert layers_list == known_good_list, "The two lists are not in the same order"

def test_get_batch_norm():
  model = ConvBN()

  model.train(False)

  graph = fx.symbolic_trace(model).graph

  layers_list = modx.get_layers_list(graph, model)

  # check against the known good order
  known_good_list = [('conv1', model.conv1), ('bn1', model.bn1), 
                      ('conv2', model.conv2), ('fc1', model.fc1), 
                      ('fc3', model.fc3)]
  assert layers_list == known_good_list, "The two lists are not in the same order"

def test_parent_search():
  model = resnet18()
  
  expected_parents = ['layer1.1.conv2', 'layer1.0.conv2', 'conv1']

  graph = fx.symbolic_trace(model).graph
  list_layers = modx.get_layers_list(graph, model)

  parents = []
  for node in graph.nodes:
    if node.name == 'add_1':
      for parent in node.args:
        modx.parent_search(parents, parent, list_layers)    

  assert set(parents) == set(expected_parents), f'\nParents list:\n{parents}\n'

  
def test_search_skip():
  model = resnet18()
  model.train(False)

  graph = fx.symbolic_trace(model).graph

  layers_list = modx.get_layers_list(graph, model)

  expected = ['layer3.1.conv2', 'layer1.1.conv2', 'layer4.1.conv2', 
              'layer2.0.conv2', 'layer4.0.downsample.0', 'conv1', 
              'layer4.0.conv2', 'layer2.0.downsample.0', 'layer3.0.downsample.0', 
              'layer1.0.conv2', 'layer2.1.conv2', 'layer3.0.conv2']
  # print(layers_list)

  skip_connections = modx.get_skipped_layers(graph, layers_list)

  assert set(skip_connections) == set(expected), f'Incorrect skip connections found, \nexpected: {expected}, \nfound: {skip_connections}'
