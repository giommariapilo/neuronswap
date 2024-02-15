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

ns.permute(layers_list, eq_indexes, skip_connections)

output_after = model(input)

print(torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-6))