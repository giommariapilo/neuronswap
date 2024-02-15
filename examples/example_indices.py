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

ns.permute(layers_list, eq_indexes)

output_after = model(input)

print(torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-8))