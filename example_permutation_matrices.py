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