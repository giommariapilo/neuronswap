import sys
sys.path.append('..')
from copy import deepcopy
import os
import random as rd
import torch

import neuronswap.nswap as ns
import neuronswap.modulexplore as modx


def set_seed(seed):
    rd.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def record_in_out(m_, x, y):
        x = x[0]
        m_.input = x
        m_.output = y


def add_input_output_hook(m_):
    m_.register_forward_hook(record_in_out)


def get_all_linear_ops_with_names(model):
    linears = []
    names = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            linears.append(m)
            names.append(name)
    return linears[:-1], names[:-1]


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


def generate_random_mask_with_permutation(linears, names):
    random_mask = {}
    permutations = {}
    for linear, name in zip(linears, names):
        num_neurons = linear.output.shape[0]
        permutation = torch.randperm(num_neurons)
        num_eq_neurons = rd.randint(0, num_neurons - 1)
        random_mask[name] = permutation[:num_eq_neurons].sort().values
        permutations[name] = torch.cat([permutation[:num_eq_neurons].sort().values, permutation[num_eq_neurons:].sort().values])
    return random_mask, permutations

def test_fully_connected_random_swap():
    set_seed(3)
    model = FCnet()
    graph = torch.fx.symbolic_trace(model).graph
    layers_list = modx.get_layers_list(graph, model)
    skip_connections = modx.get_skipped_layers(graph, layers_list)
    linears, names = get_all_linear_ops_with_names(model)

    model.apply(add_input_output_hook)

    sample_input = torch.rand(5)

    model.train(False)

    with torch.no_grad():
        _ = model(sample_input)

    pre_swapped_outputs = {}
    for linear, name in zip(linears, names):
        pre_swapped_outputs[name] = deepcopy(linear.output)

    random_mask, permutations = generate_random_mask_with_permutation(linears, names)

    ns.swap(layers_list, random_mask, skip_connections)

    with torch.no_grad():
        _ = model(sample_input)

    swapped_outputs = {}
    for linear, name in zip(linears, names):
        swapped_outputs[name] =  deepcopy(linear.output)

    for name in names:
        pre_swapped_output = pre_swapped_outputs[name]
        permutation = permutations[name]
        manually_swapped_output = pre_swapped_output[permutation]
        assert torch.equal(manually_swapped_output, swapped_outputs[name]), f"{name}'s outputs do not match"
