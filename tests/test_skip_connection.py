import sys
sys.path.append('..')
from copy import deepcopy
import os
import random as rd
import torch
from torchvision.models import resnet18
import pytest

import neuronswap as ns


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


def get_all_conv_ops_with_names(model):
    convs = []
    names = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            convs.append(m)
            names.append(name)
    return convs, names


def generate_random_mask_with_permutation(convs, names):
    random_mask = {}
    permutations = {}
    for conv, name in zip(convs, names):
        num_neurons = conv.out_channels
        permutation = torch.randperm(num_neurons)
        num_eq_neurons = rd.randint(0, num_neurons - 1)
        random_mask[name] = permutation[:num_eq_neurons].sort().values
        permutations[name] = torch.cat([permutation[:num_eq_neurons].sort().values, permutation[num_eq_neurons:].sort().values])
    return random_mask, permutations

@pytest.mark.xfail
def test_skip_connections():
    set_seed(10)
    model = resnet18()
    graph = torch.fx.symbolic_trace(model).graph
    layers_list = ns.get_layers_list(graph, model)
    skip_connections = ns.get_skipped_layers(graph, layers_list)
    convs, names = get_all_conv_ops_with_names(model)

    model.apply(add_input_output_hook)

    sample_input = torch.randn(1, 3, 128, 128)

    model.train(False)

    with torch.no_grad():
        _ = model(sample_input)

    pre_swapped_outputs = {}
    for conv, name in zip(convs, names):
        pre_swapped_outputs[name] = deepcopy(conv.output.view(conv.output.shape[1:]))


    random_mask, permutations = generate_random_mask_with_permutation(convs, names)

    ns.permutate(layers_list, random_mask, skip_connections)

    with torch.no_grad():
        _ = model(sample_input)

    swapped_outputs = {}
    for conv, name in zip(convs, names):
        swapped_outputs[name] =  deepcopy(conv.output.view(conv.output.shape[1:]))

    for name in names:
        if name not in skip_connections:
            pre_swapped_output = pre_swapped_outputs[name]
            permutation = permutations[name]
            manually_swapped_output = pre_swapped_output[permutation]
            assert torch.allclose(manually_swapped_output, swapped_outputs[name], rtol=1e-5, atol=1e-5), f"{name}'s outputs do not match"
            