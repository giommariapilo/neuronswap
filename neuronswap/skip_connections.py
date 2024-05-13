import re
import torch
from torch import nn, fx
from .linear_layer import *
from .conv_layer import *
from .make_freezable import substitute_layer_names, clean_code


def clean_code(text):
    rgx_list = [r'    ', r'(;  .* = None)']
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, '', new_text)
    return new_text

def substitute_layer_names(code, model):
    rgx_list = [rf'self.{name}' for name, _ in model.named_children()]
    new_code = code
    for index in range(len(rgx_list)):
        new_code = re.sub(rgx_list[index], f'self.layers[{index}]', new_code)
    return new_code



class SkipModelSequential(nn.Module):
    def __init__(self, model: nn.Module):
        super(SkipModelSequential, self).__init__()

# idea in the skip connection inject a sequential... dnn
# Figured!!!
# After loading model，we can directly specify 
# model.conv_x = nn.Sequential([new_layer, model.conv_x]), 
# by this way, we can still use thepretrained model.conv_x

# im still missing a bit... i want to insert it only on the skip connection
# thats the hard part    

def crazy_good_skip_connection_function(model, skip_connections):
    
    pass
class SkipModelExec(nn.Module):
    def __init__(self, model: nn.Module,skip_connection_list):
        super(SkipModelExec, self).__init__()
        self.layers = {name: layer for name, layer in model.named_children()}
        for name in self.layers.keys():
            if isinstance(self.layers[name], nn.Linear):
                layer_shape = self.layers[name].shape
                self.layers[name] = FreezableLinear(layer_shape[0], layer_shape[1])
            if isinstance(self.layers[name], nn.Conv2d):
                layer_shape = self.layers[name].shape
                self.layers[name] = FreezableConv2d(layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3])
        self.code = fx.symbolic_trace(model).code[25:-15].strip()+'\nx = fc3' # change this fc3 is not always right
        self.code = substitute_layer_names(clean_code(self.code), model)

    def forward(self, x):
        namespace = {'x': x}
        exec(self.code, namespace)
        return namespace['x']
    
def forward(self, x : torch.Tensor) -> torch.Tensor:
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    relu = self.relu(bn1);  bn1 = None
    maxpool = self.maxpool(relu);  relu = None
    layer1_0_conv1 = getattr(self.layer1, "0").conv1(maxpool)
    layer1_0_bn1 = getattr(self.layer1, "0").bn1(layer1_0_conv1);  layer1_0_conv1 = None
    layer1_0_relu = getattr(self.layer1, "0").relu(layer1_0_bn1);  layer1_0_bn1 = None
    layer1_0_conv2 = getattr(self.layer1, "0").conv2(layer1_0_relu);  layer1_0_relu = None
    layer1_0_bn2 = getattr(self.layer1, "0").bn2(layer1_0_conv2);  layer1_0_conv2 = None
    add = layer1_0_bn2 + maxpool;  layer1_0_bn2 = maxpool = None
    layer1_0_relu_1 = getattr(self.layer1, "0").relu(add);  add = None
    layer1_1_conv1 = getattr(self.layer1, "1").conv1(layer1_0_relu_1)
    layer1_1_bn1 = getattr(self.layer1, "1").bn1(layer1_1_conv1);  layer1_1_conv1 = None
    layer1_1_relu = getattr(self.layer1, "1").relu(layer1_1_bn1);  layer1_1_bn1 = None
    layer1_1_conv2 = getattr(self.layer1, "1").conv2(layer1_1_relu);  layer1_1_relu = None
    layer1_1_bn2 = getattr(self.layer1, "1").bn2(layer1_1_conv2);  layer1_1_conv2 = None
    add_1 = layer1_1_bn2 + layer1_0_relu_1;  layer1_1_bn2 = layer1_0_relu_1 = None
    layer1_1_relu_1 = getattr(self.layer1, "1").relu(add_1);  add_1 = None
    layer2_0_conv1 = getattr(self.layer2, "0").conv1(layer1_1_relu_1)
    layer2_0_bn1 = getattr(self.layer2, "0").bn1(layer2_0_conv1);  layer2_0_conv1 = None
    layer2_0_relu = getattr(self.layer2, "0").relu(layer2_0_bn1);  layer2_0_bn1 = None
    layer2_0_conv2 = getattr(self.layer2, "0").conv2(layer2_0_relu);  layer2_0_relu = None
    layer2_0_bn2 = getattr(self.layer2, "0").bn2(layer2_0_conv2);  layer2_0_conv2 = None
    layer2_0_downsample_0 = getattr(getattr(self.layer2, "0").downsample, "0")(layer1_1_relu_1);  layer1_1_relu_1 = None
    layer2_0_downsample_1 = getattr(getattr(self.layer2, "0").downsample, "1")(layer2_0_downsample_0);  layer2_0_downsample_0 = None
    add_2 = layer2_0_bn2 + layer2_0_downsample_1;  layer2_0_bn2 = layer2_0_downsample_1 = None
    layer2_0_relu_1 = getattr(self.layer2, "0").relu(add_2);  add_2 = None
    layer2_1_conv1 = getattr(self.layer2, "1").conv1(layer2_0_relu_1)
    layer2_1_bn1 = getattr(self.layer2, "1").bn1(layer2_1_conv1);  layer2_1_conv1 = None
    layer2_1_relu = getattr(self.layer2, "1").relu(layer2_1_bn1);  layer2_1_bn1 = None
    layer2_1_conv2 = getattr(self.layer2, "1").conv2(layer2_1_relu);  layer2_1_relu = None
    layer2_1_bn2 = getattr(self.layer2, "1").bn2(layer2_1_conv2);  layer2_1_conv2 = None
    add_3 = layer2_1_bn2 + layer2_0_relu_1;  layer2_1_bn2 = layer2_0_relu_1 = None
    layer2_1_relu_1 = getattr(self.layer2, "1").relu(add_3);  add_3 = None
    layer3_0_conv1 = getattr(self.layer3, "0").conv1(layer2_1_relu_1)
    layer3_0_bn1 = getattr(self.layer3, "0").bn1(layer3_0_conv1);  layer3_0_conv1 = None
    layer3_0_relu = getattr(self.layer3, "0").relu(layer3_0_bn1);  layer3_0_bn1 = None
    layer3_0_conv2 = getattr(self.layer3, "0").conv2(layer3_0_relu);  layer3_0_relu = None
    layer3_0_bn2 = getattr(self.layer3, "0").bn2(layer3_0_conv2);  layer3_0_conv2 = None
    layer3_0_downsample_0 = getattr(getattr(self.layer3, "0").downsample, "0")(layer2_1_relu_1);  layer2_1_relu_1 = None
    layer3_0_downsample_1 = getattr(getattr(self.layer3, "0").downsample, "1")(layer3_0_downsample_0);  layer3_0_downsample_0 = None
    add_4 = layer3_0_bn2 + layer3_0_downsample_1;  layer3_0_bn2 = layer3_0_downsample_1 = None
    layer3_0_relu_1 = getattr(self.layer3, "0").relu(add_4);  add_4 = None
    layer3_1_conv1 = getattr(self.layer3, "1").conv1(layer3_0_relu_1)
    layer3_1_bn1 = getattr(self.layer3, "1").bn1(layer3_1_conv1);  layer3_1_conv1 = None
    layer3_1_relu = getattr(self.layer3, "1").relu(layer3_1_bn1);  layer3_1_bn1 = None
    layer3_1_conv2 = getattr(self.layer3, "1").conv2(layer3_1_relu);  layer3_1_relu = None
    layer3_1_bn2 = getattr(self.layer3, "1").bn2(layer3_1_conv2);  layer3_1_conv2 = None
    add_5 = layer3_1_bn2 + layer3_0_relu_1;  layer3_1_bn2 = layer3_0_relu_1 = None
    layer3_1_relu_1 = getattr(self.layer3, "1").relu(add_5);  add_5 = None
    layer4_0_conv1 = getattr(self.layer4, "0").conv1(layer3_1_relu_1)
    layer4_0_bn1 = getattr(self.layer4, "0").bn1(layer4_0_conv1);  layer4_0_conv1 = None
    layer4_0_relu = getattr(self.layer4, "0").relu(layer4_0_bn1);  layer4_0_bn1 = None
    layer4_0_conv2 = getattr(self.layer4, "0").conv2(layer4_0_relu);  layer4_0_relu = None
    layer4_0_bn2 = getattr(self.layer4, "0").bn2(layer4_0_conv2);  layer4_0_conv2 = None
    layer4_0_downsample_0 = getattr(getattr(self.layer4, "0").downsample, "0")(layer3_1_relu_1);  layer3_1_relu_1 = None
    layer4_0_downsample_1 = getattr(getattr(self.layer4, "0").downsample, "1")(layer4_0_downsample_0);  layer4_0_downsample_0 = None
    add_6 = layer4_0_bn2 + layer4_0_downsample_1;  layer4_0_bn2 = layer4_0_downsample_1 = None
    layer4_0_relu_1 = getattr(self.layer4, "0").relu(add_6);  add_6 = None
    layer4_1_conv1 = getattr(self.layer4, "1").conv1(layer4_0_relu_1)
    layer4_1_bn1 = getattr(self.layer4, "1").bn1(layer4_1_conv1);  layer4_1_conv1 = None
    layer4_1_relu = getattr(self.layer4, "1").relu(layer4_1_bn1);  layer4_1_bn1 = None
    layer4_1_conv2 = getattr(self.layer4, "1").conv2(layer4_1_relu);  layer4_1_relu = None
    layer4_1_bn2 = getattr(self.layer4, "1").bn2(layer4_1_conv2);  layer4_1_conv2 = None
    add_7 = layer4_1_bn2 + layer4_0_relu_1;  layer4_1_bn2 = layer4_0_relu_1 = None
    layer4_1_relu_1 = getattr(self.layer4, "1").relu(add_7);  add_7 = None
    avgpool = self.avgpool(layer4_1_relu_1);  layer4_1_relu_1 = None
    flatten = torch.flatten(avgpool, 1);  avgpool = None
    fc = self.fc(flatten);  flatten = None
    return fc