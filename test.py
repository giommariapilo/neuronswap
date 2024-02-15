import sys
sys.path.append('..')
import neuronswap.optimizermatrixswap as omswap
import neuronswap.optimizerindexswap as oiswap
import neuronswap as ns
import torch
from torch import fx, nn, optim
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms, datasets

torch.set_default_device('cuda')

class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.AdaptiveAvgPool2d(output_size=1)
    self.fc3 = nn.Linear(6, 10)
    self.bn = nn.BatchNorm2d(6)

  def forward(self, x):

    x = F.relu(self.bn(self.conv1(x)))
    x = self.pool(x)
    x = x.squeeze(-1).squeeze(-1)
    x = self.fc3(x)
    return x

def train_one_batch(model, optimizer, training_loader, loss_fn):
  running_loss = 0.

  for i, data in enumerate(training_loader):
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = loss_fn(outputs, labels)
    loss.backward()

    optimizer.step()

    running_loss += loss.item()
    
    break

  return 


model = Classifier( )
model.to('cuda')
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

training_set = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)


training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, generator=torch.Generator(device = 'cuda'))

loss_fn = torch.nn.CrossEntropyLoss()

model.train(True)
train_one_batch(model, optimizer, training_loader, loss_fn)

model.eval()

graph = fx.symbolic_trace(model).graph
layers_list = ns.get_layers_list(graph, model)
permutation_matrices = {"conv1": torch.Tensor([[0, 1, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0],
                                               [1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 1]]).cuda(),

              "fc3": torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]).cuda()} #this shouldnt swap
                                  

permutation_indices = {'conv1': torch.Tensor([2]).cpu(),
                       'fc3': torch.Tensor([2]).cpu()}

weights = [torch.Tensor([[[[-0.0161, -0.0293,  0.0048,  0.0346,  0.0655],
                           [-0.0117, -0.0193,  0.0188,  0.0464,  0.0757],
                           [-0.0211, -0.0304,  0.0093,  0.0377,  0.0697],
                           [-0.0321, -0.0398,  0.0038,  0.0306,  0.0583],
                           [-0.0431, -0.0477, -0.0052,  0.0254,  0.0520]]],

                         [[[ 0.0076, -0.0133, -0.0318, -0.0556, -0.0619],
                           [ 0.0021, -0.0206, -0.0316, -0.0460, -0.0460],
                           [-0.0296, -0.0505, -0.0533, -0.0629, -0.0595],
                           [ 0.0205,  0.0040,  0.0017, -0.0077, -0.0142],
                           [ 0.0008, -0.0136, -0.0119, -0.0346, -0.0469]]],

                          [[[-0.0013,  0.0186,  0.0145,  0.0273,  0.0534],
                           [-0.0054,  0.0114,  0.0105,  0.0249,  0.0519],
                           [-0.0288, -0.0054,  0.0083,  0.0301,  0.0554],
                           [-0.0332, -0.0027,  0.0166,  0.0409,  0.0595],
                           [-0.0090,  0.0199,  0.0262,  0.0473,  0.0678]]],

                         [[[ 0.0181,  0.0133,  0.0030,  0.0042,  0.0051],
                           [ 0.0117,  0.0135,  0.0035, -0.0004,  0.0034],
                           [ 0.0133,  0.0111,  0.0046,  0.0021,  0.0098],
                           [ 0.0030, -0.0006, -0.0046, -0.0034,  0.0047],
                           [-0.0132, -0.0105, -0.0107, -0.0132, -0.0054]]],

                          [[[-0.0440, -0.0150, -0.0027, -0.0047, -0.0194],
                           [-0.0054,  0.0081,  0.0246,  0.0198,  0.0013],
                           [ 0.0109,  0.0251,  0.0335,  0.0325,  0.0133],
                           [ 0.0082,  0.0160,  0.0266,  0.0231,  0.0073],
                           [-0.0115, -0.0098,  0.0026,  0.0039, -0.0099]]],

                         [[[ 0.0027, -0.0131,  0.0079,  0.0553,  0.0853],
                           [ 0.0119, -0.0090,  0.0049,  0.0521,  0.0845],
                           [ 0.0045, -0.0228, -0.0106,  0.0395,  0.0701],
                           [-0.0115, -0.0364, -0.0271,  0.0222,  0.0508],
                           [-0.0228, -0.0444, -0.0372,  0.0065,  0.0337]]]]).cuda(),
            torch.Tensor([-1.0001e-06,  1.4361e-07,  2.1791e-07,  7.8905e-08,  3.2799e-08, -2.9274e-07]).cuda(),
            torch.Tensor([[-0.2084, -0.0139, -0.2453,  0.0028, -0.0492, -0.1662],
                          [ 0.0666,  0.1337,  0.0296,  0.1523,  0.1280,  0.0652],
                          [ 0.3016, -0.1294,  0.3100, -0.1256, -0.1499,  0.2443],
                          [-0.3232, -0.2988, -0.3996, -0.2896, -0.3537, -0.2992],
                          [ 0.2282, -0.1079,  0.2243, -0.0960, -0.1282,  0.1889],
                          [-0.2248,  0.2253, -0.2151,  0.2181,  0.2514, -0.1786],
                          [ 0.2776,  0.0516,  0.2964,  0.0557,  0.0542,  0.2379],
                          [-0.1901,  0.1522, -0.1563,  0.1251,  0.1941, -0.1525],
                          [ 0.2329, -0.0790,  0.3072, -0.0956, -0.0434,  0.2006],
                          [-0.1605,  0.0662, -0.1511,  0.0529,  0.0967, -0.1404]]).cuda(),
            torch.Tensor([-0.4251,  0.2420,  0.1216, -0.5012,  0.1791,  0.0572,  0.2810, -0.0100,  0.1548, -0.0994]).cuda(),
            torch.Tensor([-0.1273, -0.1601, -0.3394, -0.0871, -0.1493, -0.0837]).cuda(),
            torch.Tensor([-0.0976, -0.1095, -0.2483, -0.1089, -0.1593, -0.0851]).cuda()]
expected_weights = [torch.Tensor([[[[ 0.0076, -0.0133, -0.0318, -0.0556, -0.0619],
                                    [ 0.0021, -0.0206, -0.0316, -0.0460, -0.0460],
                                    [-0.0296, -0.0505, -0.0533, -0.0629, -0.0595],
                                    [ 0.0205,  0.0040,  0.0017, -0.0077, -0.0142],
                                    [ 0.0008, -0.0136, -0.0119, -0.0346, -0.0469]]],

                                  [[[-0.0013,  0.0186,  0.0145,  0.0273,  0.0534],
                                    [-0.0054,  0.0114,  0.0105,  0.0249,  0.0519],
                                    [-0.0288, -0.0054,  0.0083,  0.0301,  0.0554],
                                    [-0.0332, -0.0027,  0.0166,  0.0409,  0.0595],
                                    [-0.0090,  0.0199,  0.0262,  0.0473,  0.0678]]],

                                  [[[-0.0161, -0.0293,  0.0048,  0.0346,  0.0655],
                                    [-0.0117, -0.0193,  0.0188,  0.0464,  0.0757],
                                    [-0.0211, -0.0304,  0.0093,  0.0377,  0.0697],
                                    [-0.0321, -0.0398,  0.0038,  0.0306,  0.0583],
                                    [-0.0431, -0.0477, -0.0052,  0.0254,  0.0520]]],

                                  [[[ 0.0181,  0.0133,  0.0030,  0.0042,  0.0051],
                                    [ 0.0117,  0.0135,  0.0035, -0.0004,  0.0034],
                                    [ 0.0133,  0.0111,  0.0046,  0.0021,  0.0098],
                                    [ 0.0030, -0.0006, -0.0046, -0.0034,  0.0047],
                                    [-0.0132, -0.0105, -0.0107, -0.0132, -0.0054]]],

                                  [[[-0.0440, -0.0150, -0.0027, -0.0047, -0.0194],
                                    [-0.0054,  0.0081,  0.0246,  0.0198,  0.0013],
                                    [ 0.0109,  0.0251,  0.0335,  0.0325,  0.0133],
                                    [ 0.0082,  0.0160,  0.0266,  0.0231,  0.0073],
                                    [-0.0115, -0.0098,  0.0026,  0.0039, -0.0099]]],

                                  [[[ 0.0027, -0.0131,  0.0079,  0.0553,  0.0853],
                                    [ 0.0119, -0.0090,  0.0049,  0.0521,  0.0845],
                                    [ 0.0045, -0.0228, -0.0106,  0.0395,  0.0701],
                                    [-0.0115, -0.0364, -0.0271,  0.0222,  0.0508],
                                    [-0.0228, -0.0444, -0.0372,  0.0065,  0.0337]]]]).cuda(),
                    torch.Tensor([ 1.4361e-07,  2.1791e-07, -1.0001e-06,  7.8905e-08,  3.2799e-08, -2.9274e-07]).cuda(),
                    torch.Tensor([[-0.0139, -0.2453, -0.2084,  0.0028, -0.0492, -0.1662],
                                  [ 0.1337,  0.0296,  0.0666,  0.1523,  0.1280,  0.0652],
                                  [-0.1294,  0.3100,  0.3016, -0.1256, -0.1499,  0.2443],
                                  [-0.2988, -0.3996, -0.3232, -0.2896, -0.3537, -0.2992],
                                  [-0.1079,  0.2243,  0.2282, -0.0960, -0.1282,  0.1889],
                                  [ 0.2253, -0.2151, -0.2248,  0.2181,  0.2514, -0.1786],
                                  [ 0.0516,  0.2964,  0.2776,  0.0557,  0.0542,  0.2379],
                                  [ 0.1522, -0.1563, -0.1901,  0.1251,  0.1941, -0.1525],
                                  [-0.0790,  0.3072,  0.2329, -0.0956, -0.0434,  0.2006],
                                  [ 0.0662, -0.1511, -0.1605,  0.0529,  0.0967, -0.1404]]).cuda(),
                    torch.Tensor([-0.4251,  0.2420,  0.1216, -0.5012,  0.1791,  0.0572,  0.2810, -0.0100,  0.1548, -0.0994]).cuda(),
                    torch.Tensor([-0.1601, -0.3394, -0.1273, -0.0871, -0.1493, -0.0837]).cuda(),
                    torch.Tensor([-0.1095, -0.2483, -0.0976, -0.1089, -0.1593, -0.0851]).cuda()]
for index in range(len(weights)):
  optimizer.state_dict()['state'][index]['momentum_buffer'] = weights[index]

ns.moswap(layers_list, permutation_matrices, model, optimizer)

for index in range(len(weights)):
  assert torch.equal(expected_weights[index], optimizer.state_dict()['state'][index]['momentum_buffer']), f'ERROR: Different result of swap by matrix operation on parameter {index}:\nGot {optimizer.state_dict()["state"][index]["momentum_buffer"]}\n Expected {expected_weights[index]}'

ns.ioswap(layers_list, permutation_indices, model, optimizer)

for index in range(len(weights)):
  assert torch.equal(weights[index], optimizer.state_dict()['state'][index]['momentum_buffer']), f'ERROR: Different result of swap by index operation on parameter {index}:\nGot {optimizer.state_dict()["state"][index]["momentum_buffer"]}\n Expected {weights[index]}'
