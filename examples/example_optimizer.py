import torch
from torch import nn, optim
from torchvision import transforms, datasets
import torch.nn.functional as F
import neuronswap as ns

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

def train_one_epoch():
  running_loss = 0.
  last_loss = 0.
  for i, data in enumerate(training_loader):
    inputs, labels = data

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = loss_fn(outputs, labels)
    loss.backward()

    optimizer.step()

    running_loss += loss.item()
    if i % 1000 == 999:
      last_loss = running_loss / 1000 
      break

  return last_loss


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

training_set = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

model = Classifier()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

model.train(True)
_ = train_one_epoch()

print('STATE DICT')
print(optimizer.state_dict())

for item in optimizer.state_dict()['state'].values():
  print(item['momentum_buffer'])

permutations = {"conv1": torch.tensor([[0, 0, 1, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]],
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
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], #this shouldnt swap
                                    dtype = torch.float32)}

layers_list = ns.get_layers_list(torch.fx.symbolic_trace(model).graph, model)
ns.permutate_optimizer(layers_list, permutations, model, optimizer)
for item in optimizer.state_dict()['state'].values():
  print(item['momentum_buffer'])
