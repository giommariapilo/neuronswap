import torch
from torch import nn, optim
from torchvision import transforms, datasets
import torch.nn.functional as F
import neuronswap as ns

class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
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

model = GarmentClassifier()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

model.train(True)
_ = train_one_epoch()

print('STATE DICT')
print(optimizer.state_dict())

for item in optimizer.state_dict()['state'].values():
  print(item['momentum_buffer'])

# i create the permutation matrix automatically, the same for each layer
permutations = {}
for name, layer in model.named_modules():
  if isinstance(layer, (nn.Conv2d, nn.Linear)):
    matrix = torch.zeros(layer.weight.data.shape[0], layer.weight.data.shape[0], dtype = torch.float32)
    for i in range(layer.weight.data.shape[0]):
      if 0 == i:
        matrix[i, 2] = 1
      elif 2 == i:
        matrix[i, 0] = 1
      else:
        matrix[i, i] = 1
    if matrix.shape == torch.tensor([2, 2]):
      matrix = torch.tensor([[0, 1],
                              [1, 0]],
                            dtype = torch.float32)
    permutations[name] = matrix

layers_list = ns.get_layers_list(torch.fx.symbolic_trace(model).graph, model)
ns.permutate_optimizer(layers_list, permutations, model, optimizer)
for item in optimizer.state_dict()['state'].values():
  print(item['momentum_buffer'])
