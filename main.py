import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader , random_split
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
  nn.Flatten(), # 3x128x128 -> 49152
  nn.Linear(3*128*128, 512), nn.ReLU(), nn.Dropout(0.4),
  nn.Linear(512, 256), nn.ReLU(),
  nn.Linear(256, 101), # 101 food classes
  )
  def forward(self, x):
    return self.net(x)

def imshow(img):
    img = img * 0.5 + 0.5   # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")

transform = transforms.Compose([
    transforms.Resize(128), # resize any image to 128 so that all images to same size
    transforms.CenterCrop(128),# croping the center part #128*128
    transforms.ToTensor(), # converting to tensor-  changing its shape and pixel to 0-1-
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])# convrting close to 08 -1 +1
])

full_train = datasets.Food101(root='data', split='train', download=True, transform=transform)
test_ds = datasets.Food101(root='data', split='test', download=True, transform=transform)

print(len(full_train))
print(len(test_ds))
n_val = int(len(full_train) * 0.1)
n_tr=len(full_train) - n_val
train_ds, val_ds = random_split(full_train, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
print(len(train_ds))
print(len(val_ds))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,num_workers=2)
print(len(train_loader))
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,num_workers=2)
print(len(val_loader))
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,num_workers=2)
print(len(test_loader))

dataiter = iter(train_loader) # iter - we need to talk about
images,labels = next(dataiter)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 10
for epoch in range(epochs):
  model.train()
  running_loss = 0

  for images , labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  avg_loss = running_loss / len(train_loader)
  print(f"Epoch : {epoch} , Loss : {avg_loss}")

  #CNN
class FoodCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # 3 channels in,  3 to 32 filters(featurs)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # extra block for a
    self.pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(128 * 16 * 16, 256) # 128 -> 64 -> 32 ->16
    self.fc2 = nn.Linear(256, 101)
def forward(self, x):
    x = self.pool(self.relu(self.conv1(x))) # [64, 3, 128, 128]  -  [64, 3, 64, 64]
    x = self.pool(self.relu(self.conv2(x))) #[64, 3, 128, 128] - [64, 3, 32, 32]
    x = self.pool(self.relu(self.conv3(x))) #[64, 3, 128, 128] - [64, 3, 16, 16]
    x = x.view(x.size(0), -1) # batch size , image shape -[64*3*16* 16]
    x = self.relu(self.fc1(x))
    return self.fc2(x)
