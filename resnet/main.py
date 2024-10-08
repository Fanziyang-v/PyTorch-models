import os
import argparse
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class ResNet(nn.Module):
    """ResNet for CIFAR-10."""
    def __init__(self, num_channels: int, num_classes: int, layers: list[int]=[3, 3, 3]) -> None:
        """Initialize ResNet.

        Args:
            layers(list): specify number of residual blocks in each layer. Default is [3, 3, 3].
        """
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(num_channels, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.__make_layer(16, layers[0])
        self.layer2 = self.__make_layer(32, layers[1], 2)
        self.layer3 = self.__make_layer(64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
    
    def __make_layer(self, out_channels: int, blocks: int, stride=1) -> nn.Module:
        """Construct ResNet layer.

        Args:
            out_channels(int): number of output channels in this layer.
            blocks(int): number of residual block in this layer.
            stride(int): stride of convolution.

        Returns:
            Module: ResNet layer.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            images(Tensor): input images of shape (N, C, 32, 32)
        
        Returns:
            Tensor: scores matrix of shape (N, D)
        """
        out: Tensor = self.conv(images)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample=None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


# 3x3 convolution.
def conv3x3(in_channels: int, out_channels: int, stride: int=1) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1)

# Image processing.
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parser commandline arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for SGD optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum term for SGD optimizer')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
parser.add_argument('--num_epochs', type=int, default=80, help='number of training epochs')
parser.add_argument('--interval', type=int, default=10, help='number of epochs between validating on test dataset')
parser.add_argument('--logdir', type=str, default='runs', help='directory for saving running log')
args = parser.parse_args()
print(args)

writer = SummaryWriter(args.logdir)

# CIFAR-10 dataset.
training_data = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root='../data', train=False, transform=transform)

# Data loader.
train_dataloader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

# ResNet.
model = ResNet(num_channels=3, num_classes=10, layers=[3, 3, 3]).to(device)

# Loss and optimizer.
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Start training.
for epoch in range(args.num_epochs):
    model.train()
    total_loss = total_accuracy = 0
    for images, labels in train_dataloader:
        images: Tensor = images.to(device)
        labels: Tensor = labels.to(device)

        # Forward pass
        outputs: Tensor = model(images)
        loss: Tensor = criterion(outputs, labels)
        ypred = torch.argmax(outputs, dim=1)
        total_accuracy += torch.mean((ypred == labels).float())

        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    avg_acc = total_accuracy / len(train_dataloader)
    # log average loss and accuracy.
    print(f'''
============================
Epoch: [{epoch + 1}/{args.num_epochs}]
Loss: {avg_loss:.4f}
Accuracy: {100 * avg_acc:.4f} %
============================
''')
    writer.add_scalar('Loss', avg_loss, epoch + 1)
    writer.add_scalar('Accuracy', avg_acc, epoch + 1)

    if (epoch + 1) % args.interval == 0:
        model.eval()
        num_correct = total = 0
        for images, labels in test_dataloader:
            images: Tensor = images.to(device)
            labels: Tensor = labels.to(device)

            outputs: Tensor = model(images)
            ypred = torch.argmax(outputs, dim=1)
            num_correct += torch.sum((ypred == labels).float())
            total += len(images)
        test_acc = num_correct / total
        print(f'''
============================
Testset Accuracy: {100 * test_acc:.4f} %
============================
''')
        
# Save model checkpoint.
torch.save(model.state_dict(), 'resnet.ckpt')
