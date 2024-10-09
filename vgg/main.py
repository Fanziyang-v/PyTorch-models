import os
import argparse
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class VGG(nn.Module):
    """VGG-16 for CIFAR-10"""
    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize VGG Net.
        
        Args:
            num_channels(int): number of channels.
            num_classes(int): number of classes.
        """
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            *self.__block(num_channels, 64, num_convs=2),
            *self.__block(64, 128, num_convs=2),
            *self.__block(128, 256, num_convs=3),
            *self.__block(256, 512, num_convs=3),
            *self.__block(512, 512, num_convs=3),
            nn.Flatten(),
            nn.Linear(512, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, num_classes))

    
    def forward(self, images: Tensor) -> Tensor:
        """Forward pass in VGG Net.
        
        Args:
            images(Tensor): input images of shape (N, num_channels, 32, 32)
        
        Returns:
            Tensor: scores matrix of shape (N, num_classes).
        """
        return self.model(images)


    def __conv3x3(self, in_channels: int, out_channels: int) -> list[nn.Module]:
        """Construct a block with architecture: conv - batchnorm - relu
        
        Args:
            in_channels(int): number of channels of input feature map.
            out_channels(int): number of channels of output feature map.
        
        Returns:
            list: a list of modules.
        """
        return [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU()]

    def __block(self, in_channels: int, out_channels: int, num_convs: int) -> list[nn.Module]:
        """Construct a block which has architecture: [conv - batchnorm - relu] x num_convs - max pool.
        
        Args:
            in_channels(int): number of channels of input feature map.
            out_channels(int): number of channels of output feature map.
            num_convs(int): number of convolutional layers.
        
        Returns:
            list: a list of modules.
        """
        modules: list[nn.Module] = self.__conv3x3(in_channels, out_channels)
        for _ in range(1, num_convs - 1):
            modules += self.__conv3x3(out_channels, out_channels)
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return modules


# Image processing.
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parser commandline arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for SGD optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum term for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='L2 weight decay term')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--num_epochs', type=int, default=60, help='number of training epochs')
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

# VGG Net.
model = VGG(num_channels=3, num_classes=10).to(device)

# Loss and optimizer.
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
    
    scheduler.step(avg_loss)
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
torch.save(model.state_dict(), 'vgg.ckpt')
