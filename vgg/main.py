import os
import argparse
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class VGG(nn.Module):
    """VGG Net with 16 weight layers."""
    def __init__(self, num_channels: int, num_classes: int=10) -> None:
        """Initialize VGG Net.
        
        Args:
            num_channels(int): number of channels.
            num_classes(int): number of classes.
        """
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            VGGBlock(num_channels, 64, num_convs=2),
            VGGBlock(64, 128, num_convs=2),
            VGGBlock(128, 256, num_convs=3),
            VGGBlock(256, 512, num_convs=3),
            VGGBlock(512, 512, num_convs=3),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(),
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


class VGGBlock(nn.Module):
    """VGG Block for VGG Net.
    
    Architecture: [conv - relu] x n
    """
    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        """Initialize VGG Block.
        
        Args:
            in_channels(int): number of channels of input feature map.
            out_channels(int): number of channels of output feature map.
        """
        super(VGGBlock, self).__init__()
        modules: list[nn.Module] = conv3x3(in_channels, out_channels)
        for _ in range(1, num_convs - 1):
            modules += conv3x3(out_channels, out_channels)
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.model = nn.Sequential(*modules)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass in VGG Block.
        
        Args:
            x(Tensor): input feature map of shape (N, C, H, W).
        
        Returns:
            Tensor: output feature map of shape (N, C', H // 2, W // 2)
        """
        return self.model(x)


def conv3x3(in_channels: int, out_channels: int) -> list[nn.Module]:
    """Construct a block with architecture: conv - relu
    
    Args:
        in_channels(int): number of channels of input feature map.
        out_channels(int): number of channels of output feature map.
    
    Returns:
        list: a list of modules.
    """
    return [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU()]


# Image processing.
transform_cifar = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

transform_mnist = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

# device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parser commandline arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for SGD optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum term for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='L2 weight decay term')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--num_epochs', type=int, default=60, help='number of training epochs')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset(MNIST | FashionMNIST | CIFAR10)')
parser.add_argument('--interval', type=int, default=10, help='number of epochs between validating on test dataset')
parser.add_argument('--logdir', type=str, default='runs', help='directory for saving running log')
args = parser.parse_args()
print(args)

writer = SummaryWriter(args.logdir)

# Create folder if not exists
if not os.path.exists(os.path.join(args.ckpt_dir, args.dataset)):
    os.makedirs(os.path.join(args.ckpt_dir, args.dataset))

# Dataset.
if args.dataset == 'MNIST':
    training_data = datasets.MNIST(root='../data', train=True, transform=transform_mnist, download=True)
    test_data = datasets.MNIST(root='../data', train=False, transform=transform_mnist)
elif args.dataset == 'FashionMNIST':
    training_data = datasets.FashionMNIST(root='../data', train=True, transform=transform_mnist, download=True)
    test_data = datasets.FashionMNIST(root='../data', train=False, transform=transform_mnist)
elif args.dataset == 'CIFAR10':
    training_data = datasets.CIFAR10(root='../data', train=True, transform=transform_cifar, download=True)
    test_data = datasets.CIFAR10(root='../data', train=True, transform=transform_cifar)
else:
    raise Exception(f'Unkown dataset: {args.dataset}. Support dataset: (MNIST | FashionMNIST | CIFAR10)')

# Data loader.
train_dataloader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

num_channels = 3 if args.dataset == 'CIFAR10' else 1

# VGG Net.
model = VGG(num_channels=num_channels).to(device)

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
        writer.add_scalar('Test Accuracy', test_acc, epoch + 1)

# Save model checkpoint.
torch.save(model.state_dict(), os.path.join(args.ckpt_dir, args.dataset, 'vgg.ckpt'))
