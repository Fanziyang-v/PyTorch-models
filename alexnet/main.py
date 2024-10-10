import os
import argparse
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class AlexNet(nn.Module):
    """AlexNet.
    
    Architecture: [conv - relu - max pool] x 2 - [conv - relu] x 3 - max pool - [affine - dropout] x 2 - affine - softmax
    """
    def __init__(self, num_channels: int, num_classes: int=10) -> None:
        """Initialize AlexNet.

        Args:
            num_channels(int): number of channels of input images.
            num_classes(int): number of classes.
        """
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(), # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2), # 27x27
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), nn.ReLU(), # 27x27
            nn.MaxPool2d(kernel_size=3, stride=2), # 13x13
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), nn.ReLU(), # 13x13
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.ReLU(), # 13x13
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(), # 13x13
            nn.MaxPool2d(kernel_size=3, stride=2), # 6x6
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096), nn.Dropout(),
            nn.Linear(4096, 4096), nn.Dropout(),
            nn.Linear(4096, num_classes))
    
    def forward(self, images: Tensor) -> Tensor:
        """Forward pass in AlexNet.
        
        Args:
            images(Tensor): input images of shape(N, num_channels, 224, 224)
        
        Returns:
            Tensor: scores matrix of shape (N, num_classes)
        """
        return self.model(images)


# Image processing.
transform_mnist = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

transform_cifar = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for SGD optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum term for SGD optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset(MNIST | FashionMNIST | CIFAR10)')
parser.add_argument('--interval', type=int, default=2, help='number of epochs between validating on test dataset')
parser.add_argument('--logdir', type=str, default='runs', help='directory for saving running log')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='directory for saving model checkpoint')
args = parser.parse_args()

writer = SummaryWriter(os.path.join(args.logdir, args.dataset))

# Create folder if not exists.
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
    test_data = datasets.CIFAR10(root='../data', train=False, transform=transform_cifar)
else:
    raise Exception(f'Unkown dataset: {args.dataset}. Support dataset: (MNIST | FashionMNIST | CIFAR-10)')

# data loader.
train_dataloader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

num_channels = 1 if args.dataset in ('MNIST', 'FashionMNIST') else 3

# AlexNet.
model = AlexNet(num_channels=num_channels).to(device)

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
        writer.add_scalar('Test Accuracy', test_acc, epoch + 1)

# Save model checkpoint.
torch.save(model.state_dict(), os.path.join(args.ckpt_dir, args.dataset, 'alexnet.ckpt'))
