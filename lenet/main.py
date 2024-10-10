import os
import argparse
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class LeNet(nn.Module):
    """LeNet.
    
    Architecutre: [conv - sigmoid - avg pool] x 2 - [affine - sigmoid] x 2 - affine - softmax
    """
    def __init__(self) -> None:
        """Initialize LeNet."""
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 100),
            nn.Sigmoid(),
            nn.Linear(100, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10))
    
    def forward(self, images: Tensor) -> Tensor:
        """Forward pass in LeNet.
        
        Args:
            images(Tensor): input images, of shape (N, 1, 28, 28)
        
        Returns scores matrix of shape (N, 10)
        """
        return self.model(images)


# Image processing.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

# Initialize network weights.
def init_weight(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(module.weight)

# device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parser commandline arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for SGD optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum term for SGD optimizer')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset(MNIST | FashionMNIST)')
parser.add_argument('--interval', type=int, default=10, help='number of epochs between validating on test dataset')
parser.add_argument('--logdir', type=str, default='runs', help='directory for saving running log')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
args = parser.parse_args()
print(args)

writer = SummaryWriter(os.path.join(args.logdir, args.dataset))

# Create folder if not exists
if not os.path.exists(os.path.join(args.ckpt_dir, args.dataset)):
    os.makedirs(os.path.join(args.ckpt_dir, args.dataset))

# MNIST dataset.
training_data = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='../data', train=False, transform=transform)

# Data loader.
train_dataloader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

# LeNet.
model = LeNet().to(device)
model.apply(init_weight)

# Loss and optimizer.
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Start training.
for epoch in range(args.num_epochs):
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
torch.save(model.state_dict(), os.path.join(args.ckpt_dir, args.dataset, 'lenet.ckpt'))
