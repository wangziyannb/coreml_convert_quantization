import torch
import torchvision
from model import MyNet
from tqdm import tqdm

import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Parameters
batch_size = 10
nb_epochs = 20
learning_rate = 0.001
momentum = 0.9

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training set
trainset = torchvision.datasets.CIFAR10(root='datasets',
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())

# Test set
testset = torchvision.datasets.CIFAR10(root='datasets',
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())

# Training set loader
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

# Test set loader
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

# Access datasets properties
train_shape = trainset.data.shape
test_shape = testset.data.shape
train_nb = train_shape[0]
test_nb = test_shape[0]
height = train_shape[1]
width = train_shape[2]
classes = trainset.classes
print('Training set size : %d' % train_nb)
print('Test set size     : %d' % test_nb)
print('Image size        : %d x %d\n' % (height, width))

# Build the network
model = MyNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training
print('Training')
for epoch in range(nb_epochs):

    # Set the model to training mode
    model.train()

    # Running loss container
    running_loss = 0.0

    # Create a progress bar
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader))

    # Iterate through mini-batches
    for i, data in progress_bar:

        # Get the mini-batch data
        images, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Running loss update
        running_loss += loss.item()

        # Print the running loss every 1000 mini-batches
        if i % 1000 == 999:
            progress_bar.set_description(f'Epoch: {epoch}, Loss: {running_loss / 1000:.3f}')
            running_loss = 0.0

    # Set the model to evaluation mode
    model.eval()

    # Compute training set accuracy
    training_correct = 0
    training_total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            training_correct += (predicted == labels).sum().item()
            training_total += labels.size(0)
    training_accuracy = 100. * training_correct / training_total

    # Compute test set accuracy
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    test_accuracy = 100. * test_correct / test_total

    # Print the accuracies
    print('Epoch : %2d, training accuracy = %6.2f %%, test accuracy = %6.2f %%' % (epoch, training_accuracy, test_accuracy) )

# Save the network weights
torch.save(model.state_dict(), 'models/my_network.pth')