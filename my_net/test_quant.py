import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

if __name__ == '__main__':

    # Parameters
    batch_size = 10

    # Test set
    testset = torchvision.datasets.CIFAR10(root='datasets',
                                           train=False,
                                           download=True,
                                           transform=transforms.ToTensor())

    # Test set loader
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=1)
    torch.backends.quantized.engine = 'qnnpack'

    # Create the model and load the weights
    model = torch.jit.load("quantized_model.pth")

    model.eval()

    # Compute test set accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100. * correct / total

    # Print the accuracy
    print('Test accuracy = %6.2f %%' % accuracy)

    print(f"Prediction class: {predicted}")