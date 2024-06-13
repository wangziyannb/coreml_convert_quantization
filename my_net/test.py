import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Subset, DataLoader

from model import MyNet

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

    # Create the model and load the weights
    model = MyNet()
    model.load_state_dict(torch.load('my_network.pth', map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load('quantized_model.pth', map_location=torch.device('cpu')))
    # Set the model to evaluation mode
    model.eval()
    torch.backends.quantized.engine = 'qnnpack'

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')


    def fuse_model(model):

        modules_to_fuse = [
            ['conv_layers.0', 'conv_layers.1', 'conv_layers.2'],
            ['conv_layers.4', 'conv_layers.5', 'conv_layers.6'],
            ['conv_layers.8', 'conv_layers.9', 'conv_layers.10'],
            ['fc_layers.1', 'fc_layers.2'],
            ['fc_layers.3', 'fc_layers.4']
        ]


        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)


    fuse_model(model)
    model_prepared = torch.quantization.prepare(model)


    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                outputs = model(images)


    # calibration_loader = [(inference_pipline.total_mel_batch, inference_pipline.total_img_batch)]
    # calibration_loader = []
    subset_indices = list(range(10))

    subset = Subset(testset, subset_indices)
    partial_dataloader = DataLoader(subset, batch_size=10, shuffle=False)
    # calibration_loader.append(testloader)
    # calibration_loader = [(inference_pipline.total_mel_batch, inference_pipline.total_img_batch)]
    calibrate(model_prepared, partial_dataloader)

    model_quantized = torch.quantization.convert(model_prepared)

    example_input = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model_quantized, example_input)
    torch.jit.save(traced_model, "quantized_model.pth")

    # torch.save(model_quantized.state_dict(), 'quantized_model.pth')

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

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model_quantized(images)
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100. * correct / total

    # Print the accuracy
    print('Test accuracy = %6.2f %%' % accuracy)

    print(f"Prediction class: {predicted}")