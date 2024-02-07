import argparse
import warnings

import torch
import torchvision
from torch import nn

from model import Net
import torchvision.models as models
from train import Trainer

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = Net()

    resnet_model = models.resnet50(pretrained=True)
    print(resnet_model)
    # Freeze all layers except the last one
    for param in resnet_model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer with a new one (assuming you're fine-tuning for classification)
    num_classes = 10  # Assuming you have 10 classes
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

    # Modify the first convolutional layer to accept single-channel input
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)
    trainer_resnet = Trainer(model=resnet_model)

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="save/")
    trainer_resnet.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="save/")

    # model evaluation
    trainer.eval(test_loader=test_loader)
    trainer_resnet.eval(test_loader=test_loader)

    # model inference
    # Get a sample image and label from the test dataset
    sample_images, sample_labels = next(iter(test_loader))
    sample_image = sample_images[0]


    # Perform inference with your model
    predicted_class = trainer.infer(sample_image)

    print("Predicted class:", predicted_class)
    print("True class:", sample_labels[0].item())

    return


if __name__ == "__main__":
    main()
