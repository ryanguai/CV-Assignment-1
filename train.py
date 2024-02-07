import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter


class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        return

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """
        
        self._model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self._model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print('Accuracy on the test set: %d %%' % (100 * accuracy))
        return

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        if self._model is None:
            raise ValueError("Model has not been loaded yet. Please load a model first using load_model().")

        # Set the model to evaluation mode
        self._model.eval()

        # Perform inference
        with torch.no_grad():
            output = self._model(sample)  
            _, predicted = torch.max(output, 1)  # Get the index of the class with the highest probability
            return predicted.item()  # Return the predicted class index as an integer


    def load_model(self, path: str) -> None:
       # Load the model
        self._model = torch.load(path)

