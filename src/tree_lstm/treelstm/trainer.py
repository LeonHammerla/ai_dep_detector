import numpy as np
from tqdm import tqdm

import torch

from tree_lstm.treelstm import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        correct_predictions = 0
        losses = []
        targets = []
        outputs = []
        rang = len(dataset)
        # rang = 1000
        indices = torch.randperm(rang, dtype=torch.long, device='cpu')
        for idx in tqdm(range(rang), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, input, label = dataset[indices[idx]]
            input = input.to(self.device)
            target = label.float().to(self.device)
            output = self.model(tree, input).flatten()
            targets.append(target)
            outputs.append(output)
            if idx % self.args.batchsize == 0 and idx > 0:
                targets = torch.stack(targets).flatten()
                outputs = torch.stack(outputs).flatten()
                # print(targets.cpu().numpy())
                # print(outputs.detach().cpu().numpy())
                loss = self.criterion(outputs, targets)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                correct_predictions += torch.sum(outputs.round() == targets)
                targets = []
                outputs = []

        self.epoch += 1
        return np.mean(losses), correct_predictions.double() / rang

    # helper function for testing
    def test(self, dataset):
        predictions = []
        true_y = []
        correct_predictions = 0
        self.model.eval()
        rang = len(dataset)
        # rang = 100
        with torch.no_grad():
            losses = []
            targets = []
            outputs = []
            indices = torch.randperm(rang, dtype=torch.long, device='cpu')
            for idx in tqdm(range(rang), desc='Testing epoch  ' + str(self.epoch) + ''):
                tree, input, label = dataset[indices[idx]]
                input = input.to(self.device)
                target = label.float().to(self.device)
                output = self.model(tree, input).flatten()
                targets.append(target)
                outputs.append(output)
                if idx % self.args.batchsize == 0 and idx > 0:
                    targets = torch.stack(targets).flatten()
                    outputs = torch.stack(outputs).flatten()
                    loss = self.criterion(outputs, targets)
                    losses.append(loss.item())
                    correct_predictions += torch.sum(outputs.round() == targets)
                    predictions.extend(outputs.round())
                    true_y.extend(targets)
                    # print(targets.cpu().numpy())
                    # print(outputs.detach().cpu().numpy())
                    targets = []
                    outputs = []
        predictions = torch.stack(predictions).cpu()
        true_y = torch.stack(true_y).cpu()

        return np.mean(losses), correct_predictions.double() / rang, predictions, true_y