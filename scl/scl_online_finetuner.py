import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SCLLinearFinetuner():
    def __init__(self, device, in_features, num_classes, lr, hdim=512, activation="relu"):
        super().__init__()
        self.device = device
        self.classier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hdim),
            nn.ReLU() if activation=="relu" else nn.GELU(),
            nn.Linear(in_features=hdim, out_features=num_classes)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.classier.parameters(),
            lr=lr,
        )
        self.CE = nn.CrossEntropyLoss()

    def fit(self, X_train, y_train, X_test, y_test, maxepochs=5, batchsize=256):
        trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batchsize, shuffle=True)
        testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batchsize, shuffle=False)
        train_acc, test_acc = [], []
        for _ in range(maxepochs):
            train_epoch_acc = []
            self.classier.train()
            for feats, y in trainloader:
                feats, y = feats.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.classier(feats)
                loss = self.CE(logits, y)
                loss.backward()
                self.optimizer.step()
                acc = torch.mean( (torch.argmax(logits, dim=-1) == y ).float() )
                train_epoch_acc.append(acc.detach().cpu().numpy())
                train_acc.append(np.mean(train_epoch_acc))
            test_epoch_acc = []
            self.classier.eval()
            with torch.no_grad():
                for feats, y in testloader:
                    feats, y = feats.to(self.device), y.to(self.device)
                    logits = self.classier(feats)
                    acc = torch.mean( (torch.argmax(logits, dim=-1) == y ).float() )
                    test_epoch_acc.append(acc.detach().cpu().numpy())
                test_acc.append(np.mean(test_epoch_acc))

        return np.mean(train_acc), np.mean(test_acc)


