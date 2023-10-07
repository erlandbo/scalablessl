import torch
import numpy as np
from torch import nn
import copy


class SCLFinetuner():
    def __init__(self, model, device, num_classes, lr):
        super().__init__()
        self.device = device
        # TODO better way?
        self.backbone = copy.deepcopy(model)
        self.backbone.fc = nn.Identity()

        self.classier = nn.Sequential(
            nn.LazyLinear(out_features=512),
            #nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.classier.parameters(),
            lr=lr,
        )
        self.CE = nn.CrossEntropyLoss()

    def backbone_forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(x)
        feats = feats.detach().clone()
        feats.requires_grad = True
        return feats

    def fit(self, trainloader, testloader, maxepochs=5):
        train_acc, test_acc = [], []
        for _ in range(maxepochs):
            train_epoch_acc = []
            self.classier.train()
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                feats = self.backbone_forward(x)
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
                for x, y in testloader:
                    x, y = x.to(self.device), y.to(self.device)
                    feats = self.backbone_forward(x)
                    logits = self.classier(feats)
                    acc = torch.mean( (torch.argmax(logits, dim=-1) == y ).float() )
                    test_epoch_acc.append(acc.detach().cpu().numpy())
                test_acc.append(np.mean(test_epoch_acc))

        return np.mean(train_acc), np.mean(test_acc)

