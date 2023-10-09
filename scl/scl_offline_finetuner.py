from typing import Any, Optional

import torch
import lightning as L
import numpy as np
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT

from utils import get_image_stats
from torch import nn


class SCLOfflineFinetuner(L.LightningModule):
    def __init__(self, backbone_module, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone_module
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.eval()
        in_features = self.backbone.q.in_featurees

        self.classier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=self.hparams.hdim),
            nn.ReLU() if self.hparams.hdim=="relu" else nn.GELU(),
            nn.Linear(in_features=self.hparams.hdim, out_features=self.hparams.num_classes)
        ).to(self.device)

        self.CE = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classier(x)

    def _shared_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        self.backbone.eval()
        with torch.no_grad():
            embeds = self.backbone(x).detach()
        logits = self.classier(embeds)
        loss = self.CE(logits, y)
        acc = torch.mean( (torch.argmax(logits, dim=-1) == y ).float() )

        self.log_dict({
            mode + "_loss": loss,
            mode + "_acc": acc,
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        return {"loss": loss}

    def training_step(self, train_batch, batch_idx):
        return self._shared_step(train_batch, batch_idx, mode="train")

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_step(val_batch, batch_idx, mode="val")
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self._shared_step(test_batch, batch_idx, mode="test")
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=0.8
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
            )



