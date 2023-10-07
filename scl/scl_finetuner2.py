import torch
import lightning as L
from torch import nn
import copy


class SCLFinetuner(L.LightningModule):
    def __init__(self, backbone, num_classes, lr):
        super().__init__()
        self.lr = lr
        self.classier = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        # TODO find better way
        self.backbone = copy.deepcopy(backbone)
        self.backbone.fc = nn.Identity()
        self.CE = nn.CrossEntropyLoss()

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(x)
        logits = self.classier(feats)
        return logits

    def _shared_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        logits = self.forward(x)
        loss = self.CE(logits, y)
        acc = torch.mean( (torch.argmax(logits, dim=-1) == y ).float())
        self.log_dict({
            "finetune_" + mode + "_loss": loss,
            "finetune_" + mode + "_acc": acc
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        return {"loss": loss}

    def training_step(self, train_batch, batch_idx):
        return self._shared_step(train_batch, batch_idx, mode="train")

    def validation_step(self, val_batch, batch_idx):
        return self._shared_step(val_batch, batch_idx, mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.classier.parameters(),
            lr=self.lr,
        )
        return [optimizer]
