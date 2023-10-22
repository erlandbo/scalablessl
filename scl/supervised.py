import torch
import lightning as L
from ResNet import resnet18, resnet9, resnet34
from torchmodels import ResNettorch, ViTtorch
from ViT import SCLViT
from Schedulers import linear_warmup_cosine_anneal
from old_model import OldResNet
from torch import nn


class Supervised(L.LightningModule):
    def __init__(self, hparams, pretrained_model=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(hparams)
        if self.hparams.modelarch == "resnet9":
            self.model = resnet9(
                in_channels=self.hparams.in_channels,
                embed_dim=self.hparams.embed_dim,
                normlayer=self.hparams.normlayer,
                maxpool1=self.hparams.maxpool1,
                first_conv=self.hparams.first_conv
            )
        elif self.hparams.modelarch == "resnet18":
            self.model = resnet18(
                in_channels=self.hparams.in_channels,
                embed_dim=self.hparams.embed_dim,
                normlayer=self.hparams.normlayer,
                maxpool1=self.hparams.maxpool1,
                first_conv=self.hparams.first_conv
            )
        elif self.hparams.modelarch == "resnet34":
            self.model = resnet34(
                in_channels=self.hparams.in_channels,
                embed_dim=self.hparams.embed_dim,
                normlayer=self.hparams.normlayer,
                maxpool1=self.hparams.maxpool1,
                first_conv=self.hparams.first_conv
            )
        elif self.hparams.modelarch == "resnet18torch":
            self.model = ResNettorch(
                modelname="resnet18torch",
                in_channels=self.hparams.in_channels,
                embed_dim=self.hparams.embed_dim,
            )
        elif self.hparams.modelarch == "resnet34torch":
            self.model = ResNettorch(
                modelname="resnet34torch",
                in_channels=self.hparams.in_channels,
                embed_dim=self.hparams.embed_dim,
            )
        elif self.hparams.modelarch == "resnet50torch":
            self.model = ResNettorch(
                modelname="resnet50torch",
                in_channels=self.hparams.in_channels,
                embed_dim=self.hparams.embed_dim,
            )
        elif self.hparams.modelarch == "vit":
            self.model = SCLViT(
                out_dim=self.hparams.embed_dim,
                imgsize=self.hparams.imgsize,
                patch_dim=self.hparams.transformer_patchdim,
                num_layers=self.hparams.transformer_numlayers,
                d_model=self.hparams.transformer_dmodel,
                nhead=self.hparams.transformer_nhead,
                d_ff_ratio=self.hparams.transformer_dff_ration,
                dropout=self.hparams.transformer_dropout,
                activation=self.hparams.transformer_activation,
                in_channels=self.hparams.in_channels,
            )
        elif self.hparams.modelarch == "vittorch":
            self.model = ViTtorch(
                out_dim=self.hparams.embed_dim,
                imgsize=self.hparams.imgsize,
                patch_dim=self.hparams.transformer_patchdim,
                num_layers=self.hparams.transformer_numlayers,
                d_model=self.hparams.transformer_dmodel,
                nhead=self.hparams.transformer_nhead,
                d_ff_ratio=self.hparams.transformer_dff_ration,
                dropout=self.hparams.transformer_dropout,
                activation=self.hparams.transformer_activation,
                in_channels=self.hparams.in_channels,
            )
        else:
            self.model = OldResNet(
                embed_dim=self.hparams.embed_dim
            )
        if pretrained_model is not None:
            self.model = pretrained_model
            print("Using pretrained model")
        else:
            print("Training model from scratch")

        print(self.model)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        logits = self.model(x)
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

        # use scheduler
        if self.hparams.scheduler == "cosanneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = self.maxepochs,  # max iterations
                eta_min=0.0  # min lr
            )
            scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == "linwarmup_cosanneal":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=linear_warmup_cosine_anneal(
                        warmup_steps=10,
                        total_steps=self.hparams.maxepochs
                    )
                ),
                "interval": "epoch",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]

