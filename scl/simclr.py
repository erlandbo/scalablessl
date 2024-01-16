import torch
import lightning as L
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from torch.utils.data import DataLoader
from ResNet import resnet18, resnet9, resnet34
from torchmodels import ResNettorch, ViTtorch
import torchvision
from utils import get_image_stats, classlabels2name
from ViT import SCLViT
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scl_online_finetuner import SCLLinearFinetuner
from Schedulers import linear_warmup_cosine_anneal
import matplotlib
from torch import nn


class SimCLR(L.LightningModule):
    def __init__(self, hparams):
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
            raise ValueError()
        print(self.model)

        metric = self.hparams.simmetric
        if metric == "cosine":
            self.loss = InfoNCECosine()
        elif metric == "euclidean":  # actually Cauchy
            self.loss = InfoNCECauchy()
        elif metric == "gauss":
            self.loss = InfoNCEGaussian()
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")

        self.tau = self.hparams.tau

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, mode="train"):
        # Reset
        x_i, xhat_i, x_j = batch
        B = x_i.shape[0]
        x = torch.cat([x_i, xhat_i], dim=0)  # (2B,...)
        z = self.forward(x)
        loss = self.loss(z)

        self.log_dict({
            mode + "_loss": loss,
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        return {"loss": loss}

    def training_step(self, train_batch, batch_idx):
        loss = self._shared_step(train_batch, batch_idx, mode="train")
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_step(val_batch, batch_idx, mode="val")
        if batch_idx % 44 == 0:
            self._show_images(val_batch)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum
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
                T_max = self.hparams.lr_total_steps,  # max iterations
                eta_min=self.hparams.min_lr  # min lr
            )
            scheduler = {
                "scheduler": scheduler,
                "interval": self.hparams.lr_scheduler_interval,
                "frequency": 1
            }
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == "linwarmup_cosanneal":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=linear_warmup_cosine_anneal(
                        warmup_steps=self.hparams.lr_warmup_steps,
                        total_steps=self.hparams.lr_total_steps,
                        min_lr=self.hparams.min_lr
                    )
                ),
                "interval": self.hparams.lr_scheduler_interval,
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]


    # TODO Better ways to set finetune-dataset?
    def load_finetune_dataset(self, traindataset, testdataset):
        self.finetune_trainloader = DataLoader(dataset=traindataset, batch_size=self.hparams.finetune_batchsize, shuffle=True, num_workers=0)
        self.finetune_testloader = DataLoader(dataset=testdataset, batch_size=self.hparams.finetune_batchsize, shuffle=False, num_workers=0)

    # TODO suitable placement?
    # knn-finetune in eval-mode, no gradients needed
    def on_validation_epoch_end(self):
        if self.hparams.finetune_knn and self.current_epoch % self.hparams.finetune_interval == 0:
            self.knn_finetune()
        # TODO add t-SNE on d>2
        if self.hparams.plot2d_interval > 0 and self.current_epoch % self.hparams.plot2d_interval == 0 and self.hparams.embed_dim == 2:
            self.visualize_embeds()

    # TODO suitable placement?
    # linear-finetune in train-mode, gradients needed
    def on_train_epoch_end(self):
        if self.hparams.finetune_linear and self.current_epoch % self.hparams.finetune_interval == 0:
            self.linear_finetune()

    def linear_finetune(self):
        self.model.eval()
        with torch.no_grad():
            X_train, X_test, y_train, y_test = [], [], [], []
            in_features = 0
            for X, y in self.finetune_trainloader:
                embeds = self.model.m(X.to(self.device)).clone().detach()  # NOTE: extract backbone features
                in_features = embeds.shape[-1]
                X_train.append(embeds)
                y_train.append(y.detach())
            for X, y in self.finetune_testloader:
                embeds = self.model.m(X.to(self.device)).clone().detach()
                X_test.append(embeds)
                y_test.append(y.detach())

            X_train, y_train = torch.cat(X_train), torch.cat(y_train)
            X_test, y_test = torch.cat(X_test), torch.cat(y_test)
        finetuner = SCLLinearFinetuner(in_features=in_features, lr=self.hparams.finetune_lr, num_classes=self.hparams.numclasses, device=self.device)
        train_acc, test_acc = finetuner.fit(X_train, y_train, X_test, y_test, batchsize=self.hparams.finetune_batchsize)
        self.log("linear_finetune_train_acc", train_acc, on_step=False, on_epoch=True)
        self.log("linear_finetune_test_acc", test_acc, on_step=False, on_epoch=True)

    def knn_finetune(self):
        with torch.no_grad():
            knn = KNeighborsClassifier(n_neighbors=self.hparams.finetune_n_neighbours)
            X_train, X_test, y_train, y_test = [], [], [], []
            for X, y in self.finetune_trainloader:
                X_train.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_train.append(y.detach().numpy())
            for X, y in self.finetune_testloader:
                X_test.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_test.append(y.detach().numpy())
            X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
            X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)
            # import pdb
            # pdb.set_trace()
            knn.fit(X_train, y_train)
            y_hat = knn.predict(X_test)
            acc = np.mean(y_hat == y_test)
            self.log("knn_test_acc", acc, on_step=False, on_epoch=True)
            train_acc = np.mean(knn.predict(X_train) == y_train)
            self.log("knn_train_acc", train_acc, on_step=False, on_epoch=True)

    def visualize_embeds(self):
        # TODO add t-SNE and fix scatterplot cluster-color
        assert self.hparams.embed_dim == 2, "Can not scatterplot embeddim > 2"
        with torch.no_grad():
            X_train, X_test, y_train, y_test = [], [], [], []
            for X, y in self.finetune_trainloader:
                X_train.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_train.append(y.detach().numpy())
            for X, y in self.finetune_testloader:
                X_test.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_test.append(y.detach().numpy())
            X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
            X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

            label_names = classlabels2name(self.hparams.dataset)

            ###
            fig, ax = plt.subplots(1,2, figsize=(20, 10))
            colors = matplotlib.cm.jet(np.linspace(0,1,self.hparams.numclasses))
            for i in range(self.hparams.numclasses):
                ax[0].scatter(X_train[y_train == i ,0], X_train[y_train==i, 1], label=label_names[i], color=colors[i])
                ax[1].scatter(X_test[y_test==i,0], X_test[y_test==i, 1], label=label_names[i], color=colors[i])
            ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="20")
            #ax[0].set_title('Train')
            ax[0].set_axis_off()
            #ax[1].set_title('Test')
            ax[1].set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure("scatter1", fig, global_step=self.global_step)
            fig.savefig("plots/{}_{}_epoch_{}_scatter_1_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset,self.current_epoch), format="pdf")

    def _show_images(self, batch):
        mu, sigma = get_image_stats(self.hparams.dataset)
        mu = torch.tensor(mu, device=self.device)[:, None, None]
        sigma = torch.tensor(sigma, device=self.device)[:, None, None]
        num_imgs = 4
        x_i, xhat_i, x_j = batch
        x_i, xhat_i, x_j = x_i[:num_imgs], xhat_i[:num_imgs], x_j[:num_imgs]
        x_ = zip(x_i, xhat_i, x_j)
        x_ = [xk * sigma + mu for xl in x_ for xk in xl]
        grid = torchvision.utils.make_grid(torch.stack(x_)[:12], nrow=3)
        self.logger.experiment.add_image(self.hparams.dataset, grid, self.global_step)


class InfoNCECosine(nn.Module):
    def __init__(
            self,
            temperature: float = 0.5,
            reg_coef: float = 0,
            reg_radius: float = 200,
    ):
        super().__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # mean deviation from the sphere with radius `reg_radius`
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)

        a = F.normalize(a)
        b = F.normalize(b)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss


class InfoNCECauchy(nn.Module):
    def __init__(self, temperature: float = 1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        tempered_alignment = torch.diagonal_copy(sim_ab).log_().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCEGaussian(InfoNCECauchy):
    def forward(self, features):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss
