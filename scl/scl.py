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


class SCL(L.LightningModule):
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
            self.model = OldResNet(
                embed_dim=self.hparams.embed_dim
            )
        print(self.model)
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1,))  # weighted sum q
        self.register_buffer("omega", torch.zeros(1,))  # count q
        self.register_buffer("N", torch.zeros(1,) + self.hparams.nsamples)  # N samples in dataset
        # sinv_init = self.hparams.nsamples**2 / 10**self.hparams.sinv_init_coeff   # s_init = 10^t * N^-2
        sinv_init = self.hparams.nsamples**self.hparams.ncoeff   # TODO find better init
        self.register_buffer("s_inv", torch.zeros(1,) + sinv_init)  # scale parameter measure discrepancy between p and q
        self.register_buffer("alpha", torch.zeros(1,) + self.hparams.alpha)  # [0,1] adaptively extra attraction to ease training
        self.register_buffer("ro", torch.zeros(1,) + self.hparams.ro)  # [0,1] forgetting rate s_inv
        self.T = self.hparams.titer
        self.tau = self.hparams.ncoeff  # s-coefficient N**t
        # Manually log initial value
        # self.log("qij_coeff", (self.N**self.tau) / self.s_inv,)

    def forward(self, x):
        return self.model(x)

    def _sim_metric(self, z1, z2):
        if self.hparams.simmetric == "stud-tkernel":
            return 1 / (1 + torch.sum((z1 - z2)**2, dim=1))
        elif self.hparams.simmetric == "gaussian":  # TODO make more numerical stable?
            return torch.exp( - torch.sum((z1 - z2)**2,dim=1) / (2 * self.hparams.var) ).clamp(min=1e-40)
            # return torch.exp( - torch.sum((z1 - z2)**2,dim=1).clamp(max=self.hparams.clamp, min=self.hparams.eps) / (2 * self.hparams.var) )
        elif self.hparams.simmetric == "cossim":
            z1 = z1.norm(p=2, dim=1, keepdim=True)
            z2 = z2.norm(p=2, dim=1, keepdim=True)
            return torch.sum(z1 * z2, dim=1) * 0.5 + 0.5  # cosine similarity [-1,1] -> [0,1]
        else:
            assert 1 == 2, "Invalid simmetric"  # TODO clean code

    def _shared_step(self, batch, batch_idx, mode="train"):
        # Reset
        self.xi = torch.zeros(1,).to(self.device)
        self.omega = torch.zeros(1,).to(self.device)
        # Process batch
        x_i, xhat_i, x_j = batch
        B = x_i.shape[0]
        x = torch.cat([x_i, xhat_i, x_j], dim=0)  # (3B,...)
        z = self.forward(x)
        z_i, zhat_i, z_j = z[0:B], z[B:2*B], z[2*B:3*B]
        # Positive forces
        qii = self._sim_metric(z_i, zhat_i)  # (B,1)
        positive_forces = torch.mean( - torch.log(qii) )
        self.xi = self.xi + torch.sum(self.alpha * qii).detach()
        self.omega = self.omega + self.alpha * B
        # Negative forces
        qij = self._sim_metric(z_i, z_j)  # (B,1)
        negative_forces = torch.mean( self.N**self.tau * qij / self.s_inv )
        self.xi = self.xi + torch.sum( (1 - self.alpha) * qij ).detach()
        self.omega = self.omega + (1 - self.alpha) * B
        # Update only in train-step

        loss = positive_forces + negative_forces

        self.log_dict({
            mode + "_loss": loss,
            mode + "_posforces": positive_forces,
            mode + "_negforces": negative_forces,
            mode + "_qii": torch.mean(qii).item(),
            mode + "_qij": torch.mean(qij).item(),
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        return {"loss": loss}

    def training_step(self, train_batch, batch_idx):
        self.log_dict({
            "xi": self.xi.item(),
            "omega": self.omega.item(),
            "alpha": self.alpha.item(),
            "ro": self.ro.item(),
            "sinv": self.s_inv.item(),
            "qij_coeff": (self.N**self.tau) / self.s_inv,
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        loss = self._shared_step(train_batch, batch_idx, mode="train")
        # print("coeff init", self.N**self.tau / self.s_inv)

        # Update
        if self.hparams.ro < 0.0:
            self.ro = self.N**self.tau / (self.N**self.tau + self.omega)
        self.s_inv = self.ro * self.s_inv + (1 - self.ro) * self.N**self.tau * self.xi / self.omega
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_step(val_batch, batch_idx, mode="val")
        if batch_idx % 44 == 0:
            self._show_images(val_batch)
            if self.hparams.modelarch == "vit":
                self._show_attention(val_batch)

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

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = matplotlib.cm.jet(np.linspace(0,1,self.hparams.numclasses))
            for i in range(self.hparams.numclasses):
                ax.scatter(X_train[y_train == i ,0], X_train[y_train==i, 1], label=label_names[i], color=colors[i])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="medium")
            #ax.set_title('Train')
            ax.set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure("train_scatter1", fig, global_step=self.global_step)

            fig.savefig("plots/{}_{}_epoch_{}_train_1_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset,self.current_epoch), format="pdf")

            #fig.savefig("plots/plot11.pdf", format="pdf")

            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(self.hparams.numclasses):
                ax.scatter(X_test[y_test==i,0], X_test[y_test==i, 1], label=label_names[i], color=colors[i])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="medium")
            #ax.set_title('Test')
            ax.set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure("test_scatter1", fig, global_step=self.global_step)
            fig.savefig("plots/{}_{}_epoch_{}_test_1_plot.pdf".format(self.hparams.experiment_name,self.hparams.dataset, self.current_epoch), format="pdf")

            fig, ax = plt.subplots(figsize=(20, 12))
            colors = matplotlib.cm.jet(np.linspace(0,1,self.hparams.numclasses))
            for i in range(self.hparams.numclasses):
                ax.scatter(X_train[y_train == i ,0], X_train[y_train==i, 1], label=label_names[i], color=colors[i])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="20")
            #ax.set_title('Train')
            ax.set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure("train_scatter2", fig, global_step=self.global_step)

            fig.savefig("plots/{}_{}_epoch_{}_train_2_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset,self.current_epoch), format="pdf")
            #fig.savefig("plots/{}_plot12.pdf".format(self.hparams.experiment_name), format="pdf")

            fig, ax = plt.subplots(figsize=(20, 12))
            for i in range(self.hparams.numclasses):
                ax.scatter(X_test[y_test==i,0], X_test[y_test==i, 1], label=label_names[i], color=colors[i])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="20")
            #ax.set_title('Test')
            ax.set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure("test_scatter2", fig, global_step=self.global_step)
            fig.savefig("plots/{}_{}_epoch_{}_test_2_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset,self.current_epoch), format="pdf")

            fig, ax = plt.subplots(figsize=(30, 24))
            colors = matplotlib.cm.jet(np.linspace(0,1,self.hparams.numclasses))
            for i in range(self.hparams.numclasses):
                ax.scatter(X_train[y_train == i ,0], X_train[y_train==i, 1], label=label_names[i], color=colors[i])
            #ax.set_title('Train')
            ax.set_axis_off()
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="25")
            fig.tight_layout()
            self.logger.experiment.add_figure("train_scatter3", fig, global_step=self.global_step)

            fig.savefig("plots/{}_{}_epoch_{}_train_3_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset, self.current_epoch), format="pdf")
            #fig.savefig("plots/plot13.pdf", format="pdf")

            fig, ax = plt.subplots(figsize=(30, 24))
            for i in range(self.hparams.numclasses):
                ax.scatter(X_test[y_test==i,0], X_test[y_test==i, 1], label=label_names[i], color=colors[i])
            #ax.set_title('Test')
            ax.set_axis_off()
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="25")
            fig.tight_layout()
            self.logger.experiment.add_figure("test_scatter3", fig, global_step=self.global_step)

            fig.savefig("plots/{}_{}_epoch_{}_test_3_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset,self.current_epoch), format="pdf")

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

            fig, ax = plt.subplots(1,2, figsize=(10, 6))
            colors = matplotlib.cm.jet(np.linspace(0,1,self.hparams.numclasses))
            for i in range(self.hparams.numclasses):
                ax[0].scatter(X_train[y_train == i ,0], X_train[y_train==i, 1], label=label_names[i], color=colors[i])
                ax[1].scatter(X_test[y_test==i,0], X_test[y_test==i, 1], label=label_names[i], color=colors[i])
            ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="medium")
            #ax[0].set_title('Train')
            ax[0].set_axis_off()
            #ax[1].set_title('Test')
            ax[1].set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure("scatter2", fig, global_step=self.global_step)

            fig.savefig("plots/{}_{}_epoch_{}_scatter_2_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset,self.current_epoch), format="pdf")

            fig, ax = plt.subplots(1,2, figsize=(30, 20))
            colors = matplotlib.cm.jet(np.linspace(0,1,self.hparams.numclasses))
            for i in range(self.hparams.numclasses):
                ax[0].scatter(X_train[y_train == i ,0], X_train[y_train==i, 1], label=label_names[i], color=colors[i])
                ax[1].scatter(X_test[y_test==i,0], X_test[y_test==i, 1], label=label_names[i], color=colors[i])
            ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="25")
            #ax[0].set_title('Train')
            ax[0].set_axis_off()
            #ax[1].set_title('Test')
            ax[1].set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure("scatter3", fig, global_step=self.global_step)

            fig.savefig("plots/{}_{}_epoch_{}_scatter_3_plot.pdf".format(self.hparams.experiment_name, self.hparams.dataset,self.current_epoch), format="pdf")
            #fig, ax = plt.subplots(figsize=(10, 6))
            #for i in range(self.hparams.numclasses):
            #    ax.scatter(X_test[y_test==i,0], X_test[y_test==i, 1], label=label_names[i], color=colors[i])
            #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="medium")
            #fig.tight_layout()
            #self.logger.experiment.add_figure("test_scatter", fig, global_step=self.global_step)

    # Plotting
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

    def _show_attention(self, batch, numimgs=2):
        assert self.hparams.modelarch in ["vit"], "Must use custom-vit for plotting attention heatmap"
        mu, sigma = get_image_stats(self.hparams.dataset)
        mu = torch.tensor(mu)[:, None, None]
        sigma = torch.tensor(sigma)[:, None, None]
        B = self.hparams.batchsize
        x = torch.cat(batch, dim=0).detach().cpu() * sigma + mu
        attn = self.model.m.encoder[-1].attn.weights.detach().cpu()[:,:,0,1:] # (N,h,L,S) -> (N,h,S-1) drop attn(cls,cls)
        attn_size = int(attn.shape[-1] ** 0.5)
        nheads = attn.shape[1]
        B = self.hparams.batchsize
        # attn = torch.mean(attn, dim=1)  # (N,h,S) -> (N,S) -> (N,1,S) -> (N,1,H,W)
        attn = F.interpolate(attn.reshape(-1, nheads, attn_size, attn_size), size=self.hparams.imgsize)  # (N,h,H,W)
        B = self.hparams.batchsize
        attn_i, attn_hat_i, attn_j = attn[0:B], attn[B:2*B], attn[2*B:3*B]
        x_i, xhat_i, x_j = x[0:B], x[B:2*B], x[2*B:3*B]
        attn = list(zip(attn_i, attn_hat_i, attn_j))
        images = list(zip(x_i, xhat_i, x_j))
        images, attn = images[:numimgs], attn[:numimgs]

        fig, ax = plt.subplots(numimgs*3, 1 + nheads + 1, figsize=(25, 25))
        for k in range(numimgs):
            for m in range(3):
                x_m = images[k][m]
                attn_m = attn[k][m]  # (h,H',W')
                ax[k*3 + m, 0].imshow(x_m.permute(1, 2, 0).numpy())
                for l in range(nheads):
                    ax[k*3 + m, 1 + l].imshow(attn_m[l].unsqueeze(0).permute(1, 2, 0).numpy(), cmap="inferno")
                ax[k*3 + m, 1 + nheads].imshow(torch.mean(attn_m.unsqueeze(0),dim=1).permute(1, 2, 0).numpy(), cmap="inferno")
                ax[k*3 + m, 1 + nheads].set_title("mean")

        fname = "plots/attn_{}.png".format(self.hparams.dataset)
        plt.savefig(fname)
        plt.close()
