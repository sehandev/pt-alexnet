import argparse
import json
import os
from os import path
from tqdm.auto import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
import wandb

from models.alexnet import AlexNet

DATASETS_PATH = path.join(path.dirname(__file__), "data")


class Lite(LightningLite):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataloader_pbar = None
        self.log_dict = dict()

    def init_wandb(self, cfg) -> None:
        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
        wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.wandb_name}-{cfg.now}",
            config={
                "seed": cfg.seed,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
            },
        )

    def set_dataloader_pbar(
        self,
        dataloader: DataLoader,
        desc: str,
        disable: bool = True,
        leave: bool = False,
    ) -> None:
        self.dataloader_pbar = tqdm(dataloader, leave=leave, disable=disable, position=0)
        self.dataloader_pbar.set_description(desc)
        self.dataloader_pbar.set_postfix(self.log_dict)

    def run(self, cfg: DictConfig) -> None:
        if self.is_global_zero:
            print(OmegaConf.to_yaml(cfg))
            self.init_wandb(cfg)

        seed_everything(cfg.seed)

        # Dataset
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if self.is_global_zero:
            FashionMNIST(DATASETS_PATH, download=True)
        self.barrier()
        train_dataset = FashionMNIST(DATASETS_PATH, train=True, transform=transform)
        valid_dataset = FashionMNIST(DATASETS_PATH, train=False, transform=transform)

        # Dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        train_loader, valid_loader = self.setup_dataloaders(train_loader, valid_loader)

        model = AlexNet(dropout=0.3)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        model, optimizer = self.setup(model, optimizer)

        scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, cfg.epochs + 1):

            # TRAINING LOOP
            model.train()
            self.set_dataloader_pbar(train_loader, desc=f"TRAIN {epoch:02d}", disable=(not self.is_global_zero))
            for batch_idx, (data, target) in enumerate(self.dataloader_pbar):
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                self.backward(loss)

                optimizer.step()
                if self.is_global_zero:
                    if (batch_idx == 0) or ((batch_idx + 1) % cfg.log_interval == 0):
                        self.log_dict["loss"] = loss.item() / cfg.batch_size
                        self.dataloader_pbar.set_postfix(self.log_dict)
                        wandb.log({"epoch": epoch, "train_loss": self.log_dict["loss"], "lr": scheduler.get_last_lr()[0]})
            scheduler.step()

            # VALIDATION LOOP
            model.eval()
            self.set_dataloader_pbar(valid_loader, desc=f"VALID {epoch:02d}", disable=(not self.is_global_zero))
            loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.dataloader_pbar:
                    output = model(data)
                    loss += loss_fn(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            loss = self.all_gather(loss).sum() / len(valid_loader.dataset)
            acc = self.all_gather(correct).sum() * 100 / len(valid_loader.dataset)
            if self.is_global_zero:
                self.log_dict["val_loss"] = loss.item()
                self.log_dict["val_acc"] = acc.item()
                self.dataloader_pbar.set_postfix(self.log_dict)
                wandb.log({"epoch": epoch, "val_loss": self.log_dict["val_loss"], "val_acc": self.log_dict["val_acc"]})

        if cfg.is_save:
            self.save(model.state_dict(), f"./weights/{cfg.wandb_project}/{cfg.wandb_name}-{cfg.now}.pt")

        if self.is_global_zero:
            print(json.dumps(self.log_dict, indent=4))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def lite_app(cfg: DictConfig) -> None:
    Lite(
        devices=[2, 3],
        accelerator="gpu",
        strategy="ddp",
        precision=16,
    ).run(cfg)


if __name__ == "__main__":
    lite_app()
