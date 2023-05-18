from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser

import torch.nn.functional as F
import torchvision.transforms as T
import torch
from torch import nn, optim
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.eurosat_datamodule import EurosatDataModule
from models.moco2_module import MocoV2


class Classifier(LightningModule):

    def __init__(self, backbone, in_features, num_classes, weight_decay, norm):
        super().__init__()
        self.encoder = backbone
        self.weight_decay = weight_decay
        self.norm = norm

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
            if self.norm:
                feats = F.normalize(feats, p=2, dim=-1)
        logits = self.classifier(feats)
        return logits

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('test/loss', loss, prog_bar=True)
        self.log('test/acc', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(torch.argmax(logits.softmax(dim=1), dim=1), y)
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.classifier.parameters(), weight_decay=self.weight_decay)
        max_epochs = self.trainer.max_epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument('--weights', type=str, default="imagenet", choices=["random", "imagenet", "pretrain"])
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    transforms = [T.ToTensor(), T.Resize((224, 224))]
    if args.weights == "imagenet":
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
        transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    datamodule = EurosatDataModule(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, transforms=T.Compose(transforms))

    if args.weights == "random":
        if args.backbone == "resnet18":
            backbone = resnet.resnet18(pretrained=False)
            backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        else:
            backbone = resnet.resnet50(pretrained=False)
            backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.weights == "imagenet":
        if args.backbone == "resnet18":
            backbone = resnet.resnet18(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        else:
            backbone = resnet.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.weights == "pretrain":
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
    else:
        raise ValueError()

    model = Classifier(
        backbone,
        in_features=512 if args.backbone == "resnet18" else 2048,
        num_classes=datamodule.num_classes,
        weight_decay=args.weight_decay,
        norm=args.norm
    )
    model.example_input_array = torch.zeros((1, 3, 224, 224))

    experiment_name = f"{args.weights}_{args.backbone}_{args.weight_decay}"
    logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'eurosat'), name=experiment_name)
    trainer = Trainer(gpus=[args.gpus], logger=logger, checkpoint_callback=True, max_epochs=args.epochs)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")
