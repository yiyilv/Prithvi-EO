from __future__ import annotations

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassF1Score
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


class PrithviLucasClassifier(pl.LightningModule):
    """CLS-token classifier for six-band LUCAS chips."""

    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        unfreeze_last_blocks: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = TERRATORCH_BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300_tl", pretrained=True
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if unfreeze_last_blocks < 0:
            raise ValueError("unfreeze_last_blocks must be zero or greater")
        if unfreeze_last_blocks:
            self._unfreeze_last_blocks(unfreeze_last_blocks)

        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError("Prithvi backbone does not expose embed_dim")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def _unfreeze_last_blocks(self, count: int) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None and hasattr(self.backbone, "model"):
            blocks = getattr(self.backbone.model, "blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Transformer blocks on this TerraTorch backbone")
        if count < 0 or count > len(blocks):
            raise ValueError(f"Unfreeze count must be between 0 and {len(blocks)}")
        for block in list(blocks)[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name, module in self.backbone.named_modules():
            if "norm" in name.lower() and "head" not in name.lower():
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(image.unsqueeze(2))[-1]
        return self.classifier(self.dropout(tokens[:, 0]))

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch["image"].float())
        labels = batch["label"]
        loss = self.loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)
        metric = {"train": self.train_f1, "val": self.val_f1, "test": self.test_f1}[stage]
        metric.update(predictions, labels)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=len(labels))
        self.log(f"{stage}/acc", (predictions == labels).float().mean(),
                 on_step=False, on_epoch=True, prog_bar=True, batch_size=len(labels))
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "test")

    def on_train_epoch_end(self) -> None:
        self.log("train/macro_f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("val/macro_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def on_test_epoch_end(self) -> None:
        self.log("test/macro_f1", self.test_f1.compute(), prog_bar=True)
        self.test_f1.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        backbone_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        head_parameters = list(self.classifier.parameters())
        groups = [{"params": head_parameters, "lr": self.hparams.learning_rate}]
        if backbone_parameters:
            groups.append({"params": backbone_parameters,
                           "lr": self.hparams.learning_rate * 0.1})
        return torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
