from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch

from prithvi_lucas.data import LUCASDataModule
from prithvi_lucas.model import PrithviLucasClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Prithvi-EO on prepared LUCAS chips")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prithvi-lucas"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    args = parser.parse_args()

    if args.unfreeze_last_blocks < 0:
        parser.error("--unfreeze-last-blocks must be zero or greater")
    pl.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = LUCASDataModule(args.data_root, args.train_split, args.val_split,
                           args.test_split, args.batch_size, args.num_workers)
    model = PrithviLucasClassifier(
        learning_rate=args.learning_rate,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
    )
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-{epoch:02d}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val/acc", mode="max", patience=5,
                               min_delta=0.001)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=args.epochs,
        callbacks=[checkpoint, early_stop],
        default_root_dir=args.output_dir,
        log_every_n_steps=5,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model=model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    main()
