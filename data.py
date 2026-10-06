from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl


class LUCASChipsDataset(Dataset):
    """Read six-band chips using split entries formatted as ``class/point_id``."""

    def __init__(self, chips_root: str | Path, split_file: str | Path, augment: bool = False):
        self.chips_root = Path(chips_root)
        self.split_file = Path(split_file)
        self.augment = augment
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        self.samples: list[tuple[Path, int]] = []
        missing: list[Path] = []
        for line_number, raw_line in enumerate(
            self.split_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            item = raw_line.strip()
            if not item:
                continue
            if "/" not in item:
                raise ValueError(f"Invalid split entry on line {line_number}: {item!r}")
            class_id_text, point_id = item.split("/", maxsplit=1)
            try:
                class_id = int(class_id_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid class ID on split line {line_number}: {class_id_text!r}"
                ) from error
            if not 1 <= class_id <= 10:
                raise ValueError(f"Class ID must be in the range 1-10 on split line {line_number}")
            if not point_id or "/" in point_id or "\\" in point_id:
                raise ValueError(f"Invalid Point_ID on split line {line_number}: {point_id!r}")
            chip_path = self.chips_root / str(class_id) / f"{point_id}.tif"
            if chip_path.is_file():
                self.samples.append((chip_path, int(class_id) - 1))
            else:
                missing.append(chip_path)
        if missing:
            sample = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(f"{len(missing)} split chips are missing; examples: {sample}")
        if not self.samples:
            raise ValueError(f"No samples found in {self.split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path, label = self.samples[index]
        with rasterio.open(path) as source:
            image = source.read().astype(np.float32, copy=False)
        if image.shape != (6, 224, 224):
            raise ValueError(f"Expected a (6, 224, 224) chip at {path}, got {image.shape}")
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.maximum(image, 0.0)
        tensor = torch.from_numpy(np.ascontiguousarray(image))

        if self.augment:
            k = random.randrange(4)
            tensor = torch.rot90(tensor, k, dims=(-2, -1))
            if random.randrange(2):
                tensor = tensor.flip(-1)

        return {"image": tensor, "label": torch.tensor(label, dtype=torch.long), "path": str(path)}


class LUCASDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        train_split: str | Path,
        val_split: str | Path,
        test_split: str | Path,
        batch_size: int = 8,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split_paths = (Path(train_split), Path(val_split), Path(test_split))
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        train_path, val_path, test_path = self.split_paths
        if stage in (None, "fit"):
            self.train_dataset = LUCASChipsDataset(self.data_root, train_path, augment=True)
            self.val_dataset = LUCASChipsDataset(self.data_root, val_path)
        if stage in (None, "test", "predict"):
            self.test_dataset = LUCASChipsDataset(self.data_root, test_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
