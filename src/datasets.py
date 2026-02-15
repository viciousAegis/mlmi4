# data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class CIFARLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _make_transform(train: bool) -> transforms.Compose:
    tfs = []
    if train:
        tfs.append(transforms.RandomHorizontalFlip(p=0.5))
    tfs += [
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # -> [-1,1]
    ]
    return transforms.Compose(tfs)


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 256,
    val_size: int = 5000,
    num_workers: int = 4,
    seed: int = 42,
) -> CIFARLoaders:
    """
    Returns train/val/test loaders for CIFAR-10.
    Train/val are split from the official train set (50k).
    Test is the official test set (10k).
    """
    train_full = datasets.CIFAR10(
        root=root, train=True, download=True, transform=_make_transform(train=True)
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, download=True, transform=_make_transform(train=False)
    )

    if not (0 < val_size < len(train_full)):
        raise ValueError(
            f"val_size must be between 1 and {len(train_full)-1}, got {val_size}"
        )

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_full, [len(train_full) - val_size, val_size], generator=gen
    )

    # Important: val_set currently uses the same transform object as train_full (with flip).
    # For a proper validation set, we should disable augmentation.
    # We can do this by overwriting val_set.dataset.transform.
    # random_split returns Subset wrappers; they share the same underlying dataset object,
    # so we need to re-wrap to avoid changing train transform too.
    # Easiest: create a second CIFAR10 dataset for val without augmentation, and reuse indices.
    val_base = datasets.CIFAR10(
        root=root, train=True, download=False, transform=_make_transform(train=False)
    )
    val_set = torch.utils.data.Subset(val_base, val_set.indices)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    return CIFARLoaders(train=train_loader, val=val_loader, test=test_loader)


if __name__ == "__main__":
    loaders = get_cifar10_loaders(batch_size=64, val_size=5000)
    xb, yb = next(iter(loaders.train))
    print("train:", xb.shape, xb.min().item(), xb.max().item())
    xb, yb = next(iter(loaders.val))
    print("val  :", xb.shape, xb.min().item(), xb.max().item())
    xb, yb = next(iter(loaders.test))
    print("test :", xb.shape, xb.min().item(), xb.max().item())
