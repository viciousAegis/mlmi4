# data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def normalize_to_minus_one_one(x):
    """Convert [0,1] tensor to [-1,1]"""
    return x * 2.0 - 1.0


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    image_size: int
    num_channels: int


def _make_transform(train: bool) -> transforms.Compose:
    tfs = []
    if train:
        tfs.append(transforms.RandomHorizontalFlip(p=0.5))
    tfs += [
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(normalize_to_minus_one_one),  # -> [-1,1]
    ]
    return transforms.Compose(tfs)


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 256,
    val_size: int = 5000,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoaders:
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

    return DataLoaders(
        train=train_loader, 
        val=val_loader, 
        test=test_loader,
        image_size=32,
        num_channels=3
    )


def get_imagenet_loaders(
    root: str = "./data/imagenet",
    batch_size: int = 256,
    image_size: int = 64,
    val_size: int = 50000,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoaders:
    """
    Returns train/val/test loaders for ImageNet.
    Expects ImageNet directory structure:
      root/train/n01440764/*.JPEG
      root/val/n01440764/*.JPEG
    
    Args:
        root: path to ImageNet directory
        batch_size: batch size
        image_size: resize images to this size (default 64 for efficiency)
        val_size: number of samples for validation split from train set
        num_workers: number of data loading workers
        seed: random seed for train/val split
    """
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(normalize_to_minus_one_one),  # -> [-1,1]
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(normalize_to_minus_one_one),  # -> [-1,1]
    ])
    
    train_path = os.path.join(root, "train")
    val_path = os.path.join(root, "val")
    
    if not os.path.exists(train_path):
        raise ValueError(
            f"ImageNet train directory not found at {train_path}. "
            "Please set --data-root to your ImageNet directory."
        )
    
    # Load full training set
    train_full = datasets.ImageFolder(train_path, transform=train_transform)
    
    # Split train into train/val
    if val_size > 0 and val_size < len(train_full):
        gen = torch.Generator().manual_seed(seed)
        train_set, val_set_temp = random_split(
            train_full, [len(train_full) - val_size, val_size], generator=gen
        )
        
        # Create val set without augmentation
        val_base = datasets.ImageFolder(train_path, transform=val_transform)
        val_set = torch.utils.data.Subset(val_base, val_set_temp.indices)
    else:
        train_set = train_full
        # Use official val set
        if os.path.exists(val_path):
            val_set = datasets.ImageFolder(val_path, transform=val_transform)
        else:
            # No val split, use a small portion of train
            gen = torch.Generator().manual_seed(seed)
            train_set, val_set = random_split(
                train_full, [len(train_full) - 1000, 1000], generator=gen
            )
    
    # Test set (use official val set, or same as val if no official test)
    if os.path.exists(val_path):
        test_set = datasets.ImageFolder(val_path, transform=val_transform)
    else:
        test_set = val_set
    
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
    
    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        image_size=image_size,
        num_channels=3
    )


def get_dataloaders(
    dataset: str = "cifar10",
    root: str = "./data",
    batch_size: int = 256,
    val_size: int = 5000,
    num_workers: int = 4,
    seed: int = 42,
    image_size: int = 64,  # for ImageNet
) -> DataLoaders:
    """
    Unified interface for getting dataloaders.
    
    Args:
        dataset: "cifar10" or "imagenet"
        root: data directory
        batch_size: batch size
        val_size: validation split size
        num_workers: number of workers
        seed: random seed
        image_size: image size for ImageNet (ignored for CIFAR-10)
    """
    if dataset == "cifar10":
        return get_cifar10_loaders(
            root=root,
            batch_size=batch_size,
            val_size=val_size,
            num_workers=num_workers,
            seed=seed,
        )
    elif dataset == "imagenet":
        return get_imagenet_loaders(
            root=root,
            batch_size=batch_size,
            image_size=image_size,
            val_size=val_size,
            num_workers=num_workers,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'cifar10' or 'imagenet'.")


if __name__ == "__main__":
    loaders = get_cifar10_loaders(batch_size=64, val_size=5000)
    xb, yb = next(iter(loaders.train))
    print("train:", xb.shape, xb.min().item(), xb.max().item())
    xb, yb = next(iter(loaders.val))
    print("val  :", xb.shape, xb.min().item(), xb.max().item())
    xb, yb = next(iter(loaders.test))
    print("test :", xb.shape, xb.min().item(), xb.max().item())
