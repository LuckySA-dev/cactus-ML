"""
datasets.py — transforms + Dataset + DataLoader ของ data/processed/<split>/<label>/

    python -m src.datasets --model resnet50    # ตรวจจำนวนภาพต่อ split + ลองโหลด 1 batch
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from src.config import BATCH_SIZE, CLASSES, EVAL_RESIZE, IMG_SIZE, NUM_WORKERS, PROCESSED_DIR, SEED

SPLITS = ["train", "val", "test", "realworld_test"]


def load_image(path) -> Image.Image:
    """
    เปิดภาพแบบเดียวกันทุกที่ (เทรน / evaluate / gradcam / predict / app)
    exif_transpose: รูปมือถือแนวตั้งเก็บพิกเซลแนวนอน + ธง EXIF — ถ้าไม่หมุน โมเดลจะเห็นต้นนอนตะแคง
    (ภาพ iNat ไม่มี EXIF แล้ว แต่ภาพถ่ายเอง / ภาพที่ผู้ใช้อัปโหลดมี)
    """
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class CactusFolder(ImageFolder):
    """
    ImageFolder ที่ล็อกลำดับคลาสตามที่กำหนด (ค่าเริ่มต้น config.CLASSES — งานอื่นส่ง task.classes)

    ImageFolder ปกติหาคลาสจากโฟลเดอร์ที่มีอยู่ — ถ้า split ไหนขาดบางคลาส
    index จะเลื่อน แล้ว confusion matrix ติด label ผิดทั้งแถบ
    """

    def __init__(self, root, transform: Callable | None = None, classes: list[str] = CLASSES):
        self.fixed_classes = list(classes)          # ต้องตั้งก่อน super().__init__ (ซึ่งเรียก find_classes)
        super().__init__(root, transform=transform, loader=load_image, allow_empty=True)

    def find_classes(self, directory):
        return self.fixed_classes, {c: i for i, c in enumerate(self.fixed_classes)}


def build_transforms(mean: tuple, std: tuple) -> tuple[v2.Compose, v2.Compose]:
    """
    (train_tf, eval_tf) — mean/std ต้องมาจาก timm.data.resolve_model_data_config(model) เท่านั้น

    ห้ามใส่ RandomVerticalFlip (กระบองเพชรมีบน-ล่างชัด) และ hue แรง ๆ (สีใช้แยกสกุล)
    """
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)]
    train_tf = v2.Compose([
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        *to_tensor,
        v2.RandomErasing(p=0.25),
    ])
    eval_tf = v2.Compose([v2.Resize(EVAL_RESIZE), v2.CenterCrop(IMG_SIZE), *to_tensor])
    return train_tf, eval_tf


def build_loaders(
    mean: tuple,
    std: tuple,
    splits: list[str] = SPLITS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
    root: Path = PROCESSED_DIR,
    classes: list[str] = CLASSES,
) -> dict[str, DataLoader]:
    """
    {split: DataLoader} เฉพาะ split ที่มีภาพ — train สุ่ม + ใช้ augmentation, split อื่นไม่สุ่ม
    root = โฟลเดอร์ที่มี <split>/<label>/ (ปกติ data/processed, flag_suspects ใช้โฟลเดอร์ของแต่ละ fold)

    ⚠️ Windows: ต้องเรียกจากใต้ if __name__ == "__main__": เท่านั้น
    """
    train_tf, eval_tf = build_transforms(mean, std)
    loaders = {}
    for split in splits:
        is_train = split == "train"
        ds = CactusFolder(root / split, transform=train_tf if is_train else eval_tf, classes=classes)
        if not len(ds):
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            # Windows: worker ทุกตัว import torch+CUDA จอง virtual memory ~2 GB และ persistent อยู่ตลอด
            # val/test ภาพน้อย ใช้ 2 ตัวพอ — 8+8 ตัวเคยชนเพดาน commit (error 1455 / CUDA OOM ทั้งที่ GPU ว่าง)
            num_workers=num_workers if is_train else min(num_workers, 2),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),   # ลำดับสุ่มเดิมทุกครั้ง
        )
    return loaders


def main() -> None:
    import timm

    from src.config import MODELS

    p = argparse.ArgumentParser(description="ตรวจ DataLoader")
    p.add_argument("--model", default="resnet50", choices=list(MODELS))
    args = p.parse_args()

    cfg = timm.data.resolve_model_data_config(timm.create_model(MODELS[args.model], pretrained=False))
    loaders = build_loaders(cfg["mean"], cfg["std"])
    print(f"mean={cfg['mean']} std={cfg['std']}")
    for split, dl in loaders.items():
        counts = torch.bincount(torch.tensor(dl.dataset.targets), minlength=len(CLASSES)).tolist()
        per_class = ", ".join(f"{c}={n}" for c, n in zip(CLASSES, counts, strict=True))
        print(f"{split:<15} {len(dl.dataset):>5} ภาพ  {per_class}")
    x, y = next(iter(loaders["train"]))
    print(f"batch: x={tuple(x.shape)} {x.dtype}  y={tuple(y.shape)}  x.mean={x.mean():.3f}")


if __name__ == "__main__":
    main()
