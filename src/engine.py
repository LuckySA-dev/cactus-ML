"""
engine.py — ลูปเทรน / ประเมินผล 1 epoch (ใช้ร่วมกันทุกโมเดล)

- AMP ใช้ bfloat16 (Blackwell รองรับเต็มที่ ไม่ต้องใช้ GradScaler)
- channels_last เร็วขึ้น ~20-30% กับ conv net
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.config import DEVICE
from src.models import freeze_bn


def set_seed(seed: int) -> None:
    """เรียกก่อนสร้างโมเดล/DataLoader — เทรนซ้ำต้องได้ผลเดิม ไม่งั้นเขียนรายงานไม่ได้"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # จำเป็นสำหรับ deterministic บน CUDA
    # กัน OOM จาก memory fragmentation ตอนเปลี่ยน stage 1 → 2 (DenseNet เคยพังทั้งที่ GPU ว่าง 11 GB)
    # ต้องตั้งก่อนจอง GPU ครั้งแรก — set_seed ถูกเรียกก่อนสร้างโมเดล
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # เทรนแค่ไม่กี่นาที → ยอมช้าลงนิดแลกกับผลเดิมทุกครั้ง (ไม่ใช้ cudnn.benchmark)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class EvalResult:
    loss: float
    acc: float
    macro_f1: float
    y_true: np.ndarray    # (N,)
    y_pred: np.ndarray    # (N,)
    probs: np.ndarray     # (N, num_classes) — softmax ใช้กับ confidence threshold


def _to_device(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last),
            y.to(DEVICE, non_blocking=True))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    frozen_backbone: bool,
) -> tuple[float, float]:
    """คืน (train_loss, train_acc) — frozen_backbone=True ใน stage 1 เพื่อไม่ให้ BN อัปเดต running stats"""
    model.train()
    if frozen_backbone:
        freeze_bn(model)
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = _to_device(x, y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> EvalResult:
    """fp32 (ไม่ใช้ bf16) — ให้ความมั่นใจตรงกับ predict.py/แอปเป๊ะ ภาพที่อยู่ใกล้ threshold จะได้ไม่ตัดสินต่างกัน"""
    model.eval()
    loss_sum, probs, ys = 0.0, [], []
    for x, y in loader:
        x, y = _to_device(x, y)
        logits = model(x).float()
        loss_sum += criterion(logits, y).item() * len(y)
        probs.append(logits.softmax(1).cpu())
        ys.append(y.cpu())
    probs_np = torch.cat(probs).numpy()
    y_true = torch.cat(ys).numpy()
    y_pred = probs_np.argmax(1)
    return EvalResult(
        loss=loss_sum / len(y_true),
        acc=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        y_true=y_true,
        y_pred=y_pred,
        probs=probs_np,
    )
