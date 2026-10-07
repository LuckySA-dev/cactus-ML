"""
models.py — สร้างโมเดล 3 ตัวผ่าน timm + ตัวช่วยสำหรับเทรน 2 stage

Stage 1: set_backbone_trainable(model, False) + หลัง model.train() เรียก freeze_bn(model)
Stage 2: set_backbone_trainable(model, True) + optimizer ใช้ param_groups(...) (lr backbone < head)
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from src.config import CLASSES, DEVICE, DROP_RATE, MODELS, NUM_CLASSES


def build_model(key: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    key = ชื่อใน config.MODELS (mobilenetv3 / effnetv2b0 / resnet50)

    head เริ่มที่ 0 ทุกโมเดล: timm init head ของ mobilenet/effnet แบบ Google (std ~0.22 เมื่อมี 7 คลาส)
    → loss เริ่มต้น 5.8–8.1 ขณะที่ resnet50 เริ่มที่ 1.94 = เทียบ 3 โมเดลไม่ยุติธรรม
    zero-init → ทุกตัวเริ่มที่ ln(7) ≈ 1.95 เท่ากัน (gradient ของ weight ยังไม่เป็นศูนย์ เพราะ feature ไม่เป็นศูนย์)
    """
    model = timm.create_model(MODELS[key], pretrained=pretrained, num_classes=num_classes, drop_rate=DROP_RATE)
    for p in model.get_classifier().parameters():
        nn.init.zeros_(p)
    return model


def load_checkpoint(path: Path, device: str = DEVICE) -> tuple[nn.Module, dict]:
    """
    โหลด checkpoint จาก train.py → (โมเดลพร้อมใช้ eval mode, dict ของ checkpoint)
    device="cpu" ใช้ได้บนเครื่องไม่มี GPU (app / predict)
    ckpt["task"] / ckpt["classes"] บอกว่าเป็นโมเดลงานไหน (genus / species-<สกุล> / flat)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)   # ไม่ unpickle โค้ดแปลกปลอม
    ckpt.setdefault("task", "genus")                                  # checkpoint รุ่นก่อนมีงานชั้น 2
    if ckpt["task"] == "genus" and ckpt["classes"] != list(CLASSES):
        raise SystemExit(f"{Path(path).name}: คลาสใน checkpoint ไม่ตรงกับ config.CLASSES — เทรนใหม่ก่อน")
    model = build_model(ckpt["model_key"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    # น้ำหนักหลังเทรนมีค่า subnormal (DenseNet ~78k ค่า) → CPU เข้าทางช้า ทายช้าลง ~13 เท่า (37 → 500 ms)
    # ปัดเป็น 0 — ผลทายไม่เปลี่ยน (ค่าเล็กกว่า 1e-38)
    torch.set_flush_denormal(True)
    return model.to(device, memory_format=torch.channels_last).eval(), ckpt


def data_config(model: nn.Module) -> dict:
    """mean / std ที่ weight นี้ถูก pretrain มา — ห้าม hard-code เพราะเปลี่ยนตาม pretrained tag ของ timm"""
    return timm.data.resolve_model_data_config(model)


def head_params(model: nn.Module) -> list[nn.Parameter]:
    return list(model.get_classifier().parameters())


def backbone_params(model: nn.Module) -> list[nn.Parameter]:
    head = {id(p) for p in head_params(model)}
    return [p for p in model.parameters() if id(p) not in head]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in backbone_params(model):
        p.requires_grad = trainable


def freeze_bn(model: nn.Module) -> None:
    """
    เรียกหลัง model.train() ทุกครั้งใน stage 1
    model.train() ทำให้ BN อัปเดต running stats แม้ requires_grad=False → ผล stage 1 เพี้ยน
    (BatchNormAct2d ของ timm เป็น subclass ของ BatchNorm2d จึงโดนด้วย)
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float) -> list[dict]:
    """discriminative lr — head ที่ยังสุ่มอยู่เรียนเร็วกว่า backbone ที่ pretrain มาแล้ว"""
    return [
        {"params": backbone_params(model), "lr": lr_backbone},
        {"params": head_params(model), "lr": lr_head},
    ]


def count_params(model: nn.Module) -> float:
    """จำนวนพารามิเตอร์ (ล้าน) — ใช้ในตารางผลลัพธ์"""
    return sum(p.numel() for p in model.parameters()) / 1e6
