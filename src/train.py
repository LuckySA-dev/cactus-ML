"""
train.py — เทรน 2 stage (freeze backbone → fine-tune ทั้งตัว) เก็บ checkpoint ที่ val macro F1 ดีที่สุด

    python -m src.train --model resnet50 --seed 42
    python -m src.train --model resnet50 --seed 42 --no-pretrained   # baseline ไม่ใช้ transfer learning
    python -m src.train --model mobilenetv3 --smoke                  # smoke test 1+1 epoch

    # จูน: ค่าตั้งต้นมาจาก config.STAGE1/STAGE2 — override ได้ + ตั้ง --tag ไม่ให้ทับ run อื่น
    python -m src.train --model effnetv2b0 --epochs2 80 --patience 15 --tag ep80

ผลลัพธ์
    checkpoints/<run>_best.pt   (state_dict + class_to_idx + mean/std + hparams + val metrics)
    reports/logs/<run>.csv      (loss / acc / macro F1 ทุก epoch)
    reports/runs.csv            (ต่อท้าย 1 แถว/run: hparams + best val F1 — ใช้เทียบตอนจูน)

⚠️ จูนโดยดู val เท่านั้น ห้ามดู test — ไม่งั้นตัวเลข test สูงเกินจริง
--no-pretrained ใช้ขั้นตอนเดียวกันทุกอย่าง ต่างแค่น้ำหนักเริ่มต้น → เทียบได้ตรง ๆ ว่า transfer learning ช่วยแค่ไหน
ไม่ทำ resume กลาง run: 1 run ใช้ไม่กี่นาที รันใหม่ถูกกว่าเก็บ optimizer state
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from src.config import (
    CHECKPOINT_DIR,
    DEVICE,
    LABEL_SMOOTHING,
    LOG_DIR,
    MODELS,
    RUNS_CSV,
    SEED,
    STAGE1,
    STAGE2,
    WEIGHT_DECAY,
)
from src.datasets import build_loaders
from src.engine import evaluate, set_seed, train_one_epoch
from src.models import build_model, data_config, head_params, param_groups, set_backbone_trainable
from src.tasks import TASKS, genus_of, get_task, run_id

LOG_COLS = ["stage", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "is_best", "sec"]


def run_name(args: argparse.Namespace) -> str:
    # ไม่มี --tag / --no-pretrained = run หลัก (tasks.run_id) ที่ evaluate / compare / visualize / แอปใช้
    return (run_id(args.model, args.seed, args.task)
            + ("_scratch" if args.no_pretrained else "")
            + (f"_{args.tag}" if args.tag else "")
            + ("_smoke" if args.smoke else ""))


def hparams(args: argparse.Namespace) -> dict:
    keys = ("epochs1", "lr1", "epochs2", "lr_backbone", "lr_head", "patience", "weight_decay", "balance")
    hp = {k: getattr(args, k) for k in keys}
    if args.smoke:
        hp |= {"epochs1": 1, "epochs2": 1}
    return hp


class Trainer:
    """ของที่ใช้ร่วมกันทั้ง 2 stage: โมเดล, loader, log, best checkpoint"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.name = run_name(args)
        self.hp = hparams(args)
        self.task = get_task(args.task)
        self.model = build_model(args.model, pretrained=not args.no_pretrained, num_classes=len(self.task.classes))
        self.model.to(DEVICE, memory_format=torch.channels_last)
        self.cfg = data_config(self.model)
        self.loaders = build_loaders(self.cfg["mean"], self.cfg["std"], splits=["train", "val"], seed=args.seed,
                                     root=self.task.root, classes=self.task.classes)
        if set(self.loaders) != {"train", "val"}:
            raise SystemExit("ไม่มีข้อมูล train/val — รัน python -m src.data.split ก่อน")
        self.criterion = nn.CrossEntropyLoss(weight=self._balance(), label_smoothing=LABEL_SMOOTHING)
        self.ckpt_path = CHECKPOINT_DIR / f"{self.name}_best.pt"
        self.log_path = LOG_DIR / f"{self.name}.csv"
        self.best = {"val_f1": -1.0, "stage": 0, "epoch": 0}
        self.last_epoch = 0              # epoch สุดท้ายที่เทรนจริงใน stage 2 (ดูว่า early stop ตอนไหน)

    def _balance(self) -> torch.Tensor | None:
        """
        จัดการคลาสไม่สมดุลตาม --balance → คืน class weight ให้ loss (หรือ None)
            weight       น้ำหนัก loss ∝ 1/จำนวนภาพของคลาส (สูตร 'balanced' แบบ sklearn) ใช้ภาพครบ
            undersample  สุ่มตัดภาพ train ให้ทุกสกุลเหลือเท่าสกุลที่น้อยสุด (ตัดครั้งเดียว ชุดเดิมทุก epoch)
        ต้องเรียกก่อนเริ่มเทรน — worker ของ DataLoader ยังไม่ถูกสร้าง จึงแก้ dataset ตรง ๆ ได้
        """
        ds = self.loaders["train"].dataset
        if self.args.balance == "weight":
            counts = torch.bincount(torch.tensor(ds.targets), minlength=len(self.task.classes)).float()
            return (counts.sum() / (len(counts) * counts.clamp(min=1))).to(DEVICE)
        if self.args.balance == "undersample":
            by_genus = defaultdict(list)
            for i, t in enumerate(ds.targets):
                by_genus[genus_of(self.task.classes[t])].append(i)
            n = min(map(len, by_genus.values()))
            rng = random.Random(self.args.seed)
            keep = sorted(i for idx in by_genus.values() for i in rng.sample(idx, n))
            ds.samples = ds.imgs = [ds.samples[i] for i in keep]
            ds.targets = [t for _, t in ds.samples]
            print(f"  undersample: เหลือ {n} ภาพ/สกุล รวม {len(keep)} ภาพ")
        return None

    def fit(self) -> None:
        hp, t0 = self.hp, time.perf_counter()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLS).writeheader()

        print(f"[{self.name}] stage 1: freeze backbone, lr {hp['lr1']}, {hp['epochs1']} epochs")
        set_backbone_trainable(self.model, False)
        opt = AdamW(head_params(self.model), lr=hp["lr1"], weight_decay=hp["weight_decay"])
        self._run_stage(1, opt, hp["epochs1"])

        print(f"[{self.name}] stage 2: fine-tune ทั้งตัว, lr {hp['lr_backbone']} / {hp['lr_head']}, "
              f"{hp['epochs2']} epochs, patience {hp['patience']}")
        set_backbone_trainable(self.model, True)
        opt = AdamW(param_groups(self.model, hp["lr_backbone"], hp["lr_head"]), weight_decay=hp["weight_decay"])
        self._run_stage(2, opt, hp["epochs2"], CosineAnnealingLR(opt, T_max=hp["epochs2"]), hp["patience"])

        minutes = (time.perf_counter() - t0) / 60
        self._record_run(minutes)
        print(f"[{self.name}] best val macro F1 = {self.best['val_f1']:.4f} "
              f"(S{self.best['stage']} ep{self.best['epoch']}, {minutes:.1f} นาที) → {self.ckpt_path}")

    def _run_stage(
        self, stage: int, optimizer: Optimizer, epochs: int,
        scheduler: LRScheduler | None = None, patience: int | None = None,
    ) -> None:
        """patience=None คือไม่มี early stopping"""
        left = patience
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = train_one_epoch(
                self.model, self.loaders["train"], self.criterion, optimizer, frozen_backbone=stage == 1)
            if scheduler:
                scheduler.step()
            self.last_epoch = epoch
            improved = self._validate(stage, epoch, train_loss, train_acc, t0)
            if patience is not None:
                left = patience if improved else left - 1
                if left == 0:
                    print(f"  early stop: val F1 ไม่ดีขึ้น {patience} epoch ติด")
                    return

    def _validate(self, stage: int, epoch: int, train_loss: float, train_acc: float, t0: float) -> bool:
        r = evaluate(self.model, self.loaders["val"], self.criterion)
        improved = r.macro_f1 > self.best["val_f1"]     # เลือกจาก macro F1 ไม่ใช่ accuracy
        if improved:
            self.best = {"val_f1": r.macro_f1, "stage": stage, "epoch": epoch}
            self._save_checkpoint(r.acc)
        sec = time.perf_counter() - t0
        self._log(stage=stage, epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=r.loss,
                  val_acc=r.acc, val_f1=r.macro_f1, is_best=int(improved), sec=sec)
        print(f"  S{stage} ep{epoch:02d}  train {train_loss:.3f}/{train_acc:.3f}  "
              f"val {r.loss:.3f}/{r.acc:.3f}  F1 {r.macro_f1:.4f}{'  ★' if improved else ''}  ({sec:.0f}s)")
        return improved

    def _log(self, **row) -> None:
        # เปิดต่อท้ายทีละ epoch → เทรนพังกลางทาง log ที่มีแล้วก็ยังอยู่
        with self.log_path.open("a", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLS).writerow(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    def _save_checkpoint(self, val_acc: float) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "model_key": self.args.model,
            "timm_name": MODELS[self.args.model],
            "task": self.task.name,
            "classes": self.task.classes,
            "class_to_idx": self.loaders["train"].dataset.class_to_idx,
            "mean": self.cfg["mean"],
            "std": self.cfg["std"],
            "hparams": self.hp,
            "stage": self.best["stage"],
            "epoch": self.best["epoch"],
            "val_acc": val_acc,
            "val_f1": self.best["val_f1"],
            "seed": self.args.seed,
            "pretrained": not self.args.no_pretrained,
        }, self.ckpt_path)

    def _record_run(self, minutes: float) -> None:
        """ต่อท้าย reports/runs.csv — ตารางเทียบทุก run ที่เคยเทรน (ไม่ลบของเก่า)"""
        row = {"time": datetime.now().isoformat(timespec="seconds"), "run": self.name,
               "phase": "tuning" if self.args.tag else "final",       # มี --tag = run จูน
               "task": self.task.name, "model": self.args.model,
               "seed": self.args.seed, "pretrained": not self.args.no_pretrained, **self.hp,
               "best_val_f1": round(self.best["val_f1"], 4), "best_stage": self.best["stage"],
               "best_epoch": self.best["epoch"], "stage2_epochs_run": self.last_epoch, "minutes": round(minutes, 1)}
        # อ่านของเดิม + รวมคอลัมน์ แล้วเขียนใหม่ทั้งไฟล์ → เพิ่ม hparam ใหม่ทีหลังคอลัมน์ไม่เลื่อน
        rows = []
        if RUNS_CSV.exists():
            with RUNS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        rows.append(row)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with RUNS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="เทรนโมเดลจำแนกสกุลกระบองเพชร (2 stage)")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--task", default="genus", choices=TASKS,
                   help="genus = 7 สกุล · species-<สกุล> = คลาสย่อยของสกุลนั้น · flat = ทุกคลาสย่อยในโมเดลเดียว")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-pretrained", action="store_true", help="เริ่มจากน้ำหนักสุ่ม (baseline)")
    p.add_argument("--smoke", action="store_true", help="1+1 epoch เช็คว่า pipeline ไม่พัง")
    p.add_argument("--tag", default="", help="ต่อท้ายชื่อ run (ใช้ตอนจูน จะได้ไม่ทับ run หลัก)")
    g = p.add_argument_group("hyperparameter (ค่าตั้งต้นจาก config)")
    g.add_argument("--epochs1", type=int, default=STAGE1["epochs"])
    g.add_argument("--lr1", type=float, default=STAGE1["lr"])
    g.add_argument("--epochs2", type=int, default=STAGE2["epochs"])
    g.add_argument("--lr-backbone", type=float, default=STAGE2["lr_backbone"])
    g.add_argument("--lr-head", type=float, default=STAGE2["lr_head"])
    g.add_argument("--patience", type=int, default=STAGE2["patience"])
    g.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    g.add_argument("--balance", default="none", choices=["none", "weight", "undersample"],
                   help="จัดการคลาสไม่สมดุล (ดู Trainer._balance)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Trainer(args).fit()


if __name__ == "__main__":
    main()
