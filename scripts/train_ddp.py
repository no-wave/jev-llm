"""여러 GPU 로 Laya 를 파인튜닝한다 (14장). torchrun 이 프로세스를 GPU 수만큼 띄운다.

    torchrun --standalone --nproc_per_node=2 scripts/train_ddp.py data/train/human.jsonl multilingual models/ddp-human

2판에서 바꾼 것 둘. 기울기를 모으는 동안에는 no_sync() 로 all-reduce 를 미루고 갱신 직전에만 동기화한다.
토큰화는 걸음마다 하지 않고 학습 전에 한 번만 한다.

Laya 저장소의 Kaggle 2×T4 노트북(train_ddp.py, Apache-2.0)을 이 책의 데이터 형식에 맞춰 줄였다.
항목 만들기·손실·보정·저장은 laya.train 의 함수를 그대로 쓰고, 이 파일은 분산 처리만 맡는다.
GPU 가 없으면 gloo 백엔드로 CPU 프로세스 여럿이 돈다(동작 확인용, 느리다).
"""
import argparse
import contextlib
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from laya.calibrate import fit_temperature_map
from laya.common import collate_items
from laya.train import (calibration_records, encode_item, items_from_rows, load_checkpoint, read_jsonl,
                        rlcd_loss, save_checkpoint, sigma_at, soft_ce_loss, split_calibration)

p = argparse.ArgumentParser()
p.add_argument("data"); p.add_argument("base"); p.add_argument("out")
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--loss", choices=["rlcd", "soft-ce"], default="rlcd")
p.add_argument("--micro-batch", type=int, default=8)
p.add_argument("--grad-accum", type=int, default=4)        # 유효 배치 = micro_batch × GPU 수 × grad_accum
p.add_argument("--encoder-lr", type=float, default=2.5e-5)
p.add_argument("--head-lr", type=float, default=1e-4)
args = p.parse_args()

cuda = torch.cuda.is_available()
dist.init_process_group("nccl" if cuda else "gloo")
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
if cuda:
    torch.cuda.set_device(device)


def say(*a):
    if rank == 0:
        print(*a, flush=True)


model, tok, cfg = load_checkpoint(args.base)
max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
items, skipped = items_from_rows(tok, read_jsonl(args.data), max_len, head_max_len)

# 보정 조각은 나누기 전에 떼어 낸다. 시드가 같으므로 모든 rank 가 같은 조각을 뺀다.
train, calib = split_calibration(items, calib_max=400, calib_frac=0.1, seed=20260922)
train = train[: len(train) // world * world]                # rank 마다 같은 수를 받아야 all-reduce 가 맞물린다
mine = train[rank::world]
say(f"학습 {len(train)}건 (rank 당 {len(mine)}), 보정 {len(calib)}건, 건너뜀 {skipped}, GPU {world}개, 손실 {args.loss}")

encoded = [encode_item(tok, it, max_len, head_max_len) for it in mine]    # 토큰화는 학습 전에 한 번만 한다
model.to(device).train()
ddp = DDP(model, device_ids=[device.index] if cuda else None, find_unused_parameters=True)   # 마이크로 배치에 없는 질문 유형의 헤드는 기울기를 받지 않는다
enc = [q for n, q in ddp.named_parameters() if "encoder." in n]
head = [q for n, q in ddp.named_parameters() if "encoder." not in n]
opt = torch.optim.AdamW([{"params": enc, "lr": args.encoder_lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01)
steps = math.ceil(len(mine) / args.micro_batch)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, math.ceil(steps / args.grad_accum) * args.epochs), eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=cuda)

t0 = time.time()
for epoch in range(args.epochs):
    random.Random(epoch * 1000 + rank).shuffle(encoded)
    sigma, total = sigma_at(epoch, args.epochs, 0.4, 0.1), 0.0
    for step in range(steps):
        chunk = encoded[step * args.micro_batch:(step + 1) * args.micro_batch]
        b = {k: v.to(device) for k, v in collate_items([chunk], tok.pad_token_id).items() if torch.is_tensor(v)}
        sync = (step + 1) % args.grad_accum == 0 or step + 1 == steps    # 창이 차거나 epoch 끝이면 갱신한다
        with contextlib.nullcontext() if sync else ddp.no_sync():       # 갱신하지 않는 걸음에는 GPU 사이 동기화를 미룬다
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                logits, _ = ddp(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
            loss = {"rlcd": lambda: rlcd_loss(logits.float(), b["target"], b["marker_mask"], b["qtype"], sigma),
                    "soft-ce": lambda: soft_ce_loss(logits.float(), b["target"], b["marker_mask"])}[args.loss]()
            window = min(args.grad_accum, steps - step // args.grad_accum * args.grad_accum)
            scaler.scale(loss / window).backward()
        if sync:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        total += loss.item()
    say(f"epoch {epoch + 1}/{args.epochs}  평균 손실 {total / steps:.4f}  {time.time() - t0:.0f}초")
    dist.barrier()

if rank == 0:   # 보정과 저장은 한 프로세스만 한다
    model.eval()
    fitted = fit_temperature_map(calibration_records(model, tok, calib, device, max_len, head_max_len))
    out_cfg = {k: v for k, v in cfg.items() if k != "temperature_by_options"}   # 물려받은 버킷 온도가 새 값을 가리지 않게 지운다
    out_cfg.update(max_len=max_len, head_max_len=head_max_len, temperature=fitted["temperature"], fine_tuned=True,
                   **({"temperature_by_options": fitted["temperature_by_options"]} if fitted["temperature_by_options"] else {}))
    save_checkpoint(model, tok, out_cfg, args.out)
    json.dump({"data": args.data, "base": args.base, "loss": args.loss, "epochs": args.epochs, "world_size": world,
               "train_items": len(train), "calibration_items": len(calib), "temperature": fitted["temperature"],
               "seconds": round(time.time() - t0, 1)}, open(os.path.join(args.out, "train_ddp.json"), "w"), indent=1)
    say("저장:", args.out, "| 온도 (choice, score, noul):", [round(t, 3) for t in fitted["temperature"]])
dist.destroy_process_group()
