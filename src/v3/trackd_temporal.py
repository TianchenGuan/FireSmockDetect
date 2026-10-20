"""Track D: learned causal temporal head on frozen expE features.

Model: per-frame conv encoder -> ConvGRU unrolled over a causal window of
T<=5 frames (current + up to 4 past; valid output at T=1) -> pooled hidden +
current-frame max-conf scalar -> MLP -> per-frame smoke logit.

Training: windows sampled for every frame of every train sequence; temporal
augmentation (random stride in {1,2}, random past-frame drop) simulates
missed uploads; BCEWithLogits with pos_weight = n_neg/n_pos.
Selection: best image-level average precision on val frames.
Resumable: saves last.pt every epoch; --resume picks it up.

Usage (pod or devbox):
  python trackd_temporal.py --features <dir with train/ val/> --out <run_dir>
    [--epochs 8] [--bs 256] [--lr 1e-3] [--overfit-seqs 8] [--max-steps N]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WIN = 5
N_CH = 194
HW = (12, 20)


# ---------------------------------------------------------------- data
class SeqStore:
    """Loads every sequence npz of a split into RAM (fp16, ~2GB train)."""

    def __init__(self, split_dir: Path, limit_seqs: int = 0):
        self.seqs = []
        files = sorted(split_dir.glob("*.npz"))
        if limit_seqs:
            files = files[:limit_seqs]
        for f in files:
            z = np.load(f, allow_pickle=False)
            self.seqs.append({
                "seq_id": f.stem,
                "feat": z["feat"],                       # (T,194,12,20) fp16
                "score": z["score"].astype(np.float32),
                "ts": z["ts"].astype(np.int64),
                "smoke": z["smoke"].astype(np.float32),
                "max_gt_area": z["max_gt_area"].astype(np.float32),
            })
        self.index = [(i, t) for i, s in enumerate(self.seqs)
                      for t in range(len(s["smoke"]))]
        n_pos = sum(float(s["smoke"].sum()) for s in self.seqs)
        n_all = sum(len(s["smoke"]) for s in self.seqs)
        self.n_pos, self.n_neg = int(n_pos), int(n_all - n_pos)


class WindowDataset(Dataset):
    def __init__(self, store: SeqStore, augment: bool, t1_only: bool = False):
        self.st = store
        self.augment = augment
        self.t1_only = t1_only  # ablation control: current frame only

    def __len__(self):
        return len(self.st.index)

    def window_indices(self, T: int, t: int) -> list[int]:
        if self.t1_only:
            return [t]
        if self.augment:
            stride = random.choice((1, 2))
            idx = [t - k * stride for k in range(WIN - 1, 0, -1)] + [t]
            idx = [i for i in idx if i >= 0]
            # drop past frames with p=0.1 (never the current frame)
            idx = [i for i in idx[:-1] if random.random() >= 0.1] + [t]
        else:
            idx = list(range(max(0, t - (WIN - 1)), t + 1))
        return idx

    def __getitem__(self, k):
        si, t = self.st.index[k]
        s = self.st.seqs[si]
        idx = self.window_indices(len(s["smoke"]), t)
        n = len(idx)
        feat = torch.zeros((WIN, N_CH, *HW), dtype=torch.float32)
        mask = torch.zeros(WIN, dtype=torch.float32)
        dt = torch.zeros(WIN, dtype=torch.float32)
        feat[WIN - n:] = torch.from_numpy(s["feat"][idx].astype(np.float32))
        mask[WIN - n:] = 1.0
        ts = s["ts"][idx]
        for j in range(1, n):
            dt[WIN - n + j] = min((ts[j] - ts[j - 1]) / 60.0, 10.0)
        return {
            "feat": feat, "mask": mask, "dt": dt,
            "cur_score": torch.tensor(s["score"][t], dtype=torch.float32),
            "label": torch.tensor(s["smoke"][t], dtype=torch.float32),
        }


# ---------------------------------------------------------------- model
class ConvGRUCell(nn.Module):
    def __init__(self, cin: int, ch: int):
        super().__init__()
        self.zr = nn.Conv2d(cin + ch, 2 * ch, 3, padding=1)
        self.hn = nn.Conv2d(cin + ch, ch, 3, padding=1)
        self.ch = ch

    def forward(self, x, h):
        zr = torch.sigmoid(self.zr(torch.cat([x, h], 1)))
        z, r = zr.split(self.ch, dim=1)
        n = torch.tanh(self.hn(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class TemporalHead(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(N_CH, ch, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
        )
        self.cell = ConvGRUCell(ch + 1, ch)  # +1 dt channel
        self.head = nn.Sequential(
            nn.Linear(2 * ch + 1, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.ch = ch

    def forward(self, feat, mask, dt, cur_score):
        # feat (B,WIN,C,H,W); mask/dt (B,WIN); cur_score (B,)
        B = feat.shape[0]
        h = feat.new_zeros((B, self.ch, *HW))
        for j in range(feat.shape[1]):
            x = self.enc(feat[:, j])
            dch = dt[:, j].view(B, 1, 1, 1).expand(B, 1, *HW)
            hj = self.cell(torch.cat([x, dch], 1), h)
            m = mask[:, j].view(B, 1, 1, 1)
            h = m * hj + (1 - m) * h
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(torch.cat([pooled, cur_score[:, None]], dim=1))[:, 0]


# ---------------------------------------------------------------- eval
@torch.no_grad()
def score_split(model, store, device, bs=512, t1_only=False):
    model.eval()
    ds = WindowDataset(store, augment=False, t1_only=t1_only)
    dl = DataLoader(ds, batch_size=bs, num_workers=4, shuffle=False)
    logits, labels = [], []
    for b in dl:
        out = model(b["feat"].to(device), b["mask"].to(device),
                    b["dt"].to(device), b["cur_score"].to(device))
        logits.append(out.float().cpu().numpy())
        labels.append(b["label"].numpy())
    return np.concatenate(logits), np.concatenate(labels)


def average_precision(scores, labels):
    order = np.argsort(-scores, kind="stable")
    lab = labels[order]
    tp = np.cumsum(lab)
    prec = tp / (np.arange(len(lab)) + 1)
    return float((prec * lab).sum() / max(lab.sum(), 1))


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit-seqs", type=int, default=0,
                    help="tiny sanity: train+eval on first N train sequences")
    ap.add_argument("--t1-only", action="store_true",
                    help="ablation control: window = current frame only")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    froot = Path(args.features)

    tr_store = SeqStore(froot / "train", limit_seqs=args.overfit_seqs)
    va_store = tr_store if args.overfit_seqs else SeqStore(froot / "val")
    print(f"train: {len(tr_store.seqs)} seqs {len(tr_store.index)} frames "
          f"({tr_store.n_pos} pos / {tr_store.n_neg} neg); "
          f"val frames {len(va_store.index)}", flush=True)

    model = TemporalHead().to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    dl = DataLoader(WindowDataset(tr_store, augment=not args.overfit_seqs,
                                  t1_only=args.t1_only),
                    batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    steps_total = args.max_steps or args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps_total, 1))
    pos_weight = torch.tensor(tr_store.n_neg / max(tr_store.n_pos, 1),
                              dtype=torch.float32, device=args.device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start_ep, best_ap, step = 0, -1.0, 0
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best_ap, step = ck["epoch"] + 1, ck["best_ap"], ck["step"]
        print(f"resumed at epoch {start_ep} best_ap {best_ap:.4f}", flush=True)

    logf = (out / "train_log.jsonl").open("a")
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for b in dl:
            out_l = model(b["feat"].to(args.device), b["mask"].to(args.device),
                          b["dt"].to(args.device), b["cur_score"].to(args.device))
            loss = lossf(out_l, b["label"].to(args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ep_loss += loss.item()
            nb += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        logits, labels = score_split(model, va_store, args.device,
                                     t1_only=args.t1_only)
        ap_val = average_precision(logits, labels)
        rec = {"epoch": ep, "step": step, "train_loss": round(ep_loss / max(nb, 1), 5),
               "val_ap": round(ap_val, 5), "elapsed_s": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "step": step,
                    "best_ap": best_ap, "args": vars(args)}, last)
        if ap_val > best_ap:
            best_ap = ap_val
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_ap": ap_val, "n_params": n_par,
                        "args": vars(args)}, out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break
    (out / "done.json").write_text(json.dumps(
        {"best_val_ap": best_ap, "epochs": args.epochs, "params": n_par}))
    print(f"TEMPORAL_TRAIN_DONE best_val_ap={best_ap:.4f}", flush=True)


if __name__ == "__main__":
    main()
