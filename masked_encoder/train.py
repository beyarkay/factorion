"""Green-fields PoC: masked-tile pretraining of a small 2D transformer over
Factorio blueprint tensors. Standalone. Goal: prove the pipeline runs and the
model can learn to predict masked entities (BERT-style MLM on tiles).

Run:  python3 masked_encoder/train.py
"""
import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb

import vectoriser as V  # same dir

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
W = V.W
PIDX = {p: i for i, p in enumerate(V.PLANES)}
OCCUPIED = (V.ROLE["anchor"], V.ROLE["body"], V.ROLE["rail"])
MASK_FRAC = 0.15

# ---------------------------------------------------------------- data cache
def build_cache(n=4000):
    path = os.path.join(HERE, f"cache_{n}.npy")
    if os.path.exists(path):
        return np.load(path)
    print(f"building tensor cache ({n} designs)...")
    vocab = V.load_vocabs()
    lines = open(os.path.join(V.BP, "v1_clean.individual.txt")).read().splitlines()
    out = []
    for s in lines:
        if len(out) >= n:
            break
        bp = V.decode(s)
        if bp is None:
            continue
        t = V.vectorise(bp, vocab)
        if t is None:
            continue
        out.append(t)
    arr = np.stack(out).astype(np.int16)
    np.save(path, arr)
    print(f"  cache {arr.shape} -> {path}")
    return arr


# ---------------------------------------------------------------- model
class TileEncoder(nn.Module):
    def __init__(self, vocab, d=64, layers=2, heads=4):
        super().__init__()
        NE = len(vocab["entity"]) + 1
        NR = len(vocab["recipe"]) + 1
        NI = len(vocab["item"]) + 1
        NF = len(vocab["floor"]) + 1
        self.emb = nn.ModuleDict({
            "role": nn.Embedding(6, d), "entity": nn.Embedding(NE, d),
            "direction": nn.Embedding(17, d), "floor": nn.Embedding(NF, d),
            "utype": nn.Embedding(3, d), "in_prio": nn.Embedding(3, d), "out_prio": nn.Embedding(3, d),
            "recipe": nn.Embedding(NR, d), "body_dx": nn.Embedding(W, d), "body_dy": nn.Embedding(W, d),
            "item": nn.Embedding(NI, d),
        })
        self.pos = nn.Parameter(torch.zeros(1, W * W, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        # predict the WHOLE entity spec at masked tiles, one head per masked field
        self.fields = ["entity", "direction", "recipe", "type", "in_prio", "out_prio", "body_dx", "body_dy"] \
            + [f"mod{i}" for i in range(V.MOD_SLOTS)] + [f"filt{i}" for i in range(V.FILT_SLOTS)]
        card = {"entity": NE, "direction": 17, "recipe": NR, "type": 3, "in_prio": 3,
                "out_prio": 3, "body_dx": W, "body_dy": W}
        for i in range(V.MOD_SLOTS):
            card[f"mod{i}"] = NI
        for i in range(V.FILT_SLOTS):
            card[f"filt{i}"] = NI
        self.hkey = {f: "h_" + f for f in self.fields}   # 'h_' prefix dodges reserved names (e.g. 'type')
        self.heads = nn.ModuleDict({self.hkey[f]: nn.Linear(d, card[f]) for f in self.fields})

    def forward(self, x):  # x: (B, K, W, W) long
        def pl(name):
            return x[:, PIDX[name]]
        e = (self.emb["role"](pl("role")) + self.emb["entity"](pl("entity"))
             + self.emb["direction"](pl("direction")) + self.emb["floor"](pl("floor"))
             + self.emb["utype"](pl("type")) + self.emb["in_prio"](pl("in_prio"))
             + self.emb["out_prio"](pl("out_prio")) + self.emb["recipe"](pl("recipe"))
             + self.emb["body_dx"](pl("body_dx")) + self.emb["body_dy"](pl("body_dy")))
        for i in range(V.MOD_SLOTS):
            e = e + self.emb["item"](pl(f"mod{i}"))
        for i in range(V.FILT_SLOTS):
            e = e + self.emb["item"](pl(f"filt{i}"))
        B = e.shape[0]
        e = e.view(B, W * W, -1) + self.pos
        h = self.tr(e).view(B, W, W, -1)
        return {f: self.heads[self.hkey[f]](h) for f in self.fields}


# ---------------------------------------------------------------- masking
BLANK = ["entity", "direction", "recipe", "type", "in_prio", "out_prio", "body_dx", "body_dy"] \
    + [f"mod{i}" for i in range(V.MOD_SLOTS)] + [f"filt{i}" for i in range(V.FILT_SLOTS)]


def mask_batch(x, inv_e):
    """Atomic-entity masking: pick ~MASK_FRAC of ENTITIES and blank all of their
    tiles together (anchor + body) so a masked body cell can't copy its still-
    visible anchor. Rails (an occupancy blob, no anchor/body) are masked per-tile.
    orig keeps every field so we predict the WHOLE entity spec at masked tiles.
    The index-heavy part runs in numpy (no torch .item() syncs) so it doesn't
    bottleneck a fast GPU."""
    orig = x.clone()
    x = x.clone()
    role = x[:, PIDX["role"]].numpy()
    ent = x[:, PIDX["entity"]].numpy()
    drc = x[:, PIDX["direction"]].numpy()
    m = np.zeros(role.shape, dtype=bool)
    for b in range(role.shape[0]):
        ays, axs = np.where(role[b] == V.ROLE["anchor"])
        for y, xx in zip(ays, axs):
            if np.random.rand() >= MASK_FRAC:
                continue
            name = inv_e.get(int(ent[b, y, xx]))
            if name is None:
                m[b, y, xx] = True
                continue
            w, h = V.footprint(name, int(drc[b, y, xx]) - 1)
            m[b, y:y + h, xx:xx + w] = True          # whole footprint, atomically
        rys, rxs = np.where(role[b] == V.ROLE["rail"])
        rsel = np.random.rand(len(rys)) < MASK_FRAC
        m[b, rys[rsel], rxs[rsel]] = True
    m = torch.from_numpy(m)
    for name in BLANK:
        x[:, PIDX[name]][m] = 0
    x[:, PIDX["role"]][m] = V.ROLE["mask"]
    return x, orig, m


# ---------------------------------------------------------------- eval
def evaluate(model, val, bs, inv_e):
    """Per-field masked accuracy + accuracy EXCLUDING the `none` majority class
    (target != 0). Chunked — the full val set at once OOMs the 2500-token attention."""
    fields = model.fields   # score EVERY dimension, not just entity
    model.eval()
    corr = {f: 0 for f in fields}; corr_nn = {f: 0 for f in fields}
    tot_nn = {f: 0 for f in fields}; total = 0
    loss_sum = {f: 0.0 for f in fields}   # summed CE over all masked tiles (reduction='sum')
    with torch.no_grad():
        for i in range(0, len(val), bs):
            x, orig, m = mask_batch(val[i:i + bs], inv_e)
            x, orig, m = x.to(DEV).long(), orig.to(DEV).long(), m.to(DEV)
            logits = model(x)
            total += int(m.sum().item())
            for f in fields:
                tgt = orig[:, PIDX[f]][m]; hit = logits[f][m].argmax(-1) == tgt
                nn_ = tgt != 0
                corr[f] += int(hit.sum().item())
                corr_nn[f] += int((hit & nn_).sum().item())
                tot_nn[f] += int(nn_.sum().item())
                loss_sum[f] += nn.functional.cross_entropy(logits[f][m], tgt, reduction="sum").item()
    model.train()
    log = {"val/masked_tiles": total}
    for f in fields:
        log[f"val/acc/{f}"] = corr[f] / max(1, total)
        log[f"val/acc_excl_none/{f}"] = corr_nn[f] / max(1, tot_nn[f])
        log[f"val/loss/{f}"] = loss_sum[f] / max(1, total)
    log["val/acc"] = log["val/acc/entity"]
    log["val/acc_excl_none"] = float(np.mean([log[f"val/acc_excl_none/{f}"] for f in fields]))
    # summed per-field mean CE — the same quantity train/loss reports, so the curves overlay.
    log["val/loss"] = float(sum(log[f"val/loss/{f}"] for f in fields))
    return log


# ---------------------------------------------------------------- train
def main(steps=100000, bs=16, lr=3e-4, secs=1680, d=256, layers=6, heads=8,
         cache_n=60000, eval_every=250, ckpt="masked_encoder/encoder.pt", mask_frac=MASK_FRAC):
    global MASK_FRAC
    MASK_FRAC = mask_frac   # sweepable: mask_batch/evaluate read this global at call time
    torch.manual_seed(0); np.random.seed(0)
    vocab = V.load_vocabs()
    inv_e = {i: n for n, i in vocab["entity"].items()}
    data = torch.from_numpy(build_cache(cache_n))  # int16; per-batch slice cast only
    n_val = min(512, len(data) // 10)
    train, val = data[:-n_val], data[-n_val:]
    print(f"device={DEV} data={tuple(data.shape)} train={len(train)} val={len(val)} vocab={len(vocab['entity'])}")
    model = TileEncoder(vocab, d=d, layers=layers, heads=heads).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M  fields={len(model.fields)}")

    name = f"masked-d{d}L{layers}h{heads}-bs{bs}-c{cache_n//1000}k-atomic"
    wandb.init(project="masked-encoder", name=name, config={
        "d": d, "layers": layers, "heads": heads, "bs": bs, "lr": lr, "mask_frac": MASK_FRAC,
        "steps": steps, "secs": secs, "window": W, "cache_n": cache_n,
        "entity_vocab": len(vocab["entity"]), "masking": "atomic-entity",
        "fields": len(model.fields), "params": n_params})

    def field_loss(logits, orig, m):
        return sum(lossf(logits[f][m], orig[:, PIDX[f]][m]) for f in model.fields)

    t0 = time.time()
    for step in range(1, steps + 1):
        # cosine-anneal over the WALL-CLOCK budget: the run stops on `secs`, not `steps`
        # (steps is effectively unbounded), so a step-count schedule would never move.
        elapsed = time.time() - t0
        lr_now = lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, elapsed / secs)))
        for g in opt.param_groups:
            g["lr"] = lr_now
        idx = torch.randint(0, len(train), (bs,))
        x, orig, m = mask_batch(train[idx], inv_e)
        x, orig, m = x.to(DEV).long(), orig.to(DEV).long(), m.to(DEV)
        logits = model(x)
        loss = field_loss(logits, orig, m)
        opt.zero_grad(); loss.backward(); opt.step()
        # log RAW values only — smooth post-hoc (wandb smooths in-UI); don't bake in an EMA.
        # per-field acc AND acc excluding the `none` majority class (target != 0), like val.
        with torch.no_grad():
            tlog = {"train/loss": loss.item(), "lr": lr_now}
            accs = {}
            for f in model.fields:
                tgt = orig[:, PIDX[f]][m]; hit = logits[f][m].argmax(-1) == tgt
                nn_ = tgt != 0; n = int(nn_.sum().item())
                accs[f] = hit.float().mean().item()
                tlog[f"train/acc/{f}"] = accs[f]
                tlog[f"train/acc_excl_none/{f}"] = int((hit & nn_).sum().item()) / n if n else float("nan")
                tlog[f"train/loss/{f}"] = lossf(logits[f][m], tgt).item()
        tlog["train/acc"] = tlog["train/acc/entity"]
        tlog["train/acc_excl_none"] = float(np.nanmean([tlog[f"train/acc_excl_none/{f}"] for f in model.fields]))
        wandb.log(tlog, step=step)
        if step % 50 == 0 or step == 1:
            print(f"step {step:6d}  loss {loss.item():.3f}  entity {accs['entity']:.3f}  "
                  f"dir {accs['direction']:.3f}  recipe {accs['recipe']:.3f}  {time.time()-t0:.0f}s", flush=True)
        if step % eval_every == 0:
            vlog = evaluate(model, val, bs, inv_e)
            wandb.log(vlog, step=step)
            print(f"   [val@{step}] entity {vlog['val/acc/entity']:.3f}  dir {vlog['val/acc/direction']:.3f}  "
                  f"excl-none(mean) {vlog['val/acc_excl_none']:.3f}", flush=True)
        if time.time() - t0 > secs:
            print("time budget hit, stopping.")
            break

    vlog = evaluate(model, val, bs, inv_e)
    wandb.log(vlog, step=step)
    print("\nFINAL val (acc | acc-excl-none):")
    for f in model.fields:
        print(f"  {f:10} {vlog[f'val/acc/{f}']:.3f} | {vlog[f'val/acc_excl_none/{f}']:.3f}")
    torch.save({"model": model.state_dict(), "vocab": vocab,
                "config": {"d": d, "layers": layers, "heads": heads, "planes": V.PLANES}}, ckpt)
    print(f"saved encoder -> {ckpt}")
    wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in dict(steps=100000, bs=16, lr=3e-4, secs=1680, d=256, layers=6,
                     heads=8, cache_n=60000, eval_every=250, mask_frac=0.15).items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--ckpt", default="masked_encoder/encoder.pt")
    main(**vars(ap.parse_args()))
