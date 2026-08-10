"""GPU smoke test — runs one full train step on a SYNTHETIC batch (no corpus).
Catches torch/cuda/numpy incompatibilities before the real run.
Run:  PYTHONPATH=masked_encoder python3 masked_encoder/test_gpu.py
"""
import torch
import torch.nn as nn
import vectoriser as V
import train as T

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| torch:", torch.__version__)
vocab = V.load_vocabs()
inv_e = {i: n for n, i in vocab["entity"].items()}

model = T.TileEncoder(vocab, d=256, layers=6, heads=8).to(dev)
print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# synthetic batch: randomize the planes the model actually reads, keep in-range
B, K = 16, len(V.PLANES)
x = torch.zeros(B, K, V.W, V.W, dtype=torch.int16)
x[:, T.PIDX["role"]] = torch.randint(0, 6, (B, V.W, V.W), dtype=torch.int16)
x[:, T.PIDX["entity"]] = torch.randint(0, len(vocab["entity"]) + 1, (B, V.W, V.W), dtype=torch.int16)
x[:, T.PIDX["direction"]] = torch.randint(0, 17, (B, V.W, V.W), dtype=torch.int16)
x[:, T.PIDX["recipe"]] = torch.randint(0, len(vocab["recipe"]) + 1, (B, V.W, V.W), dtype=torch.int16)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
lossf = nn.CrossEntropyLoss()
for step in range(3):
    xm, orig, m = T.mask_batch(x, inv_e)
    xm, orig, m = xm.to(dev).long(), orig.to(dev).long(), m.to(dev)
    logits = model(xm)
    loss = sum(lossf(logits[f][m], orig[:, T.PIDX[f]][m]) for f in model.fields)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"step {step}  masked_tiles {int(m.sum())}  loss {float(loss):.2f}")

if dev == "cuda":
    print("peak GPU mem (MB):", int(torch.cuda.max_memory_allocated() / 1e6))
print("GPU SMOKE OK")
