import re

with open("checkpoints/ecapa_tdnn_v2/train.log", encoding="utf-8") as f:
    content = f.read()

lines = content.splitlines()
train_losses = {}
dev_losses = {}
for line in lines:
    m = re.search(r"\[Dev\].*Epoch (\d+)/200, Dev Loss: ([-\d.]+)", line)
    if m:
        ep, loss = int(m.group(1)), float(m.group(2))
        dev_losses[ep] = loss
    m = re.search(r"\[Train\] Epoch (\d+) finished.*Average Train Loss: ([-\d.]+)", line)
    if m:
        ep, loss = int(m.group(1)), float(m.group(2))
        train_losses[ep] = loss

print("Epoch | Train Loss | Dev Loss |  Gap (overfit)")
for ep in sorted(train_losses)[-20:]:
    tr = train_losses.get(ep, float("nan"))
    dv = dev_losses.get(ep, float("nan"))
    print(f"  {ep:3d} |   {tr:7.2f}  |  {dv:7.2f}  | {tr-dv:+.2f} dB")

best_ep = max(dev_losses, key=lambda e: -dev_losses[e])
last_ep = max(train_losses)
print()
print(f"Best epoch: {best_ep}, Dev={dev_losses[best_ep]:.4f} dB, Train={train_losses.get(best_ep, 'n/a'):.2f} dB")
print(f"Last epoch: {last_ep}, Dev={dev_losses.get(last_ep, 'n/a'):.4f} dB, Train={train_losses[last_ep]:.2f} dB")
