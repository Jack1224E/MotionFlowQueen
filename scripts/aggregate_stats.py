import sys
lines = open('validation_output/stats.txt').readlines()[6:]
errs, costs, mags = [], [], []
for l in lines:
    parts = l.split()
    if len(parts) >= 5:
        errs.append(float(parts[1]))
        costs.append(float(parts[3]))
        mags.append(float(parts[4]))
n = len(errs)
print(f"=== AGGREGATE STATS ({n} frames) ===")
print(f"Mean Warp Error:   avg={sum(errs)/n:.4f}, min={min(errs):.4f}, max={max(errs):.4f}")
print(f"Cost<=4 pct:       avg={sum(costs)/n:.1f}%, min={min(costs):.1f}%, max={max(costs):.1f}%")
print(f"Flow Magnitude:    avg={sum(mags)/n:.2f}px, min={min(mags):.2f}px, max={max(mags):.2f}px")
