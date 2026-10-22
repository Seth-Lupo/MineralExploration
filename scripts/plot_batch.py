"""Figures and summary table for the batched-assay experiment.

usage: python3 scripts/plot_batch.py   (reads data/batch/, writes docs/batch/)
"""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

DATA, OUT = "data/batch/", "docs/batch/"
COST = 150
BATCHES = [1, 2, 4, 8]
GRIDS = ["grid9", "grid16"]
rng = np.random.default_rng(0)

GRAPHITE, STEEL = "#1C1E20", "#5C6670"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                     "font.size": 10, "text.color": GRAPHITE, "axes.labelcolor": GRAPHITE,
                     "axes.edgecolor": STEEL, "xtick.color": STEEL, "ytick.color": STEEL,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.transparent": True})


def load(kind):
    rows = []
    for f in glob.glob(f"{DATA}{kind}_*.csv"):
        rows += [line.strip().split(",") for line in open(f) if line.strip()]
    return rows


# Combined raw results, committed next to the figures.
for kind, header in [("summary", "case,policy,B,ore,decision,holes,rounds,return"),
                     ("trace", "case,policy,B,holes,volume_mean,volume_std")]:
    rows = sorted(load(kind), key=lambda r: (r[1], int(r[2]), int(r[0]), int(r[3]) if kind == "trace" else 0))
    with open(f"{OUT}{kind}.csv", "w") as f:
        f.write(header + "\n" + "".join(",".join(r) + "\n" for r in rows))

summary = {}
for case, policy, B, ore, decision, holes, rounds, ret in load("summary"):
    key = policy if policy in GRIDS else int(B)
    summary.setdefault(key, {})[int(case)] = dict(
        ore=int(ore), mine=decision == "mine", holes=int(holes),
        rounds=1 if policy in GRIDS else int(rounds), ret=float(ret))

trace = {}
for case, policy, B, holes, mu, sd in load("trace"):
    key = policy if policy in GRIDS else int(B)
    trace.setdefault(key, {}).setdefault(int(case), []).append((int(holes), float(sd)))

keys = [k for k in BATCHES + GRIDS if k in summary]
cases = sorted(set.intersection(*(set(summary[k]) for k in keys)))


def column(key, f):
    return np.array([f(summary[key][c]) for c in cases], float)


def boot_ci(x, n=5000):
    means = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(means, [2.5, 97.5])


metrics = {
    "accuracy": lambda r: r["mine"] == (r["ore"] > COST),
    "holes": lambda r: r["holes"],
    "rounds": lambda r: r["rounds"],
    "return": lambda r: r["ret"],
}
profitable = column(1, lambda r: r["ore"] > COST).astype(bool)
ore_profit = column(1, lambda r: r["ore"])


def ore_recovered(key):
    mined = column(key, lambda r: r["mine"]).astype(bool)
    return (ore_profit * (mined & profitable)).sum() / (ore_profit * profitable).sum()


table = []
for k in keys:
    row = {"policy": f"B = {k}" if isinstance(k, int) else k.replace("grid", "grid ")}
    for name, f in metrics.items():
        x = column(k, f)
        row[name] = [x.mean(), *boot_ci(x)]
        if k != 1:  # paired difference vs fully sequential drilling
            d = x - column(1, f)
            row[name + "_delta"] = [d.mean(), *boot_ci(d)]
    row["ore_recovered"] = ore_recovered(k)
    table.append(row)
json.dump({"n_cases": len(cases), "n_profitable": int(profitable.sum()), "rows": table},
          open(f"{OUT}results.json", "w"), indent=1)

colors = dict(zip(BATCHES, plt.cm.viridis(np.linspace(0, 0.85, len(BATCHES)))))
colors.update(grid9="#A3A9AE", grid16=STEEL)


def label(k):
    return f"B = {k}" if isinstance(k, int) else k.replace("grid", "grid ")


# Fig 1: uncertainty left, per hole (money) and per lab round-trip (time). Belief only
# changes when a batch returns; a finished campaign keeps its final uncertainty.
def curve(k, x_of, xs):
    curves = []
    for c in cases:
        t = sorted(trace[k][c])
        sd0 = t[0][1]
        pts = [(x_of(i, h), sd / sd0) for i, (h, sd) in enumerate(t)]
        curves.append([[v for x, v in pts if x <= xx][-1] for xx in xs])
    curves = np.array(curves)
    lo, hi = np.array([boot_ci(curves[:, i], 1000) for i in range(len(xs))]).T
    return curves.mean(0), lo, hi


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, x_of, xs, xlab in [(axes[0], lambda i, h: h, np.arange(0, 21), "holes drilled (money)"),
                           (axes[1], lambda i, h: i, np.arange(0, 11), "lab round-trips (time)")]:
    for k in keys:
        if k in GRIDS and ax is axes[1]:  # a grid is one round: all holes, then all assays
            final = np.mean([sorted(trace[k][c])[-1][1] / sorted(trace[k][c])[0][1] for c in cases])
            ax.plot([0, 1, xs[-1]], [1, final, final], color=colors[k], ls="--", lw=1.2, label=label(k),
                    drawstyle="steps-post")
            continue
        mu, lo, hi = curve(k, x_of, xs)
        style = dict(ls="--", lw=1.2) if k in GRIDS else dict(lw=1.6)
        ax.plot(xs, mu, color=colors[k], label=label(k), drawstyle="steps-post", **style)
        ax.fill_between(xs, lo, hi, color=colors[k], alpha=0.12, lw=0, step="post")
    ax.set_xlabel(xlab)
    ax.set_xlim(0, xs[-1])
axes[0].set_ylabel("volume uncertainty  σ / σ₀")
axes[0].set_ylim(0, 1.05)
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}fig_uncertainty.svg")

# Fig 2: time vs money. Fewer lab round-trips, more holes.
fig, ax = plt.subplots(figsize=(4.6, 3.4))
offsets = {1: (6, 6), 2: (6, 6), 4: (6, 6), 8: (-34, 6), "grid9": (8, -10), "grid16": (8, -3)}
for k in keys:
    r, h = column(k, metrics["rounds"]), column(k, metrics["holes"])
    (rl, rh), (hl, hh) = boot_ci(r), boot_ci(h)
    ax.errorbar(r.mean(), h.mean(), xerr=[[r.mean() - rl], [rh - r.mean()]],
                yerr=[[h.mean() - hl], [hh - h.mean()]], fmt="s" if k in GRIDS else "o",
                color=colors[k], capsize=2, ms=5)
    ax.annotate(label(k), (r.mean(), h.mean()), textcoords="offset points", xytext=offsets[k], fontsize=8)
seq = [k for k in BATCHES if k in summary]
ax.plot([column(k, metrics["rounds"]).mean() for k in seq],
        [column(k, metrics["holes"]).mean() for k in seq], color="#bbbbbb", lw=1, zorder=0)
ax.set_xlabel("lab round-trips (time)")
ax.set_ylabel("holes drilled (money)")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.savefig(f"{OUT}fig_time_money.svg")

# Fig 3: does the final call suffer? Decision accuracy and value per case.
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
for ax, name, ylab in [(axes[0], "accuracy", "correct decisions"),
                       (axes[1], "return", "discounted return per case")]:
    for i, k in enumerate(keys):
        x = column(k, metrics[name])
        lo, hi = boot_ci(x)
        ax.errorbar(i, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt="s" if k in GRIDS else "o",
                    color=colors[k], capsize=2, ms=5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label(k).replace("B = ", "B=") for k in keys], fontsize=8)
    ax.set_ylabel(ylab)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.tight_layout()
fig.savefig(f"{OUT}fig_decisions.svg")

for row in table:
    print(row["policy"], {k: np.round(v, 3).tolist() for k, v in row.items() if k != "policy"})
