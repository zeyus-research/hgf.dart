"""Draw the benchmark figures in docs/ (each in a light and a dark SVG):

- benchmark.svg: single filters and the 1000 × 1000 batch, from the JSON
  that `dart run tool/bench_compare.dart` writes;
- benchmark_nuts.svg: NUTS across models, agents and chains, from
  `.dart_tool/nuts_sweep/{dart,python}.jsonl` (hgf_gpu's nuts_sweep.dart and
  tool/parity/nuts_sweep.py);
- benchmark_deep.svg: deep HGF networks across widths, from
  `.dart_tool/deep_sweep.json` and `.dart_tool/deep_sweep_rows.jsonl`
  (hgf_gpu's deep_sweep.dart and bench_deep.py --sweep).

    cd tool/parity && uv run python bench_plot.py

A figure whose data is missing is skipped. Colours follow the implementation
and device; f32 bars are hatched.
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Families -> categorical slot (validated order; light / dark steps).
FAMILIES = {
    "hgf.dart CPU": ("#2a78d6", "#3987e5"),
    "hgf.dart GPU": ("#eb6834", "#d95926"),
    "hgf.dart web": ("#1baf7a", "#199e70"),
    "pyhgf JAX CPU": ("#6250d6", "#9085e9"),
    "pyhgf JAX GPU": ("#eda100", "#c98500"),
    "rshgf / PyMC": ("#e87ba4", "#d55181"),
}

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e6e5e1"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#3a3a38"),
}

# backend key -> (label, family, f32)
SERIES = {
    "dart_aot": ("Dart AOT", "hgf.dart CPU", False),
    "dart_jit": ("Dart JIT", "hgf.dart CPU", False),
    "dart_wasm": ("Dart Wasm (Node)", "hgf.dart web", False),
    "dart_js": ("Dart JS (Node)", "hgf.dart web", False),
    "jax_jit_scan": ("JAX f64, jit scan", "pyhgf JAX CPU", False),
    "jax_jit_scan_f32": ("JAX f32, jit scan", "pyhgf JAX CPU", True),
    "rshgf_native": ("rshgf, native", "rshgf / PyMC", False),
    "rshgf": ("rshgf, via Python", "rshgf / PyMC", False),
}
BATCH = {
    "gpu_native_f32": ("hgf_gpu, wgpu-native", "hgf.dart GPU", True),
    "gpu_webgpu_f32": ("hgf_gpu, WebGPU in Chromium (Wasm)", "hgf.dart GPU", True),
    "dart_aot_batch_isolates": ("Dart AOT, BatchNetwork × 8 isolates", "hgf.dart CPU", False),
    "dart_aot_batch": ("Dart AOT, BatchNetwork, 1 core", "hgf.dart CPU", False),
    "dart_wasm_batch": ("Dart Wasm (Node), BatchNetwork", "hgf.dart web", False),
    "dart_js_batch": ("Dart JS (Node), BatchNetwork", "hgf.dart web", False),
    "jax_vmap": ("JAX f64, jit vmap", "pyhgf JAX CPU", False),
    "jax_vmap_f32": ("JAX f32, jit vmap", "pyhgf JAX CPU", True),
    "jax_vmap_gpu": ("JAX f64, jit vmap, GPU", "pyhgf JAX GPU", False),
    "jax_vmap_gpu_f32": ("JAX f32, jit vmap, GPU", "pyhgf JAX GPU", True),
    "rshgf_native_loop": ("rshgf, native loop", "rshgf / PyMC", False),
    "rshgf_loop": ("rshgf, via Python loop", "rshgf / PyMC", False),
}
PANELS = [
    ("continuous_3level_usdchf_614", "3-level continuous, USD/CHF (614 trials)", SERIES),
    ("continuous_3level_10k", "3-level continuous, 10k trials", SERIES),
    ("binary_3level_10k", "3-level binary, 10k trials", SERIES),
    ("batch_1000x1000_continuous_2level", "Batch: 1,000 ω × 1,000 trials, 2-level", BATCH),
]

# NUTS backends -> (label, family, f32)
NUTS = {
    "hgf_gpu": ("hgf_fit, GPU", "hgf.dart GPU", True),
    "hgf_pool8": ("hgf_fit, 8 isolates", "hgf.dart CPU", False),
    "hgf_cpu1": ("hgf_fit, 1 core", "hgf.dart CPU", False),
    "numpyro_cpu_f64": ("NumPyro, 1 core", "pyhgf JAX CPU", False),
    "numpyro_gpu_f32": ("NumPyro, GPU", "pyhgf JAX GPU", True),
    "pymc_cpu_f64": ("PyMC (wrap_jax), 1 core", "rshgf / PyMC", False),
}
NUTS_MODELS = {
    "b2": "binary 2-level",
    "b3": "binary 3-level",
    "c2": "continuous 2-level",
    "c3": "continuous 3-level",
}

# Deep backends -> (label, family, dashed)
DEEP = {
    "cpu": ("hgf_deep f64, 1 core", "hgf.dart CPU", False),
    "gpu": ("hgf_gpu f32", "hgf.dart GPU", False),
    "jax_cpu_f64": ("JAX f64, CPU", "pyhgf JAX CPU", False),
    "jax_cpu_f32": ("JAX f32, CPU", "pyhgf JAX CPU", True),
    "jax_gpu_f32": ("JAX f32, GPU", "pyhgf JAX GPU", False),
    "jax_gpu_f64": ("JAX f64, GPU", "pyhgf JAX GPU", True),
    "rshgf_f32": ("rshgf f32", "rshgf / PyMC", False),
}


def fmt(v: float) -> str:
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def style(theme: str) -> dict:
    t = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": t["text"],
            "axes.edgecolor": t["grid"],
            "axes.labelcolor": t["muted"],
            "xtick.color": t["muted"],
            "ytick.color": t["text"],
            "hatch.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )
    return t


def tidy(ax, t, grid_axis="x"):
    ax.set_facecolor(t["surface"])
    ax.grid(axis=grid_axis, which="major", color=t["grid"], linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", which="both", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])


def legend(fig, t, k, families, extra=(), y=0.955, rename=None):
    rename = rename or {}
    handles = [Patch(color=FAMILIES[f][k], label=rename.get(f, f)) for f in families] + list(extra)
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, y),
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.4,
    )


def f32_patch(t):
    return Patch(facecolor=t["muted"], edgecolor=t["surface"], hatch="////", label="f32 (others f64)")


def bars(ax, t, k, rows, xlabel, *, log=None, missing=None):
    """Ranked horizontal bars. rows: (label, family, f32, value or None)."""
    have = [r for r in rows if r[3] is not None]
    hi = max(r[3] for r in have)
    if log is None:
        log = hi / min(r[3] for r in have) > 20
    base = 10 ** math.floor(math.log10(min(r[3] for r in have) / 1.5)) if log else 0
    for i, (lab, fam, f32, v) in enumerate(rows):
        if v is None:
            ax.text(base * (1.06 if log else 1) + (0 if log else hi * 0.01), i, missing(lab) if missing else "n/a",
                    va="center", ha="left", color=t["muted"], fontsize=8, style="italic")
            continue
        c = FAMILIES[fam][k]
        ax.barh(i, v - base, left=base, height=0.62, color=c, edgecolor=t["surface"] if f32 else c,
                hatch="////" if f32 else None, linewidth=0, zorder=2)
        ax.text(v * 1.06 if log else v + hi * 0.012, i, fmt(v), va="center", ha="left", color=t["text"], fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
        ax.set_xlim(base, hi * 3)
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xlim(0, hi * 1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"{xlabel}{' (log scale)' if log else ''}", fontsize=8)
    tidy(ax, t)


def header(fig, t, title, subtitle):
    fig.text(0.01, 0.985, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.99, 0.985, subtitle, fontsize=8, color=t["muted"], va="top", ha="right")


def footer(fig, t, text):
    import textwrap

    lines = textwrap.wrap(text, int(fig.get_figwidth() * 17.5))
    fig.text(0.01, 0.006, "\n".join(lines), fontsize=7.5, color=t["muted"], va="bottom")
    return (0.08 + 0.135 * len(lines)) / fig.get_figheight()  # bottom margin for tight_layout


# ---------------------------------------------------------------------------


def draw_filters(values: dict, subtitle: str, theme: str, out: Path) -> None:
    t = style(theme)
    k = 0 if theme == "light" else 1
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), gridspec_kw=dict(height_ratios=[8, 12]), facecolor=t["surface"])
    used = set()
    for ax, (scenario, title, series) in zip(axes.flat, PANELS):
        rows = [
            (lab, fam, f32, values[f"{scenario}/{key}"])
            for key, (lab, fam, f32) in series.items()
            if f"{scenario}/{key}" in values
        ]
        rows.sort(key=lambda r: r[3])
        used |= {r[1] for r in rows}
        bars(ax, t, k, rows, "ms, lower is better")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=t["text"], pad=8)
    header(fig, t, "hgf.dart vs pyhgf (JAX) vs rshgf", subtitle)
    legend(fig, t, k, [f for f in FAMILIES if f in used], [f32_patch(t)], rename={"rshgf / PyMC": "rshgf"})
    gpu_series = [values.get(f"continuous_3level_10k/jax_jit_scan_gpu{s}") for s in ("", "_f32")]
    note = ""
    if any(gpu_series):
        note = (" JAX on the GPU is left out of the series panels: its scan launches kernels every trial "
                f"({' / '.join(fmt(v / 1e3) + ' s' for v in gpu_series if v)} on 10k trials; see the table).")
    bottom = footer(fig, t, "Median of warm runs. Series rows record trajectories. JAX excludes its one-off trace + compile "
           "(~0.4–0.8 s per model). GPU rows are end to end (ω upload, compute, readback)." + note)
    fig.tight_layout(rect=(0, bottom, 1, 0.93), h_pad=2.2, w_pad=2.5)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def draw_nuts(rows: list, subtitle: str, theme: str, out: Path) -> None:
    t = style(theme)
    k = 0 if theme == "light" else 1
    base = [r for r in rows if r["chains"] == 4]
    models = [m for m in NUTS_MODELS if any(r["model"] == m for r in base)]
    agents = sorted({r["agents"] for r in base})
    sweep = sorted({r["chains"] for r in rows if r["model"] == models[0] and r["agents"] == agents[0]})
    ncols = len(agents)
    fig = plt.figure(figsize=(11, 2.4 + 2.2 * len(models) + (3.0 if len(sweep) > 1 else 0)), facecolor=t["surface"])
    nrows = len(models) + (1 if len(sweep) > 1 else 0)
    gs = fig.add_gridspec(nrows, ncols, height_ratios=[1] * len(models) + ([1.25] if len(sweep) > 1 else []))
    used = set()
    for i, m in enumerate(models):
        for j, n in enumerate(agents):
            ax = fig.add_subplot(gs[i, j])
            cell = {r["backend"]: r for r in base if r["model"] == m and r["agents"] == n}
            items = []
            for key, (lab, fam, f32) in NUTS.items():
                r = cell.get(key)
                if r is None:
                    continue
                used.add(fam)
                rh = r.get("max_rhat")
                if rh is not None and rh > 1.05:
                    lab = f"{lab}  (R̂ {rh:.2f})"  # not converged: ESS/s means little
                items.append((lab, fam, f32, None if "timeout_s" in r else r["min_ess_per_s"], r))
            items.sort(key=lambda r: -(r[3] or -1))
            bars(ax, t, k, [it[:4] for it in items], "min bulk ESS per second, higher is better", log=True,
                 missing=lambda lab: f"over {next(it[4]['timeout_s'] for it in items if it[0] == lab) // 60} min")
            ax.set_ylim(len(NUTS) - 0.5, -0.5)  # same bar thickness in every panel
            ax.set_title(f"{NUTS_MODELS[m]}, {n} agents", loc="left", fontsize=10, fontweight="bold",
                         color=t["text"], pad=6)
    if len(sweep) > 1:
        ax = fig.add_subplot(gs[-1, :])
        m, n = models[0], agents[0]
        for key, (lab, fam, f32) in NUTS.items():
            pts = sorted((r["chains"], r["ms"] / 1e3) for r in rows
                         if r["model"] == m and r["agents"] == n and r["backend"] == key and "ms" in r)
            if len(pts) < 2:
                continue
            c = FAMILIES[fam][k]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=c, linewidth=2, linestyle="--" if key in ("hgf_cpu1",) else "-",
                    marker="o", markersize=6, markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
            ax.text(xs[-1] * 1.08, ys[-1], lab, va="center", fontsize=8, color=t["text"])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sweep, [str(c) for c in sweep])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(min(sweep) / 1.3, max(sweep) * 3.5)
        ax.set_xlabel("chains", fontsize=8)
        ax.set_ylabel("wall time, s (log)", fontsize=8)
        ax.set_title(f"Chains: {NUTS_MODELS[m]}, {n} agents, wall time", loc="left", fontsize=10,
                     fontweight="bold", color=t["text"], pad=6)
        tidy(ax, t, grid_axis="y")
    w = rows[0]["warmup"]
    d = rows[0]["draws"]
    header(fig, t, "NUTS: hgf_fit vs NumPyro vs PyMC", subtitle)
    legend(fig, t, k, [f for f in FAMILIES if f in used], [f32_patch(t)], y=1 - 0.45 / fig.get_figheight(),
           rename={"rshgf / PyMC": "PyMC"})
    bottom = footer(fig, t, f"4 chains × ({w} warmup + {d} draws), 320 trials per agent; ESS/s counts all sampling time, "
           "JAX compile excluded. hgf_fit samples each agent; NumPyro and PyMC sample all agents jointly "
           "(the same posterior). (R̂ …) marks runs that did not converge (max R̂ > 1.05): read their "
           "ESS/s with care.")
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.75 / fig.get_figheight()), h_pad=1.6, w_pad=2.5)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def draw_deep(problems: list, rows: list, subtitle: str, theme: str, out: Path) -> None:
    t = style(theme)
    k = 0 if theme == "light" else 1
    widths = [p["width"] for p in problems]
    series: dict[str, dict[str, dict[int, float]]] = {}
    for p in problems:
        for key, v in p["dart_ms"].items():
            dev, op = key.split("_", 1)
            series.setdefault(dev, {}).setdefault(op, {})[p["width"]] = v
    for r in rows:
        for op in ("fit", "batch_update", "predict"):
            series.setdefault(r["backend"], {}).setdefault(op, {})[r["width"]] = r[op]
    ops = [
        ("fit", f"fit, sequential ({problems[0].get('fit_samples', '?')} samples)"),
        ("batch_update", f"batchUpdate ({problems[0]['batch']} samples)"),
        ("predict", f"predict ({len(problems[0]['x'])} samples)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), facecolor=t["surface"])
    used = set()
    for ax, (op, title) in zip(axes, ops):
        for key, (lab, fam, dashed) in DEEP.items():
            pts = sorted(series.get(key, {}).get(op, {}).items())
            if not pts:
                continue
            used.add(fam)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=FAMILIES[fam][k], linewidth=2, linestyle="--" if dashed else "-",
                    marker="o", markersize=5, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(widths, [str(w) for w in widths])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("hidden width H (network [4, H, H, 16])", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=t["text"], pad=6)
        tidy(ax, t, grid_axis="y")
    axes[0].set_ylabel("ms (log), lower is better", fontsize=8)
    handles = [Line2D([], [], color=FAMILIES[fam][k], linewidth=2, linestyle="--" if dashed else "-", label=lab)
               for key, (lab, fam, dashed) in DEEP.items() if key in series]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=len(handles), frameon=False,
               fontsize=8, handlelength=2.2, columnspacing=1.2)
    header(fig, t, "Deep HGF networks: hgf_deep vs pyhgf (JAX) vs rshgf", subtitle)
    bottom = footer(fig, t, "tanh couplings, Adam; best of several runs. GPU and JAX rows include the readback to host memory. "
           "The GPU has no sequential fit (it fits ensembles: GpuDeepEnsemble).")
    fig.tight_layout(rect=(0, bottom, 1, 0.87), w_pad=2.0)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


# ---------------------------------------------------------------------------


def machine(md_path: Path) -> str:
    if not md_path.exists():
        return ""
    md = md_path.read_text()
    m = re.search(r"- Machine: (.+)", md)
    d = re.search(r"on (\d{4}-\d{2}-\d{2})", md)
    gpu = re.search(r"on (.+?), end to end", md)
    return ", ".join(
        s
        for s in [
            m and m.group(1).split(",")[0].replace("Intel(R) Core(TM) ", "").replace(" CPU", ""),
            gpu and gpu.group(1),
            d and d.group(1),
        ]
        if s
    )


def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def both(fn, out: Path, *args) -> None:
    fn(*args, "light", out)
    fn(*args, "dark", out.with_name(out.stem + "-dark.svg"))
    print(f"wrote {out} (+ dark)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=ROOT / ".dart_tool/hgf_bench/bench_all.json", type=Path)
    ap.add_argument("--nuts", default=ROOT / ".dart_tool/nuts_sweep", type=Path)
    ap.add_argument("--deep", default=ROOT / ".dart_tool/deep_sweep.json", type=Path)
    ap.add_argument("--out", default=ROOT / "docs", type=Path)
    a = ap.parse_args()
    subtitle = machine(a.out / "benchmark_results.md")
    if a.json.exists():
        values = {r["name"]: float(r["value"]) for r in json.loads(a.json.read_text())}
        both(draw_filters, a.out / "benchmark.svg", values, subtitle)
    nuts = jsonl(a.nuts / "dart.jsonl") + jsonl(a.nuts / "python.jsonl")
    if nuts:
        both(draw_nuts, a.out / "benchmark_nuts.svg", nuts, subtitle)
    if a.deep.exists():
        rows = jsonl(a.deep.with_name(a.deep.stem + "_rows.jsonl"))
        both(draw_deep, a.out / "benchmark_deep.svg", json.loads(a.deep.read_text()), rows, subtitle)


if __name__ == "__main__":
    main()
