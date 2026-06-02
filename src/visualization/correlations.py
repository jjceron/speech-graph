from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for visualization. Install with: py -m pip install matplotlib") from exc

from .common import append_manifest, correlation_csv, ensure_figures_dir, parse_csv_list, read_data_csv, safe_name


def _read_correlations(run_dir: Path) -> tuple[pd.DataFrame | None, Path | None]:
    path = correlation_csv(run_dir)
    if not path:
        return None, None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None, path
        if "abs_r" not in df.columns and "r" in df.columns:
            df["abs_r"] = pd.to_numeric(df["r"], errors="coerce").abs()
        return df, path
    except EmptyDataError:
        return None, path
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None, path


def plot_top_correlations(corr: pd.DataFrame, output_dir: Path, run_name: str, top_n: int = 25) -> list[dict]:
    required = {"metric", "target", "r"}
    if corr.empty or not required.issubset(corr.columns):
        return []
    corr = corr.copy()
    corr["r"] = pd.to_numeric(corr["r"], errors="coerce")
    corr["abs_r"] = corr["r"].abs()
    corr = corr.dropna(subset=["r"]).sort_values("abs_r", ascending=False).head(top_n)
    if corr.empty:
        return []

    labels = []
    for _, row in corr.iterrows():
        prefix = f"w{row['window_size']} | " if "window_size" in corr.columns else ""
        labels.append(f"{prefix}{row['metric']} ~ {row['target']}")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(corr))))
    y = list(range(len(corr)))
    ax.barh(y, corr["r"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("correlación r")
    ax.set_title(f"{run_name}: top {len(corr)} correlaciones")
    ax.invert_yaxis()
    fig.tight_layout()
    path = output_dir / "top_correlations_barplot.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return [{"figure": str(path), "kind": "top_correlations", "metric": "multiple"}]


def plot_correlation_heatmaps(corr: pd.DataFrame, output_dir: Path, run_name: str, top_n: int = 15) -> list[dict]:
    rows: list[dict] = []
    required = {"metric", "target", "r"}
    if corr.empty or not required.issubset(corr.columns):
        return rows

    corr = corr.copy()
    corr["r"] = pd.to_numeric(corr["r"], errors="coerce")
    corr["abs_r"] = corr["r"].abs()
    corr = corr.dropna(subset=["r"])
    if corr.empty:
        return rows

    groups = [("all", corr)]
    if "window_size" in corr.columns:
        groups = [(f"w{w}", sub.copy()) for w, sub in corr.groupby("window_size", dropna=False)]

    for label, sub in groups:
        top_metrics = (
            sub.groupby("metric")["abs_r"].max().sort_values(ascending=False).head(top_n).index.tolist()
        )
        top_targets = (
            sub.groupby("target")["abs_r"].max().sort_values(ascending=False).head(top_n).index.tolist()
        )
        pivot = sub[sub["metric"].isin(top_metrics) & sub["target"].isin(top_targets)].pivot_table(
            index="metric", columns="target", values="r", aggfunc="mean"
        )
        if pivot.empty:
            continue
        pivot = pivot.reindex(index=top_metrics, columns=top_targets)

        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(top_targets)), max(5, 0.35 * len(top_metrics))))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"{run_name}: mapa de correlaciones ({label})")
        fig.colorbar(im, ax=ax, label="r")
        fig.tight_layout()
        path = output_dir / f"correlation_heatmap_{safe_name(label)}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        rows.append({"figure": str(path), "kind": "correlation_heatmap", "metric": label})
    return rows


def plot_top_scatterplots(corr: pd.DataFrame, data: pd.DataFrame, output_dir: Path, run_name: str, top_n: int = 12) -> list[dict]:
    rows: list[dict] = []
    required = {"metric", "target", "r"}
    if corr.empty or data.empty or not required.issubset(corr.columns):
        return rows

    corr = corr.copy()
    corr["r"] = pd.to_numeric(corr["r"], errors="coerce")
    corr["abs_r"] = corr["r"].abs()
    corr = corr.dropna(subset=["r"]).sort_values("abs_r", ascending=False)

    made = 0
    seen: set[tuple] = set()
    for _, row in corr.iterrows():
        metric = row["metric"]
        target = row["target"]
        window_size = row.get("window_size", None)
        key = (metric, target, window_size)
        if key in seen or metric not in data.columns or target not in data.columns:
            continue
        seen.add(key)

        sub = data.copy()
        if "window_size" in data.columns and pd.notna(window_size):
            sub = sub[sub["window_size"].astype(str) == str(window_size)]
        x = pd.to_numeric(sub[metric], errors="coerce")
        y = pd.to_numeric(sub[target], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 4:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x[mask], y[mask], alpha=0.75)
        ax.set_xlabel(metric)
        ax.set_ylabel(target)
        w_text = f" | window={window_size}" if pd.notna(window_size) else ""
        ax.set_title(f"{run_name}: {metric} vs {target}{w_text}\nr={float(row['r']):.3f}")
        fig.tight_layout()
        suffix = f"_w{safe_name(window_size)}" if pd.notna(window_size) else ""
        path = output_dir / f"scatter_{made+1:02d}_{safe_name(metric)}_vs_{safe_name(target)}{suffix}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        rows.append({"figure": str(path), "kind": "scatter", "metric": metric, "target": target})
        made += 1
        if made >= top_n:
            break
    return rows


def generate_correlation_figures(run_dir: Path, top_n: int = 25, scatter_top_n: int = 12) -> list[dict]:
    corr, corr_path = _read_correlations(run_dir)
    if corr is None or corr.empty:
        print(f"[{run_dir.name}] no correlations CSV found")
        return []

    data, data_path = read_data_csv(run_dir)
    out_dir = ensure_figures_dir(run_dir, "correlations")
    rows: list[dict] = []
    rows.extend(plot_top_correlations(corr, out_dir, run_dir.name, top_n=top_n))
    rows.extend(plot_correlation_heatmaps(corr, out_dir, run_dir.name, top_n=min(15, top_n)))
    if data is not None:
        rows.extend(plot_top_scatterplots(corr, data, out_dir, run_dir.name, top_n=scatter_top_n))
    append_manifest(rows, run_dir)
    print(f"[{run_dir.name}] correlation figures: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create correlation figures from existing outputs.")
    parser.add_argument("--run-dir", default="outputs/01_windows_no_random")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--scatter-top-n", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_correlation_figures(Path(args.run_dir), top_n=args.top_n, scatter_top_n=args.scatter_top_n)


if __name__ == "__main__":
    main()
