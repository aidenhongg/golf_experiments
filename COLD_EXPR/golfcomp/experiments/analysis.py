"""Results analysis: compare experiments, rank, recommend promotions."""

import json
import os
from pathlib import Path

import pandas as pd

# Category inferred from experiment name prefix when not in summary
_NAME_TO_CATEGORY = {
    "cold_baseline": "baseline",
    **{f"c{i:02d}": "architecture" for i in range(1, 6)},
    **{f"c{i:02d}": "ablation" for i in range(6, 11)},
    **{f"c{i:02d}": "hp_sweep" for i in range(11, 15)},
    **{f"c{i:02d}": "quantization" for i in range(15, 18)},
    **{f"c{i:02d}": "eval_time" for i in range(18, 21)},
    "c21": "vocab",
}


def _infer_category(name: str) -> str:
    if name in _NAME_TO_CATEGORY:
        return _NAME_TO_CATEGORY[name]
    prefix = name.split("_")[0]
    return _NAME_TO_CATEGORY.get(prefix, "unknown")


class ResultsAnalyzer:
    """Compares experiment results and generates promotion recommendations."""

    PROMOTION_CRITERIA = {
        "architecture": {"throughput_gain": 0.05, "loss_tolerance": 0.01},
        "ablation": {"bpb_improvement": 0.002},
        "hp_sweep": {"differs_from_frontier": True},
        "quantization": {"artifact_reduction_kb": 200, "bpb_improvement": 0.001},
        "eval_time": {"bpb_improvement": 0.001},
    }

    def load_all_results(self, results_dir: str) -> pd.DataFrame:
        """Load all summary.json files into a DataFrame."""
        rows = []
        rdir = Path(results_dir)
        for p in sorted(rdir.rglob("summary.json")):
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            name = data.get("name") or data.get("config_name") or p.parent.name
            data["name"] = name
            if "category" not in data:
                data["category"] = _infer_category(name)
            rows.append(data)
        return pd.DataFrame(rows)

    def compare_to_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add delta_bpb, delta_throughput columns relative to baseline."""
        baseline = df[df["category"] == "baseline"]
        if baseline.empty:
            df["delta_bpb"] = float("nan")
            df["delta_throughput_pct"] = float("nan")
            return df
        b_bpb = baseline["bpb"].mean()
        b_tps = baseline["tokens_per_sec"].mean() if "tokens_per_sec" in baseline else 0
        df["delta_bpb"] = df["bpb"] - b_bpb
        df["delta_throughput_pct"] = (
            (df["tokens_per_sec"] - b_tps) / b_tps * 100 if b_tps > 0
            else float("nan")
        )
        return df

    def rank_by_category(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Rank experiments within each category by BPB."""
        out = {}
        for cat, grp in df.groupby("category"):
            out[cat] = grp.sort_values("bpb").reset_index(drop=True)
        return out

    def recommend_promotions(self, df: pd.DataFrame) -> list[dict]:
        """Apply promotion criteria. Return list of experiments to promote to hot."""
        if "delta_bpb" not in df.columns:
            df = self.compare_to_baseline(df)
        promotions = []
        for _, row in df.iterrows():
            cat = row.get("category", "")
            if cat == "baseline":
                continue
            reason = self._check_promotion(row, cat)
            if reason:
                promotions.append({"name": row["name"], "category": cat, "reason": reason,
                                   "bpb": row.get("bpb"), "delta_bpb": row.get("delta_bpb")})
        return promotions

    def _check_promotion(self, row, cat: str) -> str | None:
        if cat == "architecture":
            tp_gain = row.get("delta_throughput_pct", 0) or 0
            d_bpb = row.get("delta_bpb", 0) or 0
            if tp_gain >= 5.0 and d_bpb <= 0.01:
                return f"throughput +{tp_gain:.1f}%, bpb delta {d_bpb:+.4f}"
        elif cat == "ablation":
            d = row.get("delta_bpb", 0) or 0
            if d <= -0.002:
                return f"bpb improvement {d:+.4f}"
        elif cat == "hp_sweep":
            # Promote if BPB improved at all (value differs from frontier)
            d = row.get("delta_bpb", 0) or 0
            if d < 0:
                return f"optimal differs from frontier, bpb {d:+.4f}"
        elif cat == "quantization":
            d = row.get("delta_bpb", 0) or 0
            art_mb = row.get("artifact_mb", 999)
            # Baseline artifact ~16MB target; check reduction
            if d <= -0.001:
                return f"bpb improvement {d:+.4f}"
            # Can't check artifact_reduction_kb without baseline artifact, use bpb
        elif cat == "eval_time":
            d = row.get("delta_bpb", 0) or 0
            if d <= -0.001:
                return f"bpb improvement {d:+.4f}"
        return None

    def plot_loss_curves(self, df: pd.DataFrame, output_dir: str):
        """Generate loss-vs-tokens and loss-vs-wallclock plots per category."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plots")
            return

        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Per-category bar charts of BPB
        rankings = self.rank_by_category(df)
        for cat, grp in rankings.items():
            if grp.empty or "bpb" not in grp.columns:
                continue
            fig, ax = plt.subplots(figsize=(10, max(3, len(grp) * 0.5)))
            bars = ax.barh(grp["name"], grp["bpb"])
            ax.set_xlabel("BPB")
            ax.set_title(f"{cat} — BPB comparison")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(plots_dir / f"{cat}_bpb.png", dpi=100)
            plt.close(fig)

        # Scatter: BPB vs artifact_mb
        if "artifact_mb" in df.columns and "bpb" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            for cat, grp in df.groupby("category"):
                ax.scatter(grp["artifact_mb"], grp["bpb"], label=cat, s=40)
            ax.set_xlabel("Artifact MB")
            ax.set_ylabel("BPB")
            ax.set_title("BPB vs Artifact Size")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plots_dir / "bpb_vs_artifact.png", dpi=100)
            plt.close(fig)

        # Loss curves from metrics.csv if available
        rdir = Path(output_dir)
        fig_tok, ax_tok = plt.subplots(figsize=(10, 6))
        fig_wall, ax_wall = plt.subplots(figsize=(10, 6))
        has_curves = False
        for _, row in df.iterrows():
            metrics_path = rdir / row["name"] / "metrics.csv"
            if not metrics_path.exists():
                continue
            try:
                m = pd.read_csv(metrics_path)
            except Exception:
                continue
            if "loss" not in m.columns:
                continue
            has_curves = True
            label = row["name"]
            if "tokens_seen" in m.columns:
                ax_tok.plot(m["tokens_seen"], m["loss"], label=label, linewidth=0.8)
            if "wall_time_s" in m.columns:
                ax_wall.plot(m["wall_time_s"], m["loss"], label=label, linewidth=0.8)

        for ax, title, xlabel, fname in [
            (ax_tok, "Loss vs Tokens", "Tokens Seen", "loss_vs_tokens.png"),
            (ax_wall, "Loss vs Wall Clock", "Wall Time (s)", "loss_vs_wallclock.png"),
        ]:
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend(fontsize=6, ncol=2)
            ax.figure.tight_layout()

        if has_curves:
            fig_tok.savefig(plots_dir / "loss_vs_tokens.png", dpi=100)
            fig_wall.savefig(plots_dir / "loss_vs_wallclock.png", dpi=100)
        plt.close(fig_tok)
        plt.close(fig_wall)

    def generate_report(self, df: pd.DataFrame) -> str:
        """Markdown report with comparison table, rankings, promotions."""
        lines = ["# Cold Experiment Results\n"]

        # Summary table
        cols = ["name", "category", "bpb", "delta_bpb", "tokens_per_sec", "artifact_mb"]
        cols = [c for c in cols if c in df.columns]
        lines.append("## Summary\n")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.sort_values("bpb").iterrows():
            vals = []
            for c in cols:
                v = row.get(c)
                if isinstance(v, float):
                    vals.append(f"{v:.4f}" if "bpb" in c else f"{v:.2f}")
                else:
                    vals.append(str(v) if v is not None else "")
            lines.append("| " + " | ".join(vals) + " |")

        # Per-category rankings
        lines.append("\n## Rankings by Category\n")
        for cat, grp in self.rank_by_category(df).items():
            lines.append(f"### {cat}\n")
            for i, (_, row) in enumerate(grp.iterrows(), 1):
                bpb = row.get("bpb", "?")
                d = row.get("delta_bpb")
                delta_str = f" (delta: {d:+.4f})" if isinstance(d, (int, float)) and d == d else ""
                lines.append(f"{i}. **{row['name']}** — BPB: {bpb:.4f}{delta_str}")
            lines.append("")

        # Promotions
        promotions = self.recommend_promotions(df)
        lines.append("## Promotion Recommendations\n")
        if promotions:
            for p in promotions:
                lines.append(f"- **{p['name']}** ({p['category']}): {p['reason']}")
        else:
            lines.append("No experiments met promotion criteria.")

        # Concerns
        lines.append("\n## Concerns\n")
        concerns = []
        if "bpb" in df.columns:
            worst = df.nlargest(3, "bpb")
            for _, row in worst.iterrows():
                if row.get("delta_bpb", 0) and row["delta_bpb"] > 0.01:
                    concerns.append(f"- {row['name']}: BPB regression +{row['delta_bpb']:.4f}")
        if not concerns:
            concerns.append("- None identified.")
        lines.extend(concerns)

        return "\n".join(lines) + "\n"

    def save_comparison_csv(self, df: pd.DataFrame, path: str):
        """Save comparison_table.csv."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cols = [c for c in [
            "name", "category", "seed", "bpb", "delta_bpb", "final_loss",
            "tokens_per_sec", "delta_throughput_pct", "artifact_mb",
            "steps", "tokens_seen", "wall_time_s",
        ] if c in df.columns]
        df[cols].sort_values("bpb").to_csv(path, index=False)
