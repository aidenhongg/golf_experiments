#!/usr/bin/env python3
"""Generate comparison tables and plots from experiment results.
Usage: python scripts/analyze_results.py --results results/ --output results/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from golfcomp.experiments.analysis import ResultsAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze cold experiment results")
    parser.add_argument("--results", default="results/", help="Directory with experiment results")
    parser.add_argument("--output", default="results/", help="Output directory for reports")
    args = parser.parse_args()

    analyzer = ResultsAnalyzer()
    df = analyzer.load_all_results(args.results)
    if df.empty:
        print("No results found.")
        return

    df = analyzer.compare_to_baseline(df)

    # Save outputs
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    analyzer.save_comparison_csv(df, str(out / "comparison_table.csv"))
    analyzer.plot_loss_curves(df, args.output)

    promotions = analyzer.recommend_promotions(df)
    report = analyzer.generate_report(df)

    (out / "promotion_recommendations.md").write_text(report)

    print(report)
    print(f"\nPromoting {len(promotions)} experiments to hot.")
    for p in promotions:
        print(f"  -> {p['name']} ({p['category']}): {p['reason']}")


if __name__ == "__main__":
    main()
