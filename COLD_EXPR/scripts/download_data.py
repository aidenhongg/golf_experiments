#!/usr/bin/env python3
"""Download and prepare FineWeb data for cold experiments.

Usage:
    python scripts/download_data.py --variant sp8192
    python scripts/download_data.py --variant sp8192 --val-only
    python scripts/download_data.py --variant sp16384 --tokenizer-only
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


DATA_ROOT = Path("data")
VARIANTS = {
    "sp8192": {
        "vocab_size": 8192,
        "data_dir": "fineweb_sp8192",
        "tokenizer": "fineweb_8192_bpe",
    },
    "sp16384": {
        "vocab_size": 16384,
        "data_dir": "fineweb_sp16384",
        "tokenizer": "fineweb_16384_bpe",
    },
}

PARAMETER_GOLF_REPO = "https://github.com/openai/parameter-golf.git"
FINEWEB_REPO_ID = "kevclark/parameter-golf"


def ensure_dirs():
    (DATA_ROOT / "tokenizers").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "fineweb_sp8192").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "fineweb_sp16384").mkdir(parents=True, exist_ok=True)


def clone_parameter_golf(dest="/tmp/parameter-golf"):
    if os.path.exists(dest):
        print(f"  parameter-golf repo already at {dest}")
        return dest
    print(f"  Cloning {PARAMETER_GOLF_REPO} -> {dest}")
    subprocess.run(["git", "clone", "--depth=1", PARAMETER_GOLF_REPO, dest], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", f"{dest}/requirements.txt", "-q"], check=True)
    return dest


def download_sp8192(repo_dir, val_only=False):
    """Download tokenized FineWeb SP8192 using parameter-golf's data script."""
    variant = VARIANTS["sp8192"]
    data_dir = DATA_ROOT / variant["data_dir"]
    tok_path = DATA_ROOT / "tokenizers" / f"{variant['tokenizer']}.model"

    if list(data_dir.glob("*.bin")) and tok_path.exists():
        print(f"  SP8192 data already exists at {data_dir} ({len(list(data_dir.glob('*.bin')))} shards)")
        return

    print("  Downloading tokenized FineWeb (SP8192)...")
    env = os.environ.copy()
    env["MATCHED_FINEWEB_REPO_ID"] = FINEWEB_REPO_ID

    script = os.path.join(repo_dir, "data", "cached_challenge_fineweb.py")
    if os.path.exists(script):
        cmd = [sys.executable, script, "--variant", "sp8192"]
        subprocess.run(cmd, env=env, check=True, cwd=repo_dir)
        # Move data to our expected paths
        pg_data = Path(repo_dir) / "data" / "datasets" / "fineweb10B_sp8192"
        pg_tok = Path(repo_dir) / "data" / "tokenizers" / "fineweb_8192_bpe.model"
        if pg_data.exists():
            for shard in pg_data.glob("*.bin"):
                dest = data_dir / shard.name
                if not dest.exists():
                    os.symlink(shard.resolve(), dest)
            print(f"  Linked {len(list(data_dir.glob('*.bin')))} shards -> {data_dir}")
        if pg_tok.exists() and not tok_path.exists():
            os.symlink(pg_tok.resolve(), tok_path)
            print(f"  Linked tokenizer -> {tok_path}")
    else:
        print(f"  ERROR: {script} not found. Download data manually.")
        print(f"  Expected: .bin shard files in {data_dir}/")
        print(f"            SentencePiece model at {tok_path}")
        sys.exit(1)


def train_sp16384_tokenizer(raw_text_path=None):
    """Train SP16384 tokenizer (required for C21 only)."""
    variant = VARIANTS["sp16384"]
    tok_path = DATA_ROOT / "tokenizers" / f"{variant['tokenizer']}.model"

    if tok_path.exists():
        print(f"  SP16384 tokenizer already exists at {tok_path}")
        return

    if raw_text_path is None:
        print("  SP16384 tokenizer not found. To create it:")
        print(f"    python scripts/train_tokenizer.py --input <raw_text.txt> --prefix {tok_path.with_suffix('')} --vocab-size 16384")
        print("  Skipping C21 data prep (optional).")
        return

    print(f"  Training SP16384 tokenizer from {raw_text_path}...")
    subprocess.run([
        sys.executable, "scripts/train_tokenizer.py",
        "--input", raw_text_path,
        "--prefix", str(tok_path.with_suffix("")),
        "--vocab-size", "16384",
    ], check=True)


def verify():
    """Check all expected files exist."""
    print("\n=== Data Verification ===")
    ok = True

    # SP8192 data
    sp8k_dir = DATA_ROOT / "fineweb_sp8192"
    shards = list(sp8k_dir.glob("*.bin"))
    if shards:
        total_mb = sum(s.stat().st_size for s in shards) / 1e6
        print(f"  [OK] SP8192 data: {len(shards)} shards, {total_mb:.0f} MB")
    else:
        print(f"  [MISSING] SP8192 data: no .bin files in {sp8k_dir}")
        ok = False

    # SP8192 tokenizer
    tok = DATA_ROOT / "tokenizers" / "fineweb_8192_bpe.model"
    if tok.exists():
        print(f"  [OK] SP8192 tokenizer: {tok}")
    else:
        print(f"  [MISSING] SP8192 tokenizer: {tok}")
        ok = False

    # SP16384 tokenizer (optional)
    tok16 = DATA_ROOT / "tokenizers" / "fineweb_16384_bpe.model"
    if tok16.exists():
        print(f"  [OK] SP16384 tokenizer: {tok16}")
    else:
        print(f"  [SKIP] SP16384 tokenizer not found (only needed for C21)")

    # SP16384 data (optional)
    sp16k_dir = DATA_ROOT / "fineweb_sp16384"
    shards16 = list(sp16k_dir.glob("*.bin"))
    if shards16:
        print(f"  [OK] SP16384 data: {len(shards16)} shards")
    else:
        print(f"  [SKIP] SP16384 data not found (only needed for C21)")

    if ok:
        print("\n  Ready for phases 0-6. Run: python scripts/run_all_cold.py --phases 0")
    else:
        print("\n  MISSING required data. See above.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download and prepare FineWeb data")
    parser.add_argument("--variant", choices=["sp8192", "sp16384", "all"], default="sp8192")
    parser.add_argument("--val-only", action="store_true", help="Download validation data only")
    parser.add_argument("--tokenizer-only", action="store_true", help="Only train/download tokenizer")
    parser.add_argument("--raw-text", default=None, help="Raw text file for SP16384 tokenizer training")
    parser.add_argument("--repo-dir", default="/tmp/parameter-golf", help="Path to parameter-golf repo clone")
    parser.add_argument("--verify-only", action="store_true", help="Only verify data exists")
    args = parser.parse_args()

    ensure_dirs()

    if args.verify_only:
        verify()
        return

    if args.variant in ("sp8192", "all"):
        print("=== SP8192 Data ===")
        repo = clone_parameter_golf(args.repo_dir)
        download_sp8192(repo, val_only=args.val_only)

    if args.variant in ("sp16384", "all"):
        print("=== SP16384 Data ===")
        train_sp16384_tokenizer(args.raw_text)

    verify()


if __name__ == "__main__":
    main()
