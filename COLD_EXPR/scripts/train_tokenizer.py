#!/usr/bin/env python3
"""Train SP16384 tokenizer for C21."""
import sentencepiece as spm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw text file")
    parser.add_argument("--prefix", default="data/tokenizers/fineweb_16384_bpe")
    parser.add_argument("--vocab-size", type=int, default=16384)
    args = parser.parse_args()

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
    )
    print(f"Tokenizer saved to {args.prefix}.model")


if __name__ == "__main__":
    main()
