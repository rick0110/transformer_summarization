#!/usr/bin/env python3
import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import List

from train import train
from tokenizers import Tokenizer


def load_cfg(cfg_path: str):
    """Load cfg from JSON file or from a python file that defines a `cfg` dict/object.

    Supported:
    - JSON file with keys -> returns SimpleNamespace
    - Python file that when executed defines a `cfg` variable (dict or SimpleNamespace)
    """
    if cfg_path.endswith(".json"):
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        return SimpleNamespace(**data)

    # try to import as python file
    if cfg_path.endswith(".py"):
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        sys.path.insert(0, cfg_dir)
        mod = __import__(cfg_name)
        # prefer object named cfg
        if hasattr(mod, "cfg"):
            c = getattr(mod, "cfg")
            if isinstance(c, dict):
                return SimpleNamespace(**c)
            return c
        # fallback: find first SimpleNamespace or dict called CONFIG
        if hasattr(mod, "CONFIG"):
            c = getattr(mod, "CONFIG")
            if isinstance(c, dict):
                return SimpleNamespace(**c)
            return c
        raise RuntimeError(f"Python cfg {cfg_path} must define variable 'cfg' or 'CONFIG'")

    raise RuntimeError("cfg must be a .json or .py file")


def parse_text_columns(text_cols: str) -> List[str]:
    if not text_cols:
        return []
    return [c.strip() for c in text_cols.split(",") if c.strip()]


def main():
    p = argparse.ArgumentParser(description="Train transformer model (wrapper)")
    p.add_argument("--data", required=True, help="Path to dataset (.json or .csv)")
    p.add_argument("--tokenizer", required=True, help="Path to tokenizer json (from tokenizers)")
    p.add_argument("--cfg", required=True, help="Path to cfg file (.json or .py)")
    p.add_argument("--out", required=True, help="Output directory for checkpoints")
    p.add_argument("--text-cols", default="", help="Comma-separated text columns (for CSV/JSON). E.g. 'title,description'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--tb-logdir", default=None, help="TensorBoard logdir (optional)")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from (optional)")

    args = p.parse_args()

    text_columns = parse_text_columns(args.text_cols)
    cfg = load_cfg(args.cfg)

    # ensure tokenizer exists
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(args.tokenizer)

    # fill minimal cfg fields if missing
    if not hasattr(cfg, "vocab_size"):
        cfg.vocab_size = Tokenizer.from_file(args.tokenizer).get_vocab_size()
    if not hasattr(cfg, "max_len"):
        cfg.max_len = args.max_len
    if not hasattr(cfg, "padding_idx"):
        cfg.padding_idx = Tokenizer.from_file(args.tokenizer).token_to_id("[PAD]")

    train(
        cfg,
        dataset_path=args.data,
        tokenizer_path=args.tokenizer,
        text_columns=text_columns,
        output_dir=args.out,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_len=args.max_len,
        tb_log_dir=args.tb_logdir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
