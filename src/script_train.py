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
    raise NotImplementedError("Only JSON cfg files are supported currently.")

def main():
    p = argparse.ArgumentParser(description="Train transformer model (wrapper)")
    p.add_argument()

    args = p.parse_args("cfg", help="Path to cfg file (.json) for training")

    cfg = load_cfg(args.cfg)

    train(
        cfg,
        dataset_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        text_columns=[col.strip() for col in cfg.text_columns.split(",")],
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        max_len=cfg.max_len,
        tb_log_dir=cfg.tb_logdir,
        resume_from=cfg.resume_from,
        device=cfg.getattr('device', None),


    )


if __name__ == "__main__":
    main()
