#!/usr/bin/env python3
"""Interactive inference script that streams tokens as they are generated.

Usage example:
    python src/script_infer.py --ckpt checkpoints/best_checkpoint.pth \
        --tokenizer tokenizadores/vagas_tokenizer_wp.json \
        --cfg config/cfg.json

If --ckpt is omitted the script will try to find the newest .pth under `models/` or the provided --model-dir.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Optional, List

import torch

from archtecture import GenerativeModel
from tokenizers import Tokenizer


def load_cfg(cfg_path: str):
    if cfg_path is None:
        return None
    if cfg_path.endswith(".json"):
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        return SimpleNamespace(**data)
    if cfg_path.endswith(".py"):
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        sys.path.insert(0, cfg_dir)
        mod = __import__(cfg_name)
        if hasattr(mod, "cfg"):
            c = getattr(mod, "cfg")
            if isinstance(c, dict):
                return SimpleNamespace(**c)
            return c
        if hasattr(mod, "CONFIG"):
            c = getattr(mod, "CONFIG")
            if isinstance(c, dict):
                return SimpleNamespace(**c)
            return c
        raise RuntimeError(f"Python cfg {cfg_path} must define variable 'cfg' or 'CONFIG'")
    raise RuntimeError("cfg must be a .json or .py file")


def find_latest_checkpoint(search_dirs: List[str]) -> Optional[str]:
    best = None
    best_mtime = 0
    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(".pth") or fn.endswith(".pt"):
                    path = os.path.join(root, fn)
                    m = os.path.getmtime(path)
                    if m > best_mtime:
                        best_mtime = m
                        best = path
    return best


def pick_stop_token(cfg: Optional[SimpleNamespace], tokenizer: Tokenizer) -> Optional[int]:
    # prefer cfg.eos_token_id if present
    if cfg is not None:
        for k in ("eos_token_id", "eos_id", "eos", "eos_token"):
            if hasattr(cfg, k):
                return getattr(cfg, k)
    # try common names
    candidates = ["</s>", "<eos>", "<pad>", "[EOS]", "[PAD]", "</EOS>"]
    for t in candidates:
        try:
            tid = tokenizer.token_to_id(t)
            if tid is not None and tid >= 0:
                # don't treat PAD as stop token ideally, but include if no other
                return tid
        except Exception:
            pass
    # fallback: None (we will stop at max_new_tokens)
    return None


def greedy_stream_generate(model: GenerativeModel, tokenizer: Tokenizer, device: torch.device, prompt: str, max_new_tokens: int = 128, stop_id: Optional[int] = None):
    model.eval()
    with torch.no_grad():
        enc = tokenizer.encode(prompt)
        input_ids = enc.ids
        # ensure numpy->list
        ids = list(input_ids)
        # print prompt first (optional)
        # print(prompt, end="\n")

        for step in range(max_new_tokens):
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_tensor)  # (1, seq_len, vocab)
            # take last token logits
            last = logits[:, -1, :]
            next_id = int(torch.argmax(last, dim=-1).item())
            ids.append(next_id)

            # stream token text
            try:
                txt = tokenizer.decode([next_id])
            except Exception:
                # fallback: print id
                txt = f"[{next_id}]"
            # print token without newline and flush
            print(txt, end="", flush=True)

            if stop_id is not None and next_id == stop_id:
                break
        print()  # final newline
        return ids


def main():
    p = argparse.ArgumentParser(description="Interactive streaming inference")
    p.add_argument("--ckpt", default=None, help="Checkpoint (.pth) to load. If omitted will search --model-dir or models/ for latest.")
    p.add_argument("--model-dir", default="models", help="Folder to search for checkpoints if --ckpt omitted.")
    p.add_argument("--cfg", default=None, help="Path to cfg json/py (optional). If provided will be used to instantiate the model.)")
    p.add_argument("--tokenizer", required=True, help="Path to tokenizer json")
    p.add_argument("--device", default=None, help="cpu or cuda (default auto)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-stream-prompt", action="store_true", help="Don't print the initial prompt before streaming tokens")

    args = p.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    tokenizer = Tokenizer.from_file(args.tokenizer)

    cfg = None
    if args.cfg:
        cfg = load_cfg(args.cfg)

    ckpt = args.ckpt
    if not ckpt:
        ckpt = find_latest_checkpoint([args.model_dir, "models", "checkpoints"])
        if not ckpt:
            print("No checkpoint provided and none found in models/checkpoints. Use --ckpt.")
            sys.exit(1)
    print(f"Loading checkpoint: {ckpt}")
    ck = torch.load(ckpt, map_location="cpu")

    # construct cfg from ckpt if present
    if cfg is None and isinstance(ck, dict) and "cfg" in ck:
        try:
            cfg = SimpleNamespace(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
        except Exception:
            cfg = None

    if cfg is None:
        # if still None we build a minimal cfg from tokenizer
        cfg = SimpleNamespace(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=512,
            n_heads=8,
            dim_ff=2048,
            max_len=512,
            padding_idx=tokenizer.token_to_id("[PAD]") if hasattr(tokenizer, "token_to_id") else 0,
            num_layers=10,
            dropout=0.1,
            tie_weights=False,
        )

    model = GenerativeModel(cfg)
    # load state dict
    if isinstance(ck, dict) and "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"], strict=False)
    elif isinstance(ck, dict) and "model_state" in ck:
        model.load_state_dict(ck["model_state"], strict=False)
    else:
        # assume ck is raw state_dict
        try:
            model.load_state_dict(ck)
        except Exception as e:
            print("Failed to load state dict:", e)
            raise

    model.to(device)

    stop_id = pick_stop_token(cfg, tokenizer)
    if stop_id is not None:
        print(f"Using stop token id: {stop_id}")

    # interactive loop
    print("Interactive inference. Type an input and press Enter. Empty line to quit.")
    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if prompt.strip() == "":
            print("Exiting.")
            break
        if not args.no_stream_prompt:
            print(prompt)
        greedy_stream_generate(model, tokenizer, device, prompt, max_new_tokens=args.max_new_tokens, stop_id=stop_id)


if __name__ == "__main__":
    main()


## python script_infer.py --ckpt ./../model/checkpoints/checkpoint_epoch45.pth --cfg ./../config/cfg.json --tokenizer