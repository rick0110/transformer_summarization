import os
import time
import json
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from archtecture import GenerativeModel
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_path: str, text_columns: List[str], max_len: int = 512):
        self.max_len = max_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.examples: List[List[int]] = []

        if data_path.endswith(".json"):
            with open(data_path, encoding="utf-8") as f:
                rows = json.load(f)
        else:
            raise ValueError("Data must be .json")

        for r in rows:
            parts = []
            for c in text_columns:
                v = r.get(c, "")
                if v is None:
                    v = ""
                parts.append(v.strip())
            text = " ".join([p for p in parts if p])
            if not text:
                continue
            ids = self.tokenizer.encode(text).ids
            if len(ids) < 2:
                continue
            ids = ids[: self.max_len]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_batch(batch: List[List[int]], pad_id: int, device: Optional[torch.device] = None):
    batch_size = len(batch)
    max_len = max(len(x) for x in batch)
    ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    if device is not None:
        ids = ids.to(device)
    return ids


def evaluate(model: GenerativeModel, dataloader: DataLoader, device: torch.device, pad_token_id: int):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_ids in dataloader:
            batch_ids = batch_ids.to(device)
            logits = model(batch_ids)  # (b, seq_len, vocab)
            if logits.size(1) <= 1:
                continue
            pred = logits[:, :-1, :].contiguous()
            labels = batch_ids[:, 1:].contiguous()
            loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))
            total_loss += loss.item()
            total_tokens += (labels != pad_token_id).sum().item()
    avg_loss = total_loss / max(1, total_tokens)
    ppl = float("inf")
    if total_tokens > 0:
        ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def train(
    cfg,
    dataset_path: str,
    tokenizer_path: str,
    text_columns: List[str],
    output_dir: str,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 5e-5,
    max_len: int = 512,
    device: Optional[torch.device] = None,
    save_every: int = 1,
    tb_log_dir: Optional[str] = None,
    log_histograms_every: int = 1000,
    resume_from: Optional[str] = None,
):
    """
    Treina o modelo GenerativeModel definido em archtecture.py usando seus tokenizadores.
    Returns: trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # TensorBoard writer
    writer: Optional[SummaryWriter] = None
    if tb_log_dir:
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)

    ds = TextDataset(dataset_path, tokenizer_path, text_columns, max_len=max_len)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("[PAD]")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, pad_id, device))

    model = GenerativeModel(cfg).to(device)

    if writer is not None:
        try:
            sample_batch = next(iter(dl))
            sample_batch = sample_batch.to(device)
            writer.add_graph(model, sample_batch)
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * max(1, len(dl))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optim_state_dict", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 1) + 1
        global_step = ckpt.get("step", 0)
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_ids in dl:
            batch_ids = batch_ids.to(device)  # (b, seq_len)
            logits = model(batch_ids)  # (b, seq_len, vocab)
            if logits.size(1) <= 1:
                continue

            pred = logits[:, :-1, :].contiguous()
            labels = batch_ids[:, 1:].contiguous()

            loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))
            tokens = (labels != pad_id).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_tokens += tokens
            global_step += 1

            # per-step logging
            if writer is not None:
                writer.add_scalar("train/loss_sum", loss.item(), global_step)
                if tokens > 0:
                    writer.add_scalar("train/loss_token", loss.item() / max(1, tokens), global_step)
                # learning rate
                try:
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("train/lr", current_lr, global_step)
                except Exception:
                    pass

                # optional histograms
                if log_histograms_every and global_step % log_histograms_every == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            writer.add_histogram(f"params/{name}", param.detach().cpu().numpy(), global_step)
                            if param.grad is not None:
                                writer.add_histogram(f"grads/{name}", param.grad.detach().cpu().numpy(), global_step)



        avg_loss = epoch_loss / max(1, epoch_tokens)
        ppl = torch.exp(torch.tensor(avg_loss)).item() if epoch_tokens > 0 else float("inf")
        t1 = time.time()
        print(f"Epoch {epoch}/{epochs}  loss/token={avg_loss:.6f}  ppl={ppl:.3f}  time={t1-t0:.1f}s")

        eval_loss, eval_ppl = evaluate(model, dl, device, pad_id)
        print(f"  Eval loss/token: {eval_loss:.6f}  ppl={eval_ppl:.3f}")

        # per-epoch logging
        if writer is not None:
            writer.add_scalar("epoch/train_loss_token", avg_loss, epoch)
            writer.add_scalar("epoch/train_ppl", ppl, epoch)
            writer.add_scalar("epoch/eval_loss_token", eval_loss, epoch)
            writer.add_scalar("epoch/eval_ppl", eval_ppl, epoch)

        # save best
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
                "epoch": epoch,
                "step": global_step,
            }, os.path.join(output_dir, "best_checkpoint.pth"))
            print(f"  Saved best checkpoint (epoch {epoch})")

        if epoch % save_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
                "epoch": epoch,
                "step": global_step,
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch}.pth"))

    # close writer
    if writer is not None:
        writer.flush()
        writer.close()

    return model


