#!/usr/bin/env python3

import os, re, random, argparse
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import hashlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

GEN_MODEL_NAME = "hugohrban/progen2-small"
DEFAULT_DATA_PATH = "/path/to/viable_sequences.txt"
DEFAULT_OUT_DIR = "/path/to/sft_output"

PREFIX = (
    "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTS"
    "FGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEG"
    "ADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQ"
    "NDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRL"
    "MNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIF"
    "GKQGSEKTNVDIEKVMIT"
)
SUFFIX = (
    "QAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQ"
    "YTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
)

MIN_GEN, MAX_GEN = 28, 43

N_PRETRAIN_GEN = 100000
N_FINETUNE_GEN = 100000
TEMP = 1.8
TOP_P = 0.9
COMPOSITION_MAX_FRAC = 0.35
NO_REPEAT_NGRAM = 0
MAX_RUN_SAME_AA = 5
REP_PENALTY = 1.15
REJECTION_SAMPLES = 128
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
RUN4_REGEX = re.compile(r"(.)\1\1\1")

def log(msg: str):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def clean_seq(s: str) -> str:
    import re
    return "".join(re.findall(r"[A-Za-z]", s)).upper()

class CustomTokenizerAdapter:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.tok.no_padding()
        self.eos_token_id = self._try_id("<eos>") or self._try_id("</s>")
    def _try_id(self, tok):
        try:
            return self.tok.token_to_id(tok)
        except Exception:
            return None
    def encode_ids(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tok.encode(text).ids, dtype=torch.long, device=DEVICE)
    def id_to_token(self, i: int) -> str:
        return self.tok.id_to_token(i)
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size(with_added_tokens=False)


def build_forbidden_ids(tok: CustomTokenizerAdapter) -> torch.Tensor:
    ids = [i for i in range(tok.vocab_size()) if "X" in str(tok.id_to_token(i)).upper()]
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return torch.tensor(sorted(set(ids)), dtype=torch.long, device=DEVICE)

@torch.no_grad()
def apply_repetition_penalty(logits: torch.Tensor, prev_ids: torch.Tensor, pen: float):
    if pen <= 1.0 or prev_ids.numel() == 0:
        return logits
    uniq = torch.unique(prev_ids)
    logits[uniq] /= pen
    return logits


def _violates_ngram(letters: str, L: str, n: int = NO_REPEAT_NGRAM) -> bool:
    if n <= 0:
        return False
    s = letters + L
    if len(s) < 2 * n:
        return False
    last = s[-n:]
    for i in range(len(s) - n):
        if s[i : i + n] == last:
            return True
    return False


def _violates_composition(letters: str, L: str) -> bool:
    from collections import Counter

    s = letters + L
    c = Counter(s)
    total = len(s)
    return any(v / total > COMPOSITION_MAX_FRAC for v in c.values())


def _violates_runs(letters: str, L: str) -> bool:
    s = letters + L
    run = 1
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] else 1
        if run > MAX_RUN_SAME_AA:
            return True
    return RUN4_REGEX.search(s) is not None


def _safe_id_to_text(tok: CustomTokenizerAdapter, i: int) -> str:
    return str(tok.id_to_token(i))


def _letters_from_token_text(t: str) -> str:
    return "".join([c for c in t if "A" <= c <= "Z"])


@torch.no_grad()
def _choose_with_constraints(model, tok: CustomTokenizerAdapter, ids: torch.Tensor, letters: str, target_len: int, prev_ids: torch.Tensor, temp: float, forbidden_ids: torch.Tensor) -> int:
    logits = model(ids).logits[-1, :] / temp
    logits = apply_repetition_penalty(logits, prev_ids, REP_PENALTY)
    if forbidden_ids is not None:
        logits[forbidden_ids] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep = idx[cum <= TOP_P]
    if keep.numel() == 0:
        keep = idx[:1]
    keep_probs = probs[keep] / probs[keep].sum()

    for _ in range(REJECTION_SAMPLES):
        cand = keep[torch.multinomial(keep_probs, 1)].item()
        t = _safe_id_to_text(tok, cand)
        L = _letters_from_token_text(t).upper()
        if not L or any(ch not in AA20 for ch in L):
            continue
        if len(letters) + len(L) > MAX_GEN:
            continue
        if _violates_runs(letters, L):
            continue
        if _violates_ngram(letters, L):
            continue
        if _violates_composition(letters, L):
            continue
        return cand
    return int(torch.argmax(probs))


@torch.no_grad()
def sample_one_sequence(model, tok: CustomTokenizerAdapter, temp: float, forbidden_ids: torch.Tensor) -> str:
    ids = tok.encode_ids(PREFIX)
    target_len = random.randint(MIN_GEN, MAX_GEN)
    gen_token_ids: List[int] = []
    letters = ""
    while len(letters) < target_len:
        cand = _choose_with_constraints(
            model,
            tok,
            ids,
            letters,
            target_len,
            torch.tensor(gen_token_ids, device=DEVICE),
            temp,
            forbidden_ids,
        )
        gen_token_ids.append(cand)
        ids = torch.cat([ids, torch.tensor([cand], device=DEVICE)])
        t = _safe_id_to_text(tok, cand)
        letters += "".join([c for c in _letters_from_token_text(t).upper() if c in AA20])
        if len(letters) >= MAX_GEN:
            break
    letters = letters.ljust(MIN_GEN, "A")[:MAX_GEN]
    return PREFIX + letters + SUFFIX


@torch.no_grad()
def generate_batch(model, tok: CustomTokenizerAdapter, n: int, temp: float, forbidden_ids: torch.Tensor) -> List[str]:
    seqs = []
    percent_step = max(1, n // 100)
    for i in range(n):
        seqs.append(sample_one_sequence(model, tok, temp, forbidden_ids))
        if (i + 1) % percent_step == 0 or (i + 1) == n:
            pct = int(100 * (i + 1) / n)
            print(f"[GENERATION] {pct}% ({i + 1}/{n}) sequences generated.")
    return seqs

def generate_unique_batch(model, tok, n, temp, forbidden_ids, out_path):
    seen = set()
    written = 0
    with open(out_path, "w") as f:
        while len(seen) < n:
            seq = sample_one_sequence(model, tok, temp, forbidden_ids)
            if seq in seen:
                continue
            seen.add(seq)
            if len(seen) % 1000 == 0 or len(seen) == n:
                for s in list(seen)[written:len(seen)]:
                    f.write(s + "\n")
                written = len(seen)
                log(f"[GENERATION] {written}/{n} unique sequences written.")
    return list(seen)


def load_generator_pretrained():
    log(f"Loading generator: {GEN_MODEL_NAME}")
    m = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME, trust_remote_code=True).to(DEVICE).eval()
    raw_tok = Tokenizer.from_pretrained(GEN_MODEL_NAME)
    tok = CustomTokenizerAdapter(raw_tok)
    if getattr(m.config, "pad_token_id", None) is None:
        m.config.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    return m, tok


@dataclass
class EncodedExample:
    input_ids: torch.Tensor  # 1D
    labels: torch.Tensor  # 1D (with -100 outside middle + pads)


class ViableSequencesDataset(Dataset):
    def __init__(self, path: str, tok: CustomTokenizerAdapter):
        self.path = path
        self.tok = tok
        self.lines = []
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        with open(p, "r") as f:
            for line in f:
                s = clean_seq(line.strip())
                if not s:
                    continue
                if not s.startswith(PREFIX) or not s.endswith(SUFFIX):
                    continue
                mid_len = len(s) - len(PREFIX) - len(SUFFIX)
                if mid_len < MIN_GEN or mid_len > MAX_GEN:
                    continue
                self.lines.append(s)
        if not self.lines:
            raise RuntimeError("No valid sequences found after filtering.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> EncodedExample:
        seq = self.lines[idx]
        enc = self.tok.tok.encode(seq)
        ids = enc.ids  # list[int]
        offsets = enc.offsets  # list[(start,end)] in char positions
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full((len(ids),), -100, dtype=torch.long)
        mid_start_char = len(PREFIX)
        mid_end_char = len(seq) - len(SUFFIX)
        for i, (s0, s1) in enumerate(offsets):
            if s0 is None or s1 is None:
                continue
            if s1 > mid_start_char and s0 < mid_end_char:
                labels[i] = input_ids[i]
        return EncodedExample(input_ids=input_ids, labels=labels)


class LeftPadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[EncodedExample]) -> Dict[str, torch.Tensor]:
        max_len = max(ex.input_ids.size(0) for ex in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            L = ex.input_ids.size(0)
            input_ids[i, :L] = ex.input_ids
            labels[i, :L] = ex.labels
            attention_mask[i, :L] = 1
        return {
            "input_ids": input_ids.to(DEVICE),
            "labels": labels.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE),
        }


@torch.no_grad()
def evaluate(model: AutoModelForCausalLM, dl: DataLoader, max_batches: int | None = None, max_examples: int | None = None) -> float:
    model.eval()
    losses = []
    seen_batches = 0
    seen_examples = 0
    for batch in dl:
        out = model(**batch)
        losses.append(out.loss.item())
        seen_batches += 1
        seen_examples += batch["input_ids"].size(0)
        if max_batches is not None and seen_batches >= max_batches:
            break
        if max_examples is not None and seen_examples >= max_examples:
            break
    return float(np.mean(losses)) if losses else float("inf")



def train(model: AutoModelForCausalLM, tok: CustomTokenizerAdapter, dl_train: DataLoader, dl_val: DataLoader | None, epochs: int = 1, lr: float = 1e-5, grad_clip: float = 1.0, patience: int = 3, out_dir: Path | None = None, eval_every_steps: int = 0, val_max_batches: int | None = None, val_sample_size: int | None = None):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    global_step = 0
    best_val = float("inf")
    epochs_bad = 0
    history = {
        "train_loss_steps": [],
        "train_loss_epochs": [],
        "val_loss_epochs": [],
        "val_loss_steps": [],
    }

    best_dir = None
    if out_dir is not None:
        best_dir = out_dir / "progen2-sft-best"
        best_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save_best(val_loss: float, tag: str):
        nonlocal best_val, epochs_bad
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_bad = 0
            if best_dir is not None:
                log(f"New best ({tag}), saving checkpoint...")
                model.save_pretrained(best_dir)
                (best_dir / "tokenizer.json").write_text(tok.tok.to_str())
        else:
            epochs_bad += 1
            log(f"No val improvement ({epochs_bad}/{patience}) [{tag}]")

    for ep in range(1, epochs + 1):
        losses = []
        for step, batch in enumerate(dl_train, start=1):
            out = model(**batch)
            loss = out.loss
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            l = loss.item()
            losses.append(l)
            history["train_loss_steps"].append((global_step, l))
            global_step += 1

            if dl_val is not None and eval_every_steps and (global_step % eval_every_steps == 0):
                vloss = evaluate(model, dl_val, max_batches=val_max_batches, max_examples=val_sample_size)
                history["val_loss_steps"].append((global_step, vloss))
                log(f"step {global_step} | quick val loss {vloss:.4f}")
                maybe_save_best(vloss, tag=f"step {global_step}")
                if epochs_bad >= patience:
                    log("Early stopping triggered (step-based).")
                    model.eval()
                    return history, best_dir

            if step % 50 == 0:
                log(f"epoch {ep} step {step} | loss {np.mean(losses):.4f}")
        train_epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss_epochs"].append((ep, train_epoch_loss))
        log(f"epoch {ep} done | mean loss {train_epoch_loss:.4f}")

        if dl_val is not None and not eval_every_steps:
            val_loss = evaluate(model, dl_val)
            history["val_loss_epochs"].append((ep, val_loss))
            log(f"epoch {ep} | val loss {val_loss:.4f}")
            maybe_save_best(val_loss, tag=f"epoch {ep}")
            if epochs_bad >= patience:
                log("Early stopping triggered (epoch-based).")
                break

    model.eval()
    return history, (best_dir if (dl_val is not None and best_dir is not None) else None)



def param_hash(args):
    param_str = f"data={args.data_path}|epochs={args.epochs}|bs={args.batch_size}|lr={args.lr}|temp={args.temperature}|seed={args.seed}"
    return hashlib.md5(param_str.encode()).hexdigest()[:8]

def param_dirname(args):
    data_base = Path(args.data_path).stem
    return f"data={data_base}_ep={args.epochs}_bs={args.batch_size}_lr={args.lr}_temp={args.temperature}_seed={args.seed}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval_every_steps", type=int, default=0, help="If >0, run quick validation every N steps on a small subset.")
    parser.add_argument("--val_cap", type=int, default=0, help="Cap the max number of validation examples (0 = no cap).")
    parser.add_argument("--val_sample_size", type=int, default=512, help="Max examples for quick step-based validation.")
    parser.add_argument("--val_max_batches", type=int, default=8, help="Max batches for quick step-based validation.")
    parser.add_argument("--no_finetune", action="store_true", help="If set, only generate from pretrained model and skip finetuning.")
    parser.add_argument("--temperature", type=float, default=1.8, help="Sampling temperature for sequence generation.")
    parser.add_argument("--only_generate_finetuned", action="store_true", help="If set, only load the finetuned model and generate sequences.")
    parser.add_argument("--unique_sequences", action="store_true", help="If set, only generate unique sequences and write every 1000 new ones.")
    args = parser.parse_args()

    print("==== SFT RUN PARAMETERS ====")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("============================")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.only_generate_finetuned:
        run_dir = param_dirname(args)
        out_dir = Path(args.out_dir) / run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        best_dir = out_dir / "progen2-sft-best"
        last_dir = out_dir / "progen2-sft-last"
        fallback_best = Path(args.out_dir) / "progen2-sft-best"
        fallback_last = Path(args.out_dir) / "progen2-sft-last"
        if best_dir.exists() and any(best_dir.glob("*")):
            log(f"Loading best finetuned model from: {best_dir}")
            model = AutoModelForCausalLM.from_pretrained(str(best_dir), trust_remote_code=True).to(DEVICE).eval()
        elif last_dir.exists() and any(last_dir.glob("*")):
            log(f"Loading last finetuned model from: {last_dir}")
            model = AutoModelForCausalLM.from_pretrained(str(last_dir), trust_remote_code=True).to(DEVICE).eval()
        elif fallback_best.exists() and any(fallback_best.glob("*")):
            log(f"Loading best finetuned model from: {fallback_best}")
            model = AutoModelForCausalLM.from_pretrained(str(fallback_best), trust_remote_code=True).to(DEVICE).eval()
        elif fallback_last.exists() and any(fallback_last.glob("*")):
            log(f"Loading last finetuned model from: {fallback_last}")
            model = AutoModelForCausalLM.from_pretrained(str(fallback_last), trust_remote_code=True).to(DEVICE).eval()
        else:
            raise RuntimeError(f"No finetuned model found in {out_dir} or {args.out_dir}")
        raw_tok = Tokenizer.from_pretrained(GEN_MODEL_NAME)
        tok = CustomTokenizerAdapter(raw_tok)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        forb_ids = build_forbidden_ids(tok)
        log(f"Generating {N_FINETUNE_GEN} sequences with finetuned model...")
        if args.unique_sequences:
            ft_path = out_dir / "UNIQUELY_GEN_finetuned_sequences.txt"
            seqs_ft = generate_unique_batch(model, tok, N_FINETUNE_GEN, args.temperature, forb_ids, ft_path)
        else:
            ft_path = out_dir / "finetuned_sequences.txt"
            seqs_ft = generate_batch(model, tok, N_FINETUNE_GEN, args.temperature, forb_ids)
            with open(ft_path, "w") as f:
                for s in seqs_ft:
                    f.write(s + "\n")
        log(f"Wrote: {ft_path}")
        return

    run_dir = param_dirname(args)
    out_dir = Path(args.out_dir) / run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_generator_pretrained()

    forb_ids = build_forbidden_ids(tok)
    log(f"Generating {N_PRETRAIN_GEN} sequences with pretrained model...")
    if args.unique_sequences:
        pre_path = out_dir / "UNIQUELY_GEN_pretrained_sequences.txt"
        seqs_pre = generate_unique_batch(model, tok, N_PRETRAIN_GEN, args.temperature, forb_ids, pre_path)
    else:
        pre_path = out_dir / "pretrained_sequences.txt"
        seqs_pre = generate_batch(model, tok, N_PRETRAIN_GEN, args.temperature, forb_ids)
        with open(pre_path, "w") as f:
            for s in seqs_pre:
                f.write(s + "\n")
    log(f"Wrote: {pre_path}")

    if args.no_finetune:
        log("--no_finetune set: skipping training and finetuned generation.")
        return

    ds = ViableSequencesDataset(args.data_path, tok)
    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    coll = LeftPadCollator(pad_id)

    n = len(ds)
    val_n = int(n * max(0.0, min(1.0, args.val_split)))
    idx = np.arange(n)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idx)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if args.val_cap and len(val_idx) > args.val_cap:
        val_idx = val_idx[: args.val_cap]
    from torch.utils.data import Subset
    ds_train = Subset(ds, train_idx.tolist())
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=coll)
    dl_val = None
    if val_n > 0:
        ds_val = Subset(ds, val_idx.tolist())
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=coll)
        log(f"Dataset split: train={len(train_idx)} val={len(val_idx)}")
    else:
        log(f"Dataset size={n}, no validation split (val_split={args.val_split})")

    log("Starting supervised fine-tuning...")
    history, best_dir = train(
        model,
        tok,
        dl_train,
        dl_val,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        out_dir=out_dir,
        eval_every_steps=args.eval_every_steps,
        val_max_batches=(args.val_max_batches if args.val_max_batches > 0 else None),
        val_sample_size=(args.val_sample_size if args.val_sample_size > 0 else None),
    )

    last_dir = out_dir / "progen2-sft-last"
    last_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving last model to: {last_dir}")
    model.save_pretrained(last_dir)
    (last_dir / "tokenizer.json").write_text(tok.tok.to_str())

    try:
        train_steps = history["train_loss_steps"]
        val_step = history.get("val_loss_steps", [])
        val_epochs = history["val_loss_epochs"]
        train_epochs = history["train_loss_epochs"]
        if plt is not None and (train_steps or val_epochs or val_step):
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
            if train_steps:
                xs, ys = zip(*train_steps)
                axes[0].plot(xs, ys, color="tab:blue", linewidth=1, label="train step")
            if val_step:
                vxs, vys = zip(*val_step)
                axes[0].plot(vxs, vys, color="tab:red", linestyle=":", label="val step")
            axes[0].set_title("Train/Val loss (per step)")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].legend(loc="best")
            if val_epochs:
                ex, vy = zip(*val_epochs)
                axes[1].plot(ex, vy, marker="o", color="tab:orange", label="val epoch")
            if train_epochs:
                ex2, ty = zip(*train_epochs)
                axes[1].plot(ex2, ty, marker="x", color="tab:green", linestyle="--", label="train epoch")
            axes[1].set_title("Epoch losses")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend(loc="best")
            plot_path = out_dir / "training_curves.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            log(f"Wrote: {plot_path}")
    except Exception as e:
        log(f"Plotting failed: {e}")

    try:
        steps_csv = out_dir / "train_loss_steps.csv"
        with open(steps_csv, "w") as f:
            f.write("step,train_loss\n")
            for s, l in history["train_loss_steps"]:
                f.write(f"{s},{l}\n")
        log(f"Wrote: {steps_csv}")
        val_steps_csv = out_dir / "val_loss_steps.csv"
        with open(val_steps_csv, "w") as f:
            f.write("step,val_loss\n")
            for s, l in history.get("val_loss_steps", []):
                f.write(f"{s},{l}\n")
        log(f"Wrote: {val_steps_csv}")
        epochs_csv = out_dir / "val_loss_epochs.csv"
        with open(epochs_csv, "w") as f:
            f.write("epoch,val_loss,train_loss\n")
            val_dict = {e: v for e, v in history["val_loss_epochs"]}
            tr_dict = {e: v for e, v in history["train_loss_epochs"]}
            all_ep = sorted(set(val_dict) | set(tr_dict))
            for e in all_ep:
                f.write(f"{e},{val_dict.get(e, '')},{tr_dict.get(e, '')}\n")
        log(f"Wrote: {epochs_csv}")
    except Exception as e:
        log(f"CSV logging failed: {e}")

    gen_model_dir = best_dir if (best_dir is not None and any(best_dir.glob("*"))) else last_dir
    if gen_model_dir == best_dir:
        log(f"Reloading best checkpoint from: {gen_model_dir}")
        model = AutoModelForCausalLM.from_pretrained(str(gen_model_dir), trust_remote_code=True).to(DEVICE).eval()
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    log(f"Generating {N_FINETUNE_GEN} sequences with fine-tuned model...")
    if args.unique_sequences:
        ft_path = out_dir / "UNIQUELY_GEN_finetuned_sequences.txt"
        seqs_ft = generate_unique_batch(model, tok, N_FINETUNE_GEN, args.temperature, forb_ids, ft_path)
    else:
        ft_path = out_dir / "finetuned_sequences.txt"
        seqs_ft = generate_batch(model, tok, N_FINETUNE_GEN, args.temperature, forb_ids)
        with open(ft_path, "w") as f:
            for s in seqs_ft:
                f.write(s + "\n")
    log(f"Wrote: {ft_path}")


if __name__ == "__main__":
    main()
