#!/usr/bin/env python3



import os
import glob
import random
import argparse
import time
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
import esm
from transformers import AutoModel, AutoTokenizer

from PBClassifier.load_protbert import (
    PBLinearClassifier,
    PoolerClassifier,
    RawClassifier,
    safe_load_state_dict
)

from progen_revised_2026_continue import (
    DEVICE,
    FORBIDDEN_IDS,
    build_forbidden_ids,
    classifier_logits,
    generate_batch,
    load_generator,
    log,
    make_ref_model,
    middle_from_full,
    policy_loss,
)


@dataclass
class EmpiricalDist:
    sorted_vals: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.sorted_vals.mean())

    @property
    def std(self) -> float:
        return float(self.sorted_vals.std())

    @property
    def max(self) -> float:
        return float(self.sorted_vals[-1])

    def cdf(self, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.sorted_vals, x, side="right")
        return idx / float(len(self.sorted_vals))

    def cdf_with_power_tail(
        self,
        x: np.ndarray,
        alpha: float,
        power: float,
        scale: float,
    ) -> np.ndarray:
        base = self.cdf(x)
        if alpha <= 0:
            return base
        x = np.asarray(x, dtype=np.float64)
        dmax = self.max
        scale = max(float(scale), 1e-12)
        mask = x > dmax
        if not np.any(mask):
            return base
        out = base.astype(np.float64, copy=True)
        excess = (x[mask] - dmax) / scale
        out[mask] = 1.0 + float(alpha) * np.power(excess, float(power))
        return out


@dataclass
class EmbeddingState:
    model: torch.nn.Module
    alphabet: object
    batch_converter: object
    device: torch.device
    reference_emb: torch.Tensor
    bank_embs: torch.Tensor


def clean_seq(s: str) -> str:
    return "".join([c for c in str(s).upper() if "A" <= c <= "Z"])


def preprocess_esm_sequence(seq: str) -> str:
    return " ".join(clean_seq(seq).upper())


@torch.no_grad()
def embed_batch_notebook_style(
    batch_records: List[Tuple[str, str]],
    model: torch.nn.Module,
    alphabet,
    batch_converter,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    labels, strs, tokens = batch_converter(batch_records)
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    representations = out["representations"][model.num_layers]

    cls_batch: Dict[str, torch.Tensor] = {}
    for i, seq_id in enumerate(labels):
        cls_embedding = representations[i, 0, :].detach()
        cls_batch[seq_id] = cls_embedding
    return cls_batch


def load_distance_csv(
    path: str,
    seq_col: str,
    dist_col: str,
    id_col: str,
    reference_id: str,
    reference_sequence: str,
    bank_size: int,
    seed: int,
) -> Tuple[EmpiricalDist, str, List[str]]:
    df = pd.read_csv(path)
    if seq_col not in df.columns:
        raise ValueError(f"Missing sequence column '{seq_col}' in {path}")
    if dist_col not in df.columns:
        raise ValueError(f"Missing distance column '{dist_col}' in {path}")

    df = df.copy()
    df[seq_col] = df[seq_col].astype(str).map(clean_seq)
    df = df[df[seq_col].str.len() > 0]

    dist_vals = pd.to_numeric(df[dist_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if dist_vals.size == 0:
        raise ValueError(f"No valid numeric values in '{dist_col}'")

    empirical = EmpiricalDist(sorted_vals=np.sort(dist_vals))

    ref_seq = clean_seq(reference_sequence)
    if not ref_seq:
        ref_seq = clean_seq(df.iloc[-1][seq_col])
    if not ref_seq:
        if id_col in df.columns:
            ref_rows = df[df[id_col].astype(str) == str(reference_id)]
            if not ref_rows.empty:
                ref_seq = clean_seq(ref_rows.iloc[0][seq_col])
        if not ref_seq and "tag" in df.columns:
            ref_rows = df[df["tag"].astype(str).str.lower() == "reference"]
            if not ref_rows.empty:
                ref_seq = clean_seq(ref_rows.iloc[0][seq_col])
    if not ref_seq:
        raise ValueError(
            "Reference sequence not found. Provide --reference_sequence or ensure CSV has reference row by id/tag."
        )

    others_df = df[df[seq_col] != ref_seq]
    others = others_df[seq_col].dropna().tolist()
    if not others:
        raise ValueError("No non-reference sequences found in CSV for diversity bank")

    rng = random.Random(seed)
    if len(others) > bank_size:
        idx = list(range(len(others)))
        rng.shuffle(idx)
        others = [others[i] for i in idx[:bank_size]]

    return empirical, ref_seq, others


def load_esm2_embeddings(
    model_name: str,
    reference_seq: str,
    bank_sequences: List[str],
    batch_size: int,
) -> EmbeddingState:
    log(f"Loading ESM2 model: {model_name}")
    if not hasattr(esm, "pretrained"):
        raise RuntimeError(
            "Imported 'esm' module does not provide 'pretrained'. "
            "You likely installed the wrong package. Install fair-esm in this environment: "
            "`pip uninstall -y esm && pip install fair-esm`."
        )
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    device = DEVICE
    model.to(device)
    batch_converter = alphabet.get_batch_converter()

    def embed(seqs: List[str]) -> torch.Tensor:
        records = [(str(i), clean_seq(s)) for i, s in enumerate(seqs)]
        cls_map: Dict[str, torch.Tensor] = {}
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            cls_batch = embed_batch_notebook_style(batch, model, alphabet, batch_converter, device)
            cls_map.update(cls_batch)
        X = torch.stack([cls_map[str(i)] for i in range(len(records))], dim=0).to(device)
        return X

    ref_emb = embed([reference_seq])[0].unsqueeze(0)
    bank_embs = embed(bank_sequences)
    return EmbeddingState(
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
        device=device,
        reference_emb=ref_emb,
        bank_embs=bank_embs,
    )


def embed_with_state(state: EmbeddingState, seqs: List[str], batch_size: int) -> torch.Tensor:
    records = [(str(i), clean_seq(s)) for i, s in enumerate(seqs)]
    cls_map: Dict[str, torch.Tensor] = {}
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        cls_batch = embed_batch_notebook_style(
            batch_records=batch,
            model=state.model,
            alphabet=state.alphabet,
            batch_converter=state.batch_converter,
            device=state.device,
        )
        cls_map.update(cls_batch)
    X = torch.stack([cls_map[str(i)] for i in range(len(records))], dim=0).to(state.device)
    return X


def build_viability_classifier(
    variant: str,
    base_model: str,
    weights_path: str,
):
    if not weights_path:
        raise ValueError("--classifier_weights_path is required")

    base = AutoModel.from_pretrained(base_model).to(DEVICE)
    hf_tok = AutoTokenizer.from_pretrained(base_model, do_lower_case=False)

    v = variant.lower()
    if v == "pooler":
        clf = PoolerClassifier(base, hf_tok, DEVICE)
    elif v == "raw":
        clf = RawClassifier(base, hf_tok, DEVICE)
    else:
        clf = PBLinearClassifier(base, hf_tok, DEVICE)

    state = safe_load_state_dict(weights_path, DEVICE)
    clf.load_state_dict(state, strict=False)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


def rewards_to_advantages(
    reward: torch.Tensor,
    top_q: float,
    adv_gain: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adv = (reward - reward.mean()) / (reward.std() + 1e-6)
    if 0 < top_q < 1:
        k = max(1, int(len(adv) * top_q))
        keep = torch.topk(adv, k).indices
    else:
        keep = torch.arange(len(adv), device=adv.device)
    adv_kept = torch.clamp(adv[keep], min=0) * adv_gain
    return adv_kept, keep


def plot_series(values: List[float], ylabel: str, step: int, out_dir: str, prefix: str):
    if not values:
        return
    for old in glob.glob(os.path.join(out_dir, f"{prefix}_*.png")):
        try:
            os.remove(old)
        except OSError:
            pass
    plt.figure()
    plt.plot(values)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} up to step {step}")
    plt.grid(True)
    fname = os.path.join(out_dir, f"{prefix}_{step:06d}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved plot: {fname}")


def combine_rewards(
    viab_prob: torch.Tensor,
    ref_reward: torch.Tensor,
    args,
) -> torch.Tensor:
    total_reward, _, _ = reward_terms(viab_prob, ref_reward, args)
    return total_reward


def reward_terms(
    viab_prob: torch.Tensor,
    ref_reward: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = getattr(args, "reward_combine_mode", "add")
    thr = float(getattr(args, "viab_gate_threshold", 0.5))
    viab_term = args.viability_weight * viab_prob
    ref_term = args.ref_div_weight * ref_reward
    if mode == "mul":
        viab_pow = float(getattr(args, "mul_viab_power", 1.0))
        div_pow = float(getattr(args, "mul_div_power", 1.0))
        gate = (viab_prob >= thr).float()
        viab_term = gate * torch.pow(torch.clamp(viab_prob, min=1e-12), viab_pow)
        ref_term = gate * torch.pow(torch.clamp(ref_reward, min=1e-12), div_pow)
        return viab_term * ref_term, viab_term, ref_term
    return viab_term + ref_term, viab_term, ref_term


def parse_temp_list(raw: str) -> List[float]:
    temps: List[float] = []
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            val = float(tok)
        except ValueError:
            continue
        if val > 0:
            temps.append(val)
    out: List[float] = []
    seen = set()
    for t in temps:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@torch.no_grad()
def export_sequences_100k(
    model,
    tok,
    forbidden_ids: torch.Tensor,
    args,
    out_root: str,
):
    if not args.generate_100k_enable:
        return

    total = int(args.generate_100k_num_sequences)
    batch_n = int(args.generate_100k_batch_size)
    temps = parse_temp_list(args.generate_100k_temps)
    if not temps:
        temps = [args.temp]

    for gen_temp in temps:
        temp_tag = str(gen_temp).replace(".", "p")
        out_dir = os.path.join(out_root, "generated_100k", f"temp_{temp_tag}")
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, "sequences_100k.txt")
        pkl_path = os.path.join(out_dir, "sequences_100k.pkl")
        cfg_path = os.path.join(out_dir, "run_config.json")

        run_config = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_sequences": total,
            "batch_size": batch_n,
            "seed": args.seed,
            "temp": gen_temp,
            "top_p": args.top_p,
            "rep_penalty": args.rep_penalty,
            "rejection_samples": args.rejection_samples,
            "comp_max_frac": args.comp_max_frac,
            "no_repeat_ngram": args.no_repeat_ngram,
            "max_run_same_aa": args.max_run_same_aa,
        }
        with open(cfg_path, "w") as fcfg:
            json.dump(run_config, fcfg, indent=2)
        log(f"Saved generation config: {cfg_path}")

        sequences: List[str] = []
        log(f"Starting post-train generation: {total} sequences at temp={gen_temp}")
        with open(txt_path, "w") as ftxt:
            generated = 0
            while generated < total:
                cur = min(batch_n, total - generated)
                batch = generate_batch(
                    model=model,
                    tok=tok,
                    n=cur,
                    temp=gen_temp,
                    top_p=args.top_p,
                    rep_penalty=args.rep_penalty,
                    rejection_samples=args.rejection_samples,
                    forbidden_ids=forbidden_ids,
                    max_run_same_aa=args.max_run_same_aa,
                    comp_max_frac=args.comp_max_frac,
                    no_repeat_ngram=args.no_repeat_ngram,
                )
                for seq, _ in batch:
                    sequences.append(seq)
                    ftxt.write(seq + "\n")
                generated += cur
                log(f"[temp={gen_temp}] Generated {generated}/{total} sequences")

        with open(pkl_path, "wb") as fpkl:
            pickle.dump(sequences, fpkl)
        log(f"Wrote {len(sequences)} sequences to: {txt_path}")
        log(f"Wrote {len(sequences)} sequences to: {pkl_path}")


@torch.no_grad()
def generate_only(args):
    os.makedirs(args.out_dir, exist_ok=True)
    gen_model, gen_tok = load_generator(args.init_model_path.strip() or None)
    if args.resume_weights_dir:
        log(f"Loading generation weights from checkpoint: {args.resume_weights_dir}")
        gen_model = gen_model.__class__.from_pretrained(args.resume_weights_dir, trust_remote_code=True).to(DEVICE)
        gen_model.eval()
    forbid_ids = FORBIDDEN_IDS if FORBIDDEN_IDS is not None else build_forbidden_ids(gen_tok)
    export_sequences_100k(
        model=gen_model,
        tok=gen_tok,
        forbidden_ids=forbid_ids,
        args=args,
        out_root=args.out_dir,
    )


def train(args):
    t0 = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dirs = {
        "root": args.out_dir,
        "logs": os.path.join(args.out_dir, "logs"),
        "plots": os.path.join(args.out_dir, "plots"),
        "samples": os.path.join(args.out_dir, "samples"),
        "checkpoints": os.path.join(args.out_dir, "checkpoints"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    empirical, reference_seq, bank_sequences = load_distance_csv(
        path=args.dist_csv,
        seq_col=args.csv_seq_col,
        dist_col=args.csv_dist_col,
        id_col=args.csv_id_col,
        reference_id=args.reference_id,
        reference_sequence=args.reference_sequence,
        bank_size=args.bank_size,
        seed=args.seed,
    )
    log(
        f"Loaded distance distribution from CSV: n={len(empirical.sorted_vals)} "
        f"mean={empirical.mean:.6e} std={empirical.std:.6e}"
    )
    log(f"Reference sequence length={len(reference_seq)}, bank size={len(bank_sequences)}")

    emb_state = load_esm2_embeddings(
        model_name=args.esm_model,
        reference_seq=reference_seq,
        bank_sequences=bank_sequences,
        batch_size=args.esm_batch_size,
    )

    gen_model, gen_tok = load_generator(args.init_model_path.strip() or None)
    ref_model = make_ref_model(gen_model)
    optim = AdamW(gen_model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    clf = build_viability_classifier(
        variant=args.classifier_variant,
        base_model=args.classifier_base,
        weights_path=args.classifier_weights_path,
    )

    forbid_ids = FORBIDDEN_IDS if FORBIDDEN_IDS is not None else build_forbidden_ids(gen_tok)

    metrics_rows: List[Dict] = []
    total_loss_hist, reward_hist, viab_hist, ref_hist = [], [], [], []

    es_mode = args.early_stop_mode
    if es_mode == "auto":
        es_mode = "min" if args.early_stop_metric in {"loss", "nll", "kl"} else "max"
    best_val: Optional[float] = None
    bad_steps = 0

    for step in range(1, args.steps + 1):
        batch = generate_batch(
            gen_model,
            gen_tok,
            args.seqs_per_step,
            args.temp,
            args.top_p,
            args.rep_penalty,
            args.rejection_samples,
            forbid_ids,
            args.max_run_same_aa,
            args.comp_max_frac,
            args.no_repeat_ngram,
        )
        seqs = [s for s, _ in batch]

        with torch.no_grad():
            logits = classifier_logits(clf, batch)
            viab_prob = torch.softmax(logits, dim=1)[:, 1]

        emb_gen = embed_with_state(emb_state, seqs, args.esm_batch_size)

        emb_gen_n = F.normalize(emb_gen, p=2, dim=1)
        ref_n = F.normalize(emb_state.reference_emb, p=2, dim=1)
        ref_sim = emb_gen_n @ ref_n.T
        dist_ref_t = 1.0 - ref_sim.squeeze(1)
        dist_ref = dist_ref_t.detach().cpu().numpy()
        dist_ref_argmax = int(np.argmax(dist_ref))
        if args.ref_div_tail_mode == "power":
            tail_scale = empirical.std if args.ref_div_tail_scale == "std" else empirical.max
            ref_cdf = empirical.cdf_with_power_tail(
                dist_ref,
                alpha=args.ref_div_tail_alpha,
                power=args.ref_div_tail_power,
                scale=tail_scale,
            )
        else:
            ref_cdf = empirical.cdf(dist_ref)
        ref_reward = torch.tensor(ref_cdf, dtype=torch.float32, device=DEVICE)
        bank_reward = torch.zeros_like(ref_reward)
        total_reward, viab_component, ref_component = reward_terms(viab_prob, ref_reward, args)

        advs, keep_idx = rewards_to_advantages(total_reward, args.top_q, args.adv_gain)

        kept = keep_idx.tolist()
        batch_kept = [batch[i] for i in kept]
        viab_kept = viab_prob[keep_idx]
        ref_kept = ref_reward[keep_idx]
        viab_component_kept = viab_component[keep_idx]
        ref_component_kept = ref_component[keep_idx]
        bank_kept = bank_reward[keep_idx]
        reward_kept = total_reward[keep_idx]

        loss, pl_metrics = policy_loss(
            model=gen_model,
            ref_model=ref_model,
            tok=gen_tok,
            batch=batch_kept,
            advs=advs,
            kl_beta=args.kl_beta,
            entropy_bonus=args.entropy_bonus,
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), args.clip_grad_norm)
        optim.step()

        mid = [middle_from_full(s) for s, _ in batch_kept]
        metrics = {
            "step": step,
            "loss": float(loss.item()),
            "reward_mean": float(reward_kept.mean().item()) if reward_kept.numel() else 0.0,
            "viab_mean": float(viab_kept.mean().item()) if viab_kept.numel() else 0.0,
            "viab_component_mean": float(viab_component_kept.mean().item()) if viab_component_kept.numel() else 0.0,
            "ref_div_mean": float(ref_kept.mean().item()) if ref_kept.numel() else 0.0,
            "ref_div_component_mean": float(ref_component_kept.mean().item()) if ref_component_kept.numel() else 0.0,
            "bank_div_mean": float(bank_kept.mean().item()) if bank_kept.numel() else 0.0,
            "dist_ref_mean": float(np.mean(dist_ref)),
            "dist_ref_max": float(np.max(dist_ref)),
            "dist_ref_max_div_score": float(ref_reward[dist_ref_argmax].item()),
            "dist_ref_p50": float(np.quantile(dist_ref, 0.50)),
            "dist_ref_p90": float(np.quantile(dist_ref, 0.90)),
            "num_ref_tail": int(np.sum(dist_ref > empirical.max)),
            "nll": pl_metrics["nll"],
            "kl": pl_metrics["kl"],
            "entropy": pl_metrics["entropy"],
            "kept": int(len(kept)),
            "unique_ratio": float(len(set(seqs)) / max(1, len(seqs))),
            "mid_len_mean": float(np.mean([len(m) for m in mid])) if mid else 0.0,
        }
        metrics_rows.append(metrics)

        total_loss_hist.append(metrics["loss"])
        reward_hist.append(metrics["reward_mean"])
        viab_hist.append(metrics["viab_mean"])
        ref_hist.append(metrics["ref_div_mean"])
        log(
            f"[step {step:04d}] loss={metrics['loss']:.4f} "
            f"reward={metrics['reward_mean']:.4f} "
            f"viab_score={metrics['viab_mean']:.4f} "
            f"div_score={metrics['ref_div_mean']:.6f} "
            f"dist_ref_max={metrics['dist_ref_max']:.6e} "
            f"dist_ref_max_div_score={metrics['dist_ref_max_div_score']:.6f} "
            f"num_ref_tail={metrics['num_ref_tail']}"
        )

        stop_now = False
        if args.early_stop_enable:
            cur = float(metrics[args.early_stop_metric])
            if best_val is None:
                best_val = cur
            else:
                improved = (cur < (best_val - args.early_stop_min_delta)) if es_mode == "min" else (cur > (best_val + args.early_stop_min_delta))
                if improved:
                    best_val = cur
                    bad_steps = 0
                elif step >= args.early_stop_min_steps:
                    bad_steps += 1

            if step >= args.early_stop_min_steps and bad_steps >= args.early_stop_patience:
                log(
                    f"Early stopping triggered at step {step}: "
                    f"metric={args.early_stop_metric} mode={es_mode} "
                    f"best={best_val:.6f} patience={args.early_stop_patience} "
                    f"min_delta={args.early_stop_min_delta}"
                )
                stop_now = True

        plotted_this_step = False
        if step % args.plot_every == 0:
            plot_series(total_loss_hist, "policy loss", step, out_dirs["plots"], "loss")
            plot_series(reward_hist, "reward mean", step, out_dirs["plots"], "reward")
            plot_series(viab_hist, "viability reward", step, out_dirs["plots"], "viability")
            plot_series(ref_hist, "reference diversity reward", step, out_dirs["plots"], "ref_div")
            csv_path = os.path.join(out_dirs["logs"], f"step_metrics_{step:06d}.csv")
            pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
            log(f"Wrote metrics CSV: {csv_path}")
            plotted_this_step = True

        if stop_now and not plotted_this_step:
            plot_series(total_loss_hist, "policy loss", step, out_dirs["plots"], "loss")
            plot_series(reward_hist, "reward mean", step, out_dirs["plots"], "reward")
            plot_series(viab_hist, "viability reward", step, out_dirs["plots"], "viability")
            plot_series(ref_hist, "reference diversity reward", step, out_dirs["plots"], "ref_div")
            csv_path = os.path.join(out_dirs["logs"], f"step_metrics_{step:06d}.csv")
            pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
            log(f"Wrote metrics CSV: {csv_path}")

        if step % args.ckpt_every == 0 or step == args.steps or stop_now:
            ckpt_dir = os.path.join(out_dirs["checkpoints"], f"step_{step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            gen_model.save_pretrained(ckpt_dir)
            torch.save({"optimizer": optim.state_dict(), "step": step}, os.path.join(ckpt_dir, "state.pt"))

            df_samples = pd.DataFrame({
                "seq": [s for s, _ in batch_kept],
                "middle": [middle_from_full(s) for s, _ in batch_kept],
                "viability": viab_kept.detach().cpu().tolist(),
                "ref_div": ref_kept.detach().cpu().tolist(),
                "bank_div": bank_kept.detach().cpu().tolist(),
                "reward": reward_kept.detach().cpu().tolist(),
            })
            sample_path = os.path.join(out_dirs["samples"], f"samples_{step:06d}.csv")
            df_samples.to_csv(sample_path, index=False)
            log(f"Wrote samples CSV: {sample_path}")

        if stop_now:
            break

    final_csv = os.path.join(out_dirs["logs"], "step_metrics_FINAL.csv")
    pd.DataFrame(metrics_rows).to_csv(final_csv, index=False)
    log(f"Wrote final metrics CSV: {final_csv}")

    max_num_ref_tail = max((int(r.get("num_ref_tail", 0)) for r in metrics_rows), default=0)
    if args.generate_100k_only_if_num_ref_tail_gt >= 0 and max_num_ref_tail <= args.generate_100k_only_if_num_ref_tail_gt:
        log(
            "Skipping post-train generation: "
            f"max_num_ref_tail={max_num_ref_tail} <= threshold={args.generate_100k_only_if_num_ref_tail_gt}"
        )
    else:
        export_sequences_100k(
            model=gen_model,
            tok=gen_tok,
            forbidden_ids=forbid_ids,
            args=args,
            out_root=out_dirs["root"],
        )

    elapsed_sec = time.perf_counter() - t0
    h = int(elapsed_sec // 3600)
    m = int((elapsed_sec % 3600) // 60)
    s = elapsed_sec % 60.0
    log(f"Training completed in {h:02d}:{m:02d}:{s:05.2f} (hh:mm:ss)")


def parse_args():
    p = argparse.ArgumentParser(description="RL with viability + diversity from dist_to_reference distribution")

    p.add_argument("--init_model_path", type=str, default="", help="Generator checkpoint path; empty uses base model")
    p.add_argument("--out_dir", type=str, default="/path/to/rl_output_dir")
    p.add_argument("--post_train_generate_only", type=int, default=0, help="Skip RL training and only run post-train generation from --init_model_path")
    p.add_argument("--resume_weights_dir", type=str, default="", help="Checkpoint directory to load into the already-instantiated generator for generation-only mode")

    p.add_argument("--dist_csv", type=str, required=True, help="CSV containing sequence list and dist_to_reference")
    p.add_argument("--csv_seq_col", type=str, default="Sequence")
    p.add_argument("--csv_dist_col", type=str, default="dist_to_reference")
    p.add_argument("--csv_id_col", type=str, default="sequence_id")
    p.add_argument("--reference_id", type=str, default="reference")
    p.add_argument("--reference_sequence", type=str, default="", help="Optional explicit reference sequence")
    p.add_argument("--bank_size", type=int, default=512, help="How many CSV sequences to keep as diversity bank")

    p.add_argument("--esm_model", type=str, default="esm2_t33_650M_UR50D")
    p.add_argument("--esm_batch_size", type=int, default=8)

    p.add_argument("--classifier_base", type=str, default="Rostlab/prot_bert_bfd")
    p.add_argument("--classifier_variant", type=str, default="mean", choices=["mean", "pooler", "raw"])
    p.add_argument(
        "--classifier_weights_path",
        type=str,
        default="/path/to/viability_classifier_weights.pt",
    )

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seqs_per_step", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--kl_beta", type=float, default=0.005)
    p.add_argument("--entropy_bonus", type=float, default=0.0015)
    p.add_argument("--top_q", type=float, default=0.25)
    p.add_argument("--adv_gain", type=float, default=48.0)

    p.add_argument("--viability_weight", type=float, default=1.0)
    p.add_argument("--ref_div_weight", type=float, default=0.7)
    p.add_argument("--bank_div_weight", type=float, default=0.0, help="Deprecated; ignored (bank diversity removed)")
    p.add_argument("--reward_combine_mode", type=str, default="add", choices=["add", "mul"])
    p.add_argument("--viab_gate_threshold", type=float, default=0.5)
    p.add_argument("--mul_viab_power", type=float, default=1.0, help="Exponent for viability score in mul reward mode")
    p.add_argument("--mul_div_power", type=float, default=1.0, help="Exponent for diversity score in mul reward mode")
    p.add_argument("--ref_div_tail_mode", type=str, default="none", choices=["none", "power"])
    p.add_argument("--ref_div_tail_alpha", type=float, default=0.1)
    p.add_argument("--ref_div_tail_power", type=float, default=1.5)
    p.add_argument("--ref_div_tail_scale", type=str, default="std", choices=["std", "max"])

    p.add_argument("--temp", type=float, default=1.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--rep_penalty", type=float, default=1.15)
    p.add_argument("--rejection_samples", type=int, default=128)
    p.add_argument("--comp_max_frac", type=float, default=0.35)
    p.add_argument("--no_repeat_ngram", type=int, default=0)
    p.add_argument("--max_run_same_aa", type=int, default=5)

    p.add_argument("--plot_every", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--generate_100k_enable", type=int, default=1)
    p.add_argument("--generate_100k_num_sequences", type=int, default=100000)
    p.add_argument("--generate_100k_batch_size", type=int, default=512)
    p.add_argument(
        "--generate_100k_only_if_num_ref_tail_gt",
        type=int,
        default=-1,
        help="If >=0, skip post-train generation unless max(step num_ref_tail) is greater than this value",
    )
    p.add_argument(
        "--generate_100k_temps",
        type=str,
        default="1.2",
        help="Comma-separated temperatures for post-train generation, e.g. '0.8,1.2'",
    )
    p.add_argument("--early_stop_enable", type=int, default=1)
    p.add_argument(
        "--early_stop_metric",
        type=str,
        default="loss",
        choices=["reward_mean", "loss", "viab_mean", "ref_div_mean", "bank_div_mean", "nll", "kl", "entropy"],
    )
    p.add_argument("--early_stop_mode", type=str, default="auto", choices=["auto", "max", "min"])
    p.add_argument("--early_stop_patience", type=int, default=200)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--early_stop_min_steps", type=int, default=300)

    return p.parse_args()


def main():
    args = parse_args()
    log(f"DEVICE: {DEVICE}")
    log("==== RL RUN PARAMETERS ====")
    for k, v in sorted(vars(args).items()):
        log(f"{k}: {v}")
    log("===========================")
    if int(args.post_train_generate_only):
        generate_only(args)
        return
    train(args)


if __name__ == "__main__":
    main()
