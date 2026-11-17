"""Raw-only arithmetic data and evaluation helpers for the nanoGPT baseline.

The learner sees only lines of the form:

    37+45=82
    37*45=1665

There are no partial products, carries, worked examples, or strategy annotations. The Python
arithmetic below is only the external label generator and evaluator; it is never imported by the
model or placed in its input stream.

This script does not edit nanoGPT's model/trainer. It prepares compatible character-level data
and can evaluate an existing checkpoint on fresh operand pairs.
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np


def equation(a, b, op):
    result = a + b if op == "+" else a * b
    return "%d%s%d=%d" % (a, op, b, result)


def integer_with_digits(rng, digits):
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    return rng.randint(lo, 10**digits)


def pairs_for_digits(rng, digits, count):
    return [(integer_with_digits(rng, da), integer_with_digits(rng, db), op)
            for _ in range(count)
            for da, db, op in [(digits, digits, "+" if rng.random() < 0.5 else "*")]]


def write_split(path, train_lines, val_lines, stoi, meta):
    os.makedirs(path, exist_ok=True)
    train_text = "\n".join(train_lines) + "\n"
    val_text = "\n".join(val_lines) + "\n"
    np.fromiter((stoi[ch] for ch in train_text), dtype=np.uint16,
                count=len(train_text)).tofile(os.path.join(path, "train.bin"))
    np.fromiter((stoi[ch] for ch in val_text), dtype=np.uint16,
                count=len(val_text)).tofile(os.path.join(path, "val.bin"))
    with open(os.path.join(path, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


def prepare(out_dir, memory_repeats=64, form_examples=120000, val_examples=6000, seed=1337):
    """Create a two-phase corpus: exhaustive one-digit associations, then raw 2-3 digit equations."""
    rng = random.Random(seed)
    memory = [equation(a, b, op)
              for op in ("+", "*")
              for a in range(10)
              for b in range(10)]
    # Shuffle each repetition so the model sees the facts in varied local contexts.
    memory_lines = []
    for _ in range(memory_repeats):
        block = memory[:]
        rng.shuffle(block)
        memory_lines.extend(block)

    form_pairs = []
    for _ in range(form_examples):
        da = rng.choice((2, 3))
        db = rng.choice((2, 3))
        op = "+" if rng.random() < 0.5 else "*"
        form_pairs.append((integer_with_digits(rng, da),
                           integer_with_digits(rng, db), op))
    form_lines = [equation(a, b, op) for a, b, op in form_pairs]

    val = []
    for _ in range(val_examples):
        da = rng.choice((2, 3))
        db = rng.choice((2, 3))
        op = "+" if rng.random() < 0.5 else "*"
        a, b = integer_with_digits(rng, da), integer_with_digits(rng, db)
        val.append(equation(a, b, op))

    alphabet = sorted(set("\n".join(memory_lines + form_lines + val) + "\n"))
    stoi = {ch: i for i, ch in enumerate(alphabet)}
    itos = {i: ch for ch, i in stoi.items()}
    meta = dict(vocab_size=len(alphabet), stoi=stoi, itos=itos,
                min_digits=1, max_digits=3, ops="+*", format="raw-equation-final-answer")

    # Separate phase directories and phase-matched validation; same symbols preserve checkpoint
    # compatibility when the learner moves from memory facts to form examples.
    write_split(os.path.join(out_dir, "memory"), memory_lines, memory * 8, stoi, meta)
    write_split(os.path.join(out_dir, "form"), form_lines, val, stoi, meta)

    print("raw-only corpus written:", os.path.abspath(out_dir))
    print("memory phase: %d one-digit facts x %d shuffled repetitions = %d lines"
          % (len(memory), memory_repeats, len(memory_lines)))
    print("form phase: %d random 2-3 digit equations; validation: %d equations"
          % (len(form_lines), len(val)))
    print("vocabulary:", repr("".join(alphabet)))
    print("sample memory:", " | ".join(memory_lines[:4]))
    print("sample form:", " | ".join(form_lines[:4]))
    print("format check: one '=' per line; no intermediate-work lines:",
          all(line.count("=") == 1 for line in memory_lines + form_lines + val))


def evaluate(nanogpt_root, checkpoint, meta_dir, seed=20261118, per_cell=128,
             max_digits=6, device="cpu"):
    """Fresh random pairs by digit-length cell; scoring oracle is not part of model input."""
    sys.path.insert(0, os.path.abspath(nanogpt_root))
    import torch
    import eval_local as evaluator

    with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runner = evaluator.Runner(ck, meta["stoi"], meta["itos"], torch.device(device))
    rng = random.Random(seed)

    def first_line_answers(pairs, op):
        """Decode only through the first newline after the prompted equation.

        The shared eval_local Runner has a stop condition intended for echoed/multiline
        generations. A raw-only model may emit just the answer, so this experiment uses the
        task's actual contract: the first line after `a op b=` is the answer.
        """
        prompts = ["\n%d%s%d=" % (a, op, b) for a, b in pairs]
        if len({len(s) for s in prompts}) != 1:
            raise ValueError("evaluation batches must have equal-length prompts")
        ids = torch.tensor([[runner.stoi[ch] for ch in s] for s in prompts],
                           dtype=torch.long, device=runner.dev)
        done = [False] * len(prompts)
        output = [""] * len(prompts)
        with torch.no_grad():
            for _ in range(min(32, runner.bs - ids.shape[1])):
                logits, _ = runner.model(ids[:, -runner.bs:])
                nxt = logits[:, -1].argmax(-1).tolist()
                col = []
                for i, token in enumerate(nxt):
                    if done[i]:
                        col.append(runner.nl)
                        continue
                    if token == runner.nl:
                        done[i] = True
                        col.append(token)
                    else:
                        output[i] += runner.itos.get(token, "?")
                        col.append(token)
                ids = torch.cat([ids, torch.tensor(col, device=runner.dev).unsqueeze(1)], 1)
                if all(done):
                    break
        answers = []
        for line in output:
            s = line.strip()
            if "=" in s:
                s = s.rsplit("=", 1)[-1].strip()
            answers.append(int(s) if s.isdigit() else None)
        return answers

    print("checkpoint:", os.path.abspath(checkpoint))
    print("model args:", {k: ck["model_args"].get(k) for k in
                          ("n_layer", "n_head", "n_embd", "n_loop", "block_size")})
    print("Fresh random-pair exact match; %d examples per equal-digit cell" % per_cell)
    print("digits    add exact      multiply exact")
    for digits in range(1, max_digits + 1):
        lo = 0 if digits == 1 else 10 ** (digits - 1)
        hi = 10**digits
        pairs = [(rng.randrange(lo, hi), rng.randrange(lo, hi)) for _ in range(per_cell)]
        add_out = first_line_answers(pairs, "+")
        mul_out = first_line_answers(pairs, "*")
        add_ok = sum(got == a + b for (a, b), got in zip(pairs, add_out))
        mul_ok = sum(got == a * b for (a, b), got in zip(pairs, mul_out))
        print("%3d digit   %4d/%-4d %.3f   %4d/%-4d %.3f"
              % (digits, add_ok, per_cell, add_ok / per_cell,
                 mul_ok, per_cell, mul_ok / per_cell))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("out_dir")
    prep.add_argument("--memory-repeats", type=int, default=64)
    prep.add_argument("--form-examples", type=int, default=120000)
    prep.add_argument("--val-examples", type=int, default=6000)
    prep.add_argument("--seed", type=int, default=1337)
    ev = sub.add_parser("eval")
    ev.add_argument("--nanogpt-root", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--meta-dir", required=True)
    ev.add_argument("--seed", type=int, default=20261118)
    ev.add_argument("--per-cell", type=int, default=128)
    ev.add_argument("--max-digits", type=int, default=6)
    ev.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.out_dir, a.memory_repeats, a.form_examples, a.val_examples, a.seed)
    else:
        evaluate(a.nanogpt_root, a.checkpoint, a.meta_dir, a.seed,
                 a.per_cell, a.max_digits, a.device)


if __name__ == "__main__":
    main()
