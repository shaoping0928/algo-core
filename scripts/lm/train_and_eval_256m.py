from collections import Counter
import gc
import os
import sys
import time
import numpy as np

ROOT = r"c:\workspace\code\algo_core"
sys.path.insert(0, ROOT)
for dp, _, _ in os.walk(os.path.join(ROOT, "core")):
    if "__pycache__" not in dp:
        sys.path.insert(0, dp)

from substrate_bundle import StructuralSubstrateBundle
from memory_governor import current_rss_bytes

corpus_path = r"C:\workspace\code\data\wiki_zhihu\corpus.txt"
ckpt_256m = os.path.join(ROOT, "checkpoints", "wiki_zhihu_256m_v2000.pkl.gz")
VOCAB_V = 2000
SEED_N = 16_000_000
CHUNK_N = 16_000_000
TARGET_N = 256_000_000
EVAL_OFFSET = 260_000_000
EVAL_LEN = 20_000

t_global = time.perf_counter()

# 1. Read initial 16M seed to build Top-2000 vocab (fast, ~1.5s)
with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
    seed_text = f.read(SEED_N)

counts = Counter(seed_text)
itos = ["<UNK>"] + [ch for ch, _ in counts.most_common(VOCAB_V - 1)]
vocab = {ch: i for i, ch in enumerate(itos)}

def encode_text(text: str) -> np.ndarray:
    vget = vocab.get
    return np.fromiter((vget(ch, 0) for ch in text), dtype=np.int32, count=len(text))

def decode_tokens(seq) -> str:
    return "".join(itos[int(i)] if 0 <= int(i) < len(itos) else "?" for i in seq)

seed_toks = encode_text(seed_text)
del seed_text, counts
gc.collect()

# Load strictly untouched held-out slice from offset=260M (zero overlap with 0..256M!)
# Read in 16M chunks to skip cleanly to EVAL_OFFSET
with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
    rem = EVAL_OFFSET
    while rem > 0:
        rd = f.read(min(rem, 16_000_000))
        if not rd:
            break
        rem -= len(rd)
    ev_text = f.read(EVAL_LEN)
ev = encode_text(ev_text)
del ev_text
print(
    f"Prepared 16M seed (UNK={100*float(np.mean(seed_toks==0)):.2f}%) & untouched ev@260M "
    f"(len={len(ev)}, UNK={100*float(np.mean(ev==0)):.2f}%) [{time.perf_counter()-t_global:.1f}s] | "
    f"RSS={current_rss_bytes()/(1024**2):.1f} MB",
    flush=True,
)

# 2. Build 16M base bundle via bit-packed multi-core parallel engine under 16GB ceiling
bundle = StructuralSubstrateBundle.fit_parallel(
    seed_toks,
    vocab_size=VOCAB_V,
    vocab=vocab,
    itos=itos,
    n_workers=16,
    max_memory_bytes=16 * (1024 ** 3),
    post_fit_noise_prune=True,
    noise_prune_pressure=0.22,
    stage_slice=400_000,
    seed=42,
)
cf = bundle.temp_memory.enable_circuit_field(use_gpu=True, fit_tokens=seed_toks[:4_000_000])
del seed_toks
gc.collect()
print(
    f"\n[Milestone 16M] wall={bundle.metadata['wall_seconds']}s | timings={bundle.metadata['component_timings']} | "
    f"RSS={current_rss_bytes()/(1024**2):.1f} MB / 16384 MB | GPU={cf.backend}",
    flush=True,
)

emb = bundle.class_policy.feature_map.embeddings_
emb_norm = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)

def quick_eval_milestone(tag: str):
    ctrl_eval = bundle.make_controller(use_arbitrator=False)
    bundle.temp_memory.reset()
    h = np.zeros(emb.shape[1], dtype=float)
    for i in range(4950, 5000):
        bundle.temp_memory.observe(int(ev[i]), source="external")
        h = bundle.class_policy.update_slow_state(h, int(ev[i]))
    h1 = h8 = sc = sa = n = 0
    for i in range(5000, 11000):
        tgt = int(ev[i])
        if tgt > 0:
            acts, _, _ = ctrl_eval.decide_distribution(ev[i - 8:i], h, k=8)
            n += 1
            if acts and acts[0] == tgt: h1 += 1
            if tgt in acts: h8 += 1
            if acts:
                if bundle.class_policy.sym_to_class_[acts[0]] == bundle.class_policy.sym_to_class_[tgt]: sc += 1
                if float(np.dot(emb_norm[acts[0]], emb_norm[tgt])) >= 0.45: sa += 1
        bundle.temp_memory.observe(tgt, source="external")
        h = bundle.class_policy.update_slow_state(h, tgt)
    rss_mb = current_rss_bytes() / (1024.0 ** 2)
    n_cls_k = len(bundle.class_policy.class_index._postings)
    n_loc_k = len(bundle.class_policy.local_index._postings)
    n_chk_k = len(bundle.chunk_memory.chunk_index._postings) if bundle.chunk_memory.chunk_index else 0
    print(
        f"  >> LEARNING CURVE [{tag:5s}] (n={n}): Top-1={100*h1/n:5.1f}% | Top-8={100*h8/n:5.1f}% | "
        f"SameCls={100*sc/n:5.1f}% | SemAlign={100*sa/n:5.1f}% | keys(cls/loc/chk)={n_cls_k:,}/{n_loc_k:,}/{n_chk_k:,} | "
        f"RSS={rss_mb:7.1f} MB ({100*rss_mb/16384:.1f}%) | elapsed={time.perf_counter()-t_global:.1f}s",
        flush=True,
    )

quick_eval_milestone("16M")

# 3. Stream remaining 240M characters (30 x 8M chunks = 256M total!) under 16GB ceiling
milestones = {32_000_000, 64_000_000, 128_000_000, 192_000_000, 256_000_000}
with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
    # Skip first 16M already ingested
    f.read(SEED_N)
    ingested = SEED_N
    chunk_idx = 0
    while ingested < TARGET_N:
        take = min(CHUNK_N, TARGET_N - ingested)
        txt = f.read(take)
        if not txt:
            break
        c_toks = encode_text(txt)
        del txt
        chunk_idx += 1
        rep = bundle.ingest_stream_chunk(
            c_toks,
            n_workers=12,
            max_memory_bytes=16 * (1024 ** 3),
            min_high_order_count=2,
        )
        del c_toks
        ingested = rep["total_tokens"]
        mem_i = rep["memory"]
        print(
            f"  [Chunk {chunk_idx:2d} -> {ingested/1e6:5.1f}M] {rep['seconds']:.2f}s | "
            f"RSS={mem_i['rss_mb']:.1f} MB ({mem_i['ratio']*100:.1f}%) | "
            f"pressure={mem_i['pressure']:.3f} | evicted={mem_i['evicted_items']:,}",
            flush=True,
        )
        if ingested in milestones:
            quick_eval_milestone(f"{ingested // 1_000_000}M")

# 4. Cross-article online calibration on ev[5:5000] (gate + arbitrator + per-node R/C feedback)
t_cal = time.perf_counter()
ctrl_cal = bundle.make_controller(use_arbitrator=True)
bundle.temp_memory.reset()
slow = np.zeros(emb.shape[1], dtype=float)
for i in range(5):
    bundle.temp_memory.observe(int(ev[i]), source="external")
    slow = bundle.class_policy.update_slow_state(slow, int(ev[i]))
for i in range(5, 5000):
    ctx = ev[max(0, i - 8):i]
    tgt = int(ev[i])
    if tgt > 0:
        if i < 2000:
            ctrl_cal.learn_gate_step(ctx, slow, tgt, k=8)
        if i >= 500:
            ctrl_cal.learn_arbitrator_step(ctx, slow, tgt, k=8)
            ctrl_cal.learn_rc_step(ctx, slow, tgt, lr=0.08)
    bundle.temp_memory.observe(tgt, source="external")
    slow = bundle.class_policy.update_slow_state(slow, tgt)
bundle.arbitrator.consolidate_episode()
cf.sync_rc_to_gpu()
print(f"\n=== 256M Online Calibration ({time.perf_counter()-t_cal:.1f}s) ===", flush=True)
print(f"  Gate      : {bundle.learned_gate.snapshot()}", flush=True)
print(f"  Arbitrator: {bundle.arbitrator.snapshot()}", flush=True)

# 5. Final strictly untouched evaluation on ev[5000:11000] (Direct L3 vs Arbitrator L4)
ctrl_d = bundle.make_controller(use_arbitrator=False)
ctrl_a = bundle.make_controller(use_arbitrator=True)
h1_d = h8_d = sc_d = sa_d = h1_a = h8_a = sc_a = sa_a = n_ev = 0
for i in range(5000, 11000):
    tgt = int(ev[i])
    if tgt > 0:
        ctx = ev[i - 8:i]
        cd, _, _ = ctrl_d.decide_distribution(ctx, slow, k=8)
        ca, _, _ = ctrl_a.decide_distribution(ctx, slow, k=8)
        n_ev += 1
        if cd and cd[0] == tgt: h1_d += 1
        if tgt in cd: h8_d += 1
        if cd:
            if bundle.class_policy.sym_to_class_[cd[0]] == bundle.class_policy.sym_to_class_[tgt]: sc_d += 1
            if float(np.dot(emb_norm[cd[0]], emb_norm[tgt])) >= 0.45: sa_d += 1
        if ca and ca[0] == tgt: h1_a += 1
        if tgt in ca: h8_a += 1
        if ca:
            if bundle.class_policy.sym_to_class_[ca[0]] == bundle.class_policy.sym_to_class_[tgt]: sc_a += 1
            if float(np.dot(emb_norm[ca[0]], emb_norm[tgt])) >= 0.45: sa_a += 1
    bundle.temp_memory.observe(tgt, source="external")
    slow = bundle.class_policy.update_slow_state(slow, tgt)

print(f"\n=== FINAL 256M Strictly Untouched Held-Out @260M ev[5000:11000] (n={n_ev}, V={VOCAB_V}) ===", flush=True)
print(f"  256M Direct Fixed (L3)             : Top-1={100*h1_d/n_ev:.1f}% | Top-8={100*h8_d/n_ev:.1f}% | SameCls={100*sc_d/n_ev:.1f}% | SemAlign={100*sa_d/n_ev:.1f}%")
print(f"  256M Cross-Episode Arbitrator (L4) : Top-1={100*h1_a/n_ev:.1f}% | Top-8={100*h8_a/n_ev:.1f}% | SameCls={100*sc_a/n_ev:.1f}% | SemAlign={100*sa_a/n_ev:.1f}%", flush=True)

# 6. Save reusable 256M checkpoint!
sv = bundle.save(ckpt_256m, compresslevel=1)
print(f"\nSaved reusable 256M checkpoint: {sv} | Final RSS={current_rss_bytes()/(1024**2):.1f} MB / 16384 MB | Total Wall={time.perf_counter()-t_global:.1f}s", flush=True)

# 7. Two 100-char continuations on strictly untouched passages @260M+ (GPU multi-branch lookahead)
nl_id = vocab.get("\n", -1)
prompts = []
cur_p = 11200
while len(prompts) < 2 and cur_p < 19000:
    found = None
    for idx in range(cur_p, min(cur_p + 2500, 19400)):
        if int(ev[idx]) == nl_id and idx + 180 < len(ev):
            seg = decode_tokens(ev[idx + 1:idx + 65])
            if "<UNK>" not in seg and "\n" not in seg[:50] and len(seg.strip()) >= 50:
                if sum(1 for c in seg[:60] if "\u4e00" <= c <= "\u9fff") >= 40:
                    found = idx + 1
                    break
    if found is not None:
        prompts.append(found)
        cur_p = found + 2400
    else:
        cur_p += 1200

print("\n=== 256M GPU Multi-Branch Lookahead 100-Char Continuations (@260M+) ===", flush=True)
for p_idx, sp in enumerate(prompts, 1):
    p_toks = [int(x) for x in ev[sp:sp + 60]]
    gt_toks = [int(x) for x in ev[sp + 60:sp + 160]]
    ptree = bundle.build_prompt_stage_tree(p_toks)
    ctrl_g = bundle.make_controller(use_arbitrator=True, stage_tree=ptree, lookahead_depth=2, lookahead_beam=5)
    ctrl_g.circuit_diff_weight = 0.85

    bundle.temp_memory.reset()
    hg = np.zeros(emb.shape[1], dtype=float)
    for tok in p_toks:
        bundle.temp_memory.observe(tok, source="external")
        hg = bundle.class_policy.update_slow_state(hg, tok)
    bundle.temp_memory.pin_root_context()
    if ctrl_g._anchor_state is None or len(ctrl_g._anchor_state) != len(hg):
        ctrl_g.pin_anchor_state(hg)

    rng_g = np.random.RandomState(11 + p_idx)
    seq = list(p_toks)
    gen = []
    tg = time.perf_counter()
    for _ in range(100):
        acts, probs, info = ctrl_g.decide_with_lookahead(seq[-8:], hg, k=8, depth=2, beam_width=5)
        valid = [(a, p) for a, p in zip(acts, probs) if a > 0]
        va = [a for a, _ in valid]
        vp = np.power(np.clip([p for _, p in valid], 1e-12, 1.0), 1.0 / 0.65)
        vp /= vp.sum()
        nxt = int(rng_g.choice(va, p=vp))
        gen.append(nxt)
        seq.append(nxt)
        bundle.temp_memory.observe(nxt, source="self")
        hg = bundle.class_policy.update_slow_state(hg, nxt)
    ms_s = 1000.0 * (time.perf_counter() - tg) / 100.0
    gs = decode_tokens(gen)
    pv, gv, gtv = emb[p_toks].mean(0), emb[gen].mean(0), emb[gt_toks].mean(0)
    cp = float(np.dot(gv, pv) / (np.linalg.norm(gv) * np.linalg.norm(pv) + 1e-9))
    cg = float(np.dot(gv, gtv) / (np.linalg.norm(gv) * np.linalg.norm(gtv) + 1e-9))
    print(f"\n[256M Passage #{p_idx} @ 260M+{sp}] {ms_s:.1f} ms/step | uniq={len(set(gen))}/100 | spaces={gs.count(' ')} | cos(p)={cp:.3f} cos(gt)={cg:.3f}")
    print(f"  PROMPT: {decode_tokens(p_toks)}")
    print(f"  GT100 : {decode_tokens(gt_toks)}")
    print(f"  TEXT  : {gs}", flush=True)

cf.close()
