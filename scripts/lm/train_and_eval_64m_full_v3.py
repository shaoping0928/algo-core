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
ckpt_64m = os.path.join(ROOT, "checkpoints", "wiki_zhihu_64m_v2000.pkl.gz")
VOCAB_V = 2000
SEED_N = 16_000_000
CHUNK_N = 16_000_000
TARGET_N = 64_000_000
EVAL_OFFSET = 260_000_000
EVAL_LEN = 20_000

t_global = time.perf_counter()

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

def unit_text(cm, uid: int) -> str:
    atoms = cm.chunk_to_atoms_.get(int(uid), (int(uid),))
    return "".join(itos[a] if 0 <= a < len(itos) else "?" for a in atoms)

seed_toks = encode_text(seed_text)
del seed_text, counts
gc.collect()

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
    f"Prepared 16M seed & untouched ev@260M (len={len(ev)}, UNK={100*float(np.mean(ev==0)):.2f}%) "
    f"[{time.perf_counter()-t_global:.1f}s] | RSS={current_rss_bytes()/(1024**2):.1f} MB",
    flush=True,
)

# 1. Build 16M seed with 64-dim embeddings + expanded 6000/level domain chunks + GPU 2-layer 512-gate policy!
bundle = StructuralSubstrateBundle.fit_parallel(
    seed_toks,
    vocab_size=VOCAB_V,
    vocab=vocab,
    itos=itos,
    n_workers=16,
    max_memory_bytes=16 * (1024 ** 3),
    feature_dim=64,
    max_chunks_per_level=6000,
    n_chunk_classes=224,
    post_fit_noise_prune=True,
    noise_prune_pressure=0.22,
    stage_slice=400_000,
    seed=42,
)
cf = bundle.temp_memory.enable_circuit_field(use_gpu=True, fit_tokens=seed_toks[:4_000_000])
# Fuse self-learned RC persistence tau = R * C into selective entity write gates
sp = bundle.stateful_policy
if cf is not None and sp is not None:
    tau = (cf.R * cf.C)[:VOCAB_V]
    t_hi = max(float(np.percentile(tau[1:], 95)), 1.0)
    rc_wg = np.clip(tau / t_hi, 0.02, 1.0).astype(np.float32)
    sp.write_gate_ = np.clip(0.65 * sp.write_gate_ + 0.35 * rc_wg, 0.015, 1.0).astype(np.float32)
    sp.write_gate_[0] = 0.0

del seed_toks
gc.collect()
print(
    f"\n[16M Seed] wall={bundle.metadata['wall_seconds']}s | timings={bundle.metadata['component_timings']} | "
    f"chunks={bundle.chunk_memory.n_total_units:,} | ctx_nodes={len(bundle.context_integrator.forward_graph_):,} | "
    f"sp_backend={sp.train_backend} | sp_loss={sp.last_loss_:.3f} | sp_params={sp.param_bytes()/1024**2:.2f} MB | "
    f"RSS={current_rss_bytes()/(1024**2):.1f} MB / 16384 MB",
    flush=True,
)

# 2. Stream remaining 48M (3 x 16M -> 64M default config!)
with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
    f.read(SEED_N)
    ingested = SEED_N
    c_idx = 0
    while ingested < TARGET_N:
        take = min(CHUNK_N, TARGET_N - ingested)
        txt = f.read(take)
        if not txt:
            break
        c_toks = encode_text(txt)
        del txt
        c_idx += 1
        rep = bundle.ingest_stream_chunk(
            c_toks, n_workers=12, max_memory_bytes=16 * (1024 ** 3), min_high_order_count=2,
        )
        del c_toks
        ingested = rep["total_tokens"]
        mem_i = rep["memory"]
        print(
            f"  [Chunk {c_idx}/3 -> {ingested/1e6:4.1f}M] {rep['seconds']:.1f}s | "
            f"ctx_nodes={len(bundle.context_integrator.forward_graph_):,} | "
            f"sp_samples={sp.total_samples:,} sp_loss={sp.last_loss_:.3f} ({sp.train_backend}) | "
            f"RSS={mem_i['rss_mb']:.1f} MB ({mem_i['ratio']*100:.1f}%) | p={mem_i['pressure']:.3f}",
            flush=True,
        )
sp.close_gpu()

# 3. X-Ray: Check expanded domain chunk discovery on P1 & P2!
print("\n=== X-RAY: Expanded 14K-Chunk Domain Coverage & Multi-Cue Intersection ===", flush=True)
for tag, sp_pos in [("P1(立冬/农历/节假日)", 16367), ("P2(个税/收入/BI数据分析)", 19122)]:
    pt = [int(x) for x in ev[sp_pos:sp_pos + 60]]
    bundle.temp_memory.reset()
    for t in pt:
        bundle.temp_memory.observe(t, source="external")
    slots = bundle.context_integrator.active_slots_snapshot(top_k=14)
    slot_str = " ".join(f"{unit_text(bundle.chunk_memory, u)}({w:.1f})" for u, w in slots)
    top_pred = bundle.context_integrator.contextual_unit_scores(top_k=20)
    pred_str = " ".join(f"{unit_text(bundle.chunk_memory, u)}({sc:.2f},k={sup})" for u, sc, sup in top_pred)
    print(f"  {tag}:")
    print(f"    Clean Slots : {slot_str}")
    print(f"    Top20 Inter : {pred_str}", flush=True)

# 4. Online Calibration on ev[5:5000]
emb_raw = bundle.class_policy.feature_map.embeddings_
emb_norm = emb_raw / np.maximum(np.linalg.norm(emb_raw, axis=1, keepdims=True), 1e-8)
emb_cn = bundle.class_policy.emb_centered_norm_

t_cal = time.perf_counter()
ctrl_cal = bundle.make_controller(use_arbitrator=True)
bundle.temp_memory.reset()
slow = np.zeros(emb_raw.shape[1], dtype=float)
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
print(f"\n=== 64M Online Calibration ({time.perf_counter()-t_cal:.1f}s) ===", flush=True)
print(f"  Arbitrator: {bundle.arbitrator.snapshot()}", flush=True)

# 5. Strict Ablation on @260M ev[5000:11000] (n=5876)
ctrl_base = bundle.make_controller(use_arbitrator=True)
ctrl_base.stateful_policy = None

bundle.temp_memory.reset()
slow = np.zeros(emb_raw.shape[1], dtype=float)
for i in range(4940, 5000):
    bundle.temp_memory.observe(int(ev[i]), source="external")
    slow = bundle.class_policy.update_slow_state(slow, int(ev[i]))

betas = [0.0, 0.25, 0.45, 0.70]
stats = {"DEEP_2L_ONLY": [0, 0, 0, 0, 0]}
for bt in betas:
    stats[f"L4_b={bt:.2f}"] = [0, 0, 0, 0, 0]

n_ev = 0
for i in range(5000, 11000):
    tgt = int(ev[i])
    if tgt > 0:
        ctx = ev[i - 8:i]
        acts, probs, info = ctrl_base.decide_distribution(ctx, slow, k=16)
        sm_only = sp.predict_symbol_map(k=8, self_mode=False)
        ca_only = [a for a, _ in sorted(sm_only.items(), key=lambda kv: kv[1], reverse=True)[:8]]
        log_bf = sp.context_log_bayes_factor(self_mode=False)
        c_gate = float(info.get("content_gate", 1.0))
        ent_g = float(info.get("entropy_gate", 1.0))
        n_ev += 1

        st0 = stats["DEEP_2L_ONLY"]
        if ca_only and ca_only[0] == tgt: st0[0] += 1
        if tgt in ca_only: st0[1] += 1
        if ca_only:
            if bundle.class_policy.sym_to_class_[ca_only[0]] == bundle.class_policy.sym_to_class_[tgt]: st0[2] += 1
            if float(np.dot(emb_norm[ca_only[0]], emb_norm[tgt])) >= 0.45: st0[3] += 1
            if float(np.dot(emb_cn[ca_only[0]], emb_cn[tgt])) >= 0.20: st0[4] += 1

        for bt in betas:
            if bt == 0.0 or log_bf is None:
                ranked = acts[:8]
            else:
                eff = bt * ent_g * max(c_gate, 0.20)
                sc = [float(p) * float(np.exp(np.clip(eff * float(log_bf[a]), -1.8, 2.2))) for a, p in zip(acts, probs)]
                order = np.argsort(sc)[::-1][:8]
                ranked = [acts[j] for j in order]
            st = stats[f"L4_b={bt:.2f}"]
            if ranked[0] == tgt: st[0] += 1
            if tgt in ranked: st[1] += 1
            if bundle.class_policy.sym_to_class_[ranked[0]] == bundle.class_policy.sym_to_class_[tgt]: st[2] += 1
            if float(np.dot(emb_norm[ranked[0]], emb_norm[tgt])) >= 0.45: st[3] += 1
            if float(np.dot(emb_cn[ranked[0]], emb_cn[tgt])) >= 0.20: st[4] += 1
    bundle.temp_memory.observe(tgt, source="external")
    slow = bundle.class_policy.update_slow_state(slow, tgt)

print(f"\n=== 64M v3 (64-dim E_dyn + 2-Layer 512-Gate + 14K Chunks) Ablation @260M ev[5000:11000] (n={n_ev}) ===", flush=True)
for key in ["DEEP_2L_ONLY"] + [f"L4_b={bt:.2f}" for bt in betas]:
    h1, h8, sc, sa, csa = stats[key]
    print(
        f"  64M {key:12s} : Top-1={100*h1/n_ev:5.2f}% | Top-8={100*h8/n_ev:5.2f}% | "
        f"SameCls={100*sc/n_ev:5.2f}% | SemAlign={100*sa/n_ev:5.2f}% | CenteredSem={100*csa/n_ev:5.2f}%"
    )

# 6. Save reusable 64M v3 checkpoint!
sv = bundle.save(ckpt_64m, compresslevel=1)
print(
    f"\nSaved reusable 64M v3 checkpoint: {sv} | Final RSS={current_rss_bytes()/(1024**2):.1f} MB / 16384 MB",
    flush=True,
)

# 7. Audit exact same P1, P2, P3 100-Char Continuations (with Whole-Chunk Commitment + 64-dim 2L GPU Policy!)
def ngram_repeat_rate(seq, n_g=3):
    grams = [tuple(seq[j:j + n_g]) for j in range(len(seq) - n_g + 1)]
    return 1.0 - (len(set(grams)) / float(max(len(grams), 1)))

wiki_sp = 0
for j in range(len(ev) - 180):
    if decode_tokens(ev[j:j + 6]) == "大上海时代广":
        wiki_sp = j
        break

test_list = [("P1(立冬/日历)", 16367), ("P2(个税/BI数据)", 19122)]
if wiki_sp > 0:
    test_list.append(("P3(Wiki-上海广场)", wiki_sp))

print("\n=== 64M v3 Whole-Chunk Commitment + 64-dim 2L Residual GPU Lookahead 100-Char Continuations ===", flush=True)
for p_idx, (label, sp_pos) in enumerate(test_list, 1):
    p_toks = [int(x) for x in ev[sp_pos:sp_pos + 60]]
    gt_toks = [int(x) for x in ev[sp_pos + 60:sp_pos + 160]]
    ptree = bundle.build_prompt_stage_tree(p_toks)
    ctrl_g = bundle.make_controller(use_arbitrator=True, stage_tree=ptree, lookahead_depth=2, lookahead_beam=5)
    ctrl_g.circuit_diff_weight = 0.85

    bundle.temp_memory.reset()
    hg = np.zeros(emb_raw.shape[1], dtype=float)
    for tok in p_toks:
        bundle.temp_memory.observe(tok, source="external")
        hg = bundle.class_policy.update_slow_state(hg, tok)
    bundle.temp_memory.pin_root_context()
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
    pv_c, gv_c, gtv_c = emb_cn[p_toks].mean(0), emb_cn[gen].mean(0), emb_cn[gt_toks].mean(0)
    cp_c = float(np.dot(gv_c, pv_c) / (np.linalg.norm(gv_c) * np.linalg.norm(pv_c) + 1e-9))
    cg_c = float(np.dot(gv_c, gtv_c) / (np.linalg.norm(gv_c) * np.linalg.norm(gtv_c) + 1e-9))
    rep3 = ngram_repeat_rate(gen, 3)
    print(
        f"\n[{label} @ 260M+{sp_pos}] {ms_s:.1f} ms/step | uniq={len(set(gen))}/100 | "
        f"3gram_rep={100*rep3:.1f}% | spaces={gs.count(' ')} | centered_cos(p)={cp_c:.3f} centered_cos(gt)={cg_c:.3f}"
    )
    print(f"  PROMPT: {decode_tokens(p_toks)}")
    print(f"  TEXT  : {gs}", flush=True)

cf.close()
print(f"\nTotal Wall: {time.perf_counter()-t_global:.1f}s", flush=True)
