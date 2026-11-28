"""Build, Train & Blind-Evaluate 64M V=4096 Full P0-1 + P0-2 System:
  - P0-1: Slow-Floor Fast-Weight Gated Delta-Rule Associative State (fw_dim=64, lam in [0.85, 0.995])
  - P0-2A: Dual-Stream Multi-Granularity Input (Atom E[sym] + HierarchicalChunkMemory E_chunk[chk])
  - P0-2B: Expanded Vocabulary V=4096 (<UNK> < 0.5%) + H=768 x 2-Layer Residual RC on RX 9070 XT

Strict Protocol:
  - Vocab built from [0, 16M) (itos[:2000] == legacy V=2000 vocab identically).
  - Symbol layer + SSM trained strictly on [0, 64M).
  - UnionBlend alpha selected strictly on @257M dev.
  - Sealed @260M ev[5000:11000] evaluated ONCE with frozen alpha, reporting BOTH:
      (a) Full V=4096 metrics, and
      (b) Comparable V=2000 subset (1 <= tgt < 2000, exact same n=5876 positions as legacy table).
"""
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

from learnable_ssm_policy import LearnableStateSpacePolicy
from memory_governor import current_rss_bytes
from substrate_bundle import StructuralSubstrateBundle

corpus_path = r"C:\workspace\code\data\wiki_zhihu\corpus.txt"
ckpt_out = os.path.join(ROOT, "checkpoints", "wiki_zhihu_64m_v4096_p01_p02.pkl.gz")
VOCAB_V = 4096
SEED_N = 16_000_000
CHUNK_N = 16_000_000
TARGET_N = 64_000_000
DEV_OFFSET = 257_000_000
EVAL_OFFSET = 260_000_000
EVAL_LEN = 20_000


def ngram_repeat_rate(seq, n_g=3):
    grams = [tuple(seq[j : j + n_g]) for j in range(len(seq) - n_g + 1)]
    return 1.0 - (len(set(grams)) / float(max(len(grams), 1)))


def main():
    t_global = time.perf_counter()

    # 1. Build V=4096 vocab from [0, 16M)
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        seed_text = f.read(SEED_N)
    counts = Counter(seed_text)
    itos = ["<UNK>"] + [ch for ch, _ in counts.most_common(VOCAB_V - 1)]
    vocab = {ch: i for i, ch in enumerate(itos)}
    vget = vocab.get

    def encode_text(text: str) -> np.ndarray:
        return np.fromiter((vget(ch, 0) for ch in text), dtype=np.int32, count=len(text))

    def read_encoded_slice(offset: int, length: int) -> np.ndarray:
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            rem = offset
            while rem > 0:
                rd = f.read(min(rem, 16_000_000))
                if not rd:
                    raise RuntimeError(f"Corpus ended before {offset:,}")
                rem -= len(rd)
            txt = f.read(length)
        return encode_text(txt)

    def decode_tokens(seq) -> str:
        return "".join(itos[int(i)] if 0 <= int(i) < len(itos) else "?" for i in seq)

    seed_toks = encode_text(seed_text)
    del seed_text, counts
    gc.collect()

    dev = read_encoded_slice(DEV_OFFSET, EVAL_LEN)
    ev = read_encoded_slice(EVAL_OFFSET, EVAL_LEN)
    ev_unk_pct = 100.0 * float(np.mean(ev == 0))
    ev_v2k_unk_pct = 100.0 * float(np.mean((ev == 0) | (ev >= 2000)))
    print(
        f"Prepared V={VOCAB_V} vocab & slices [{time.perf_counter()-t_global:.1f}s] | "
        f"@260M UNK={ev_unk_pct:.2f}% (vs V=2000 UNK={ev_v2k_unk_pct:.2f}%) | "
        f"RSS={current_rss_bytes()/(1024**2):.1f} MB",
        flush=True,
    )

    # 2. Build 64M V=4096 symbol layer (16M seed + 3x16M streaming ingest; skip legacy StatefulContextPolicy)
    bundle = StructuralSubstrateBundle.fit_parallel(
        seed_toks,
        vocab_size=VOCAB_V,
        vocab=vocab,
        itos=itos,
        n_workers=16,
        max_memory_bytes=16 * (1024 ** 3),
        n_classes=360,
        n_singletons=36,
        feature_dim=64,
        max_chunks_per_level=6000,
        n_chunk_classes=224,
        post_fit_noise_prune=True,
        noise_prune_pressure=0.22,
        stage_slice=400_000,
        use_learnable_ssm=False,
        seed=42,
    )
    bundle.temp_memory.enable_circuit_field(use_gpu=True, fit_tokens=seed_toks[:4_000_000])
    # Detach legacy StatefulContextPolicy during 3x16M ingest so we train our GPU-BPTT P0-1+P0-2 SSM directly on full 64M
    bundle.stateful_policy = None
    bundle.temp_memory._stateful_policy = None
    print(
        f"[16M V={VOCAB_V} Seed] wall={bundle.metadata['wall_seconds']}s | "
        f"chunks={bundle.chunk_memory.n_total_units:,} (multi={bundle.chunk_memory.n_multi_chunks:,}) | "
        f"RSS={current_rss_bytes()/(1024**2):.1f} MB",
        flush=True,
    )

    toks_64m = np.empty(TARGET_N, dtype=np.int32)
    toks_64m[: len(seed_toks)] = seed_toks
    pos = len(seed_toks)
    del seed_toks
    gc.collect()

    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        rem = SEED_N
        while rem > 0:
            rem -= len(f.read(min(rem, 16_000_000)))
        for ci in range(3):
            txt = f.read(CHUNK_N)
            if not txt:
                break
            ctoks = encode_text(txt)
            del txt
            toks_64m[pos : pos + len(ctoks)] = ctoks
            pos += len(ctoks)
            rep = bundle.ingest_stream_chunk(ctoks, n_workers=12, max_memory_bytes=16 * (1024 ** 3))
            del ctoks
            gc.collect()
            print(
                f"  [Ingest chunk {ci+2}/4 -> {pos//1_000_000}M] {rep['seconds']}s | "
                f"RSS={current_rss_bytes()/(1024**2):.1f} MB",
                flush=True,
            )
    toks_64m = toks_64m[:pos]

    # 3. Align 64M dual-stream self-discovered chunks & train P0-1+P0-2 SSM (H=768x2, FW=64 slow-floor, V=4096)
    cm = bundle.chunk_memory
    c_vocab = cm.n_multi_chunks + 1
    emb_raw = bundle.class_policy.feature_map.embeddings_
    emb_norm = emb_raw / np.maximum(np.linalg.norm(emb_raw, axis=1, keepdims=True), 1e-8)
    emb_cn = bundle.class_policy.emb_centered_norm_
    priors = bundle.class_policy.feature_map.global_probs_
    rare_mask = (priors > 0) & (priors <= np.median(priors[1:]))
    rare_mask[0] = False

    t_cm = time.perf_counter()
    chks_64m = cm.match_stream_chunks(toks_64m, slice_len=4_000_000)
    print(
        f"Aligned 64M V={VOCAB_V} dual-stream chunks in {time.perf_counter()-t_cm:.1f}s | "
        f"C={c_vocab} | chunk_hit={100.0*np.mean(chks_64m>0):.1f}% | RSS={current_rss_bytes()/(1024**2):.1f} MB",
        flush=True,
    )

    ssm = LearnableStateSpacePolicy(
        vocab_size=VOCAB_V,
        embed_dim=64,
        hidden_dim=768,
        num_layers=2,
        fw_dim=64,
        chunk_vocab_size=c_vocab,
        bptt_len=96,
        batch_streams=256,
        lr=0.0032,
        min_lr_ratio=0.15,
        weight_decay=1e-5,
        grad_clip=1.5,
        slow_anchor_strength=0.42,
        fw_anchor_blend=0.0,
        use_gpu=True,
        seed=42,
    )
    print(
        f"\n=== Training 64M V={VOCAB_V} P0-1(slow-floor FW64)+P0-2(DualChunk+H768x2) SSM (2 epochs = 128M tok) ===",
        flush=True,
    )
    t_train = time.perf_counter()
    ssm.fit_stream(toks_64m, chunk_ids=chks_64m, init_embeddings=emb_raw, epochs=2, log_every=250)
    vram_peak = ssm.vram_allocated_mb
    train_sec = time.perf_counter() - t_train
    ssm.close_gpu()
    del toks_64m, chks_64m
    gc.collect()

    ktok_s = ssm.total_tokens_trained / max(train_sec, 1e-6) / 1e3
    print(
        f"Finished 64M V={VOCAB_V} GPU BPTT in {train_sec:.1f}s ({ktok_s:.1f} ktok/s) | "
        f"backend={ssm.train_backend} | active_VRAM~{vram_peak + 488.3:.1f} MB | "
        f"params={ssm.param_bytes()/1024**2:.2f} MB | total_tokens={ssm.total_tokens_trained:,} | "
        f"final_loss={ssm.last_loss_:.4f} | RSS={current_rss_bytes()/(1024**2):.1f} MB",
        flush=True,
    )

    # 4. Inspect tau spectra & slow-floor fast-weight gate distribution on @257M dev
    bundle.stateful_policy = ssm
    bundle.temp_memory.attach_stateful_policy(ssm)
    bundle.temp_memory.reset()
    z1_trace, z2_trace = [], []
    lam_tr, eta_tr, og_tr = [], [], []
    h_dim = ssm.hidden_dim
    dm = ssm.fw_dim
    dev_chks = cm.match_stream_chunks(dev)
    for i in range(4940, 5600):
        sym = int(dev[i])
        x = ssm.E_[sym].copy() if (0 < sym < VOCAB_V) else np.zeros(ssm.embed_dim, dtype=np.float32)
        c_loc = int(dev_chks[i])
        if c_loc > 0:
            x = x + ssm.E_chunk_[c_loc]
        xg1 = x @ ssm.W_x_ + ssm.b_x_
        sg1 = ssm.state.s @ ssm.U_s_ + ssm.b_s_
        z1 = 1.0 / (1.0 + np.exp(-np.clip(xg1[:h_dim] + sg1[:h_dim], -16.0, 16.0)))
        r1 = 1.0 / (1.0 + np.exp(-np.clip(xg1[h_dim:2*h_dim] + sg1[h_dim:2*h_dim], -16.0, 16.0)))
        ht1 = np.tanh(xg1[2*h_dim:] + r1 * sg1[2*h_dim:])
        s1_next = (1.0 - z1) * ssm.state.s + z1 * ht1
        xg2 = s1_next @ ssm.W_x2_ + ssm.b_x2_
        sg2 = ssm.state.s2 @ ssm.U_s2_ + ssm.b_s2_
        z2 = 1.0 / (1.0 + np.exp(-np.clip(xg2[:h_dim] + sg2[:h_dim], -16.0, 16.0)))
        qvg = s1_next @ ssm.W_qvg_ + ssm.b_qvg_
        if i >= 5000:
            z1_trace.append(z1)
            z2_trace.append(z2)
            lam_tr.append(1.0 / (1.0 + np.exp(-np.clip(qvg[2*dm], -16.0, 16.0))))
            eta_tr.append(1.0 / (1.0 + np.exp(-np.clip(qvg[2*dm+1], -16.0, 16.0))))
            og_tr.append(1.0 / (1.0 + np.exp(-np.clip(qvg[2*dm+2], -16.0, 16.0))))
        cid = (c_loc + VOCAB_V - 1) if c_loc > 0 else 0
        ssm.observe(sym, chunk_id=cid, source="external")

    for lname, ztr in [("Layer-1 (Fast/Mid)", z1_trace), ("Layer-2 (Slow Macro)", z2_trace)]:
        Z = np.stack(ztr, axis=0)
        tau = 1.0 / np.clip(Z.mean(axis=0), 1e-3, 1.0)
        pcts = np.percentile(tau, [5, 25, 50, 75, 95])
        print(
            f"  {lname:20s} tau [p5..p95] = [{pcts[0]:.2f}, {pcts[1]:.2f}, {pcts[2]:.2f}, {pcts[3]:.2f}, {pcts[4]:.2f}]",
            flush=True,
        )
    print(
        f"  Fast-Weight gates on @257M: lam={np.mean(lam_tr):.4f} [{np.min(lam_tr):.3f},{np.max(lam_tr):.3f}] | "
        f"eta={np.mean(eta_tr):.4f} [{np.min(eta_tr):.3f},{np.max(eta_tr):.3f}] | "
        f"read_gate o={np.mean(og_tr):.4f} [{np.min(og_tr):.3f},{np.max(og_tr):.3f}]",
        flush=True,
    )

    # 5. Strict @257M dev alpha selection -> single frozen @260M blind report (BOTH Full V=4096 & V=2000 subset!)
    alphas = [0.00, 0.45, 0.55, 0.65, 0.75, 0.85]

    def evaluate_blend(tokens: np.ndarray, blend_alphas: list, label: str):
        ctrl_base = bundle.make_controller(use_arbitrator=True)
        ctrl_base.stateful_policy = None
        blend_full = {float(al): [0, 0, 0, 0, 0] for al in blend_alphas}
        blend_v2k = {float(al): [0, 0, 0, 0, 0] for al in blend_alphas}
        ssm_full = [0, 0, 0, 0, 0]
        ssm_v2k = [0, 0, 0, 0, 0]
        ce_full = ce_v2k_raw = ce_v2k_norm = 0.0
        n_full = n_v2k = 0
        rr_ce_sum = 0.0
        rr_t1 = rr_t8 = rr_n = 0
        last_seen = {}

        bundle.temp_memory.reset()
        slow = np.zeros(emb_raw.shape[1], dtype=float)
        for p_idx in range(4616, 5000):
            sym = int(tokens[p_idx])
            if sym > 0:
                last_seen[sym] = p_idx
            bundle.temp_memory.observe(sym, source="external")
            slow = bundle.class_policy.update_slow_state(slow, sym)

        t_eval = time.perf_counter()
        for i in range(5000, 11000):
            tgt = int(tokens[i])
            if tgt > 0:
                ctx = tokens[i - 8 : i]
                acts, probs, _ = ctrl_base.decide_distribution(ctx, slow, k=16)
                s_eff = ssm._effective_readout_vec(ssm.state, self_mode=False)
                logits = s_eff @ ssm.W_out_ + ssm.b_out_
                logits[0] = -1e4
                lmax = float(logits.max())
                ex = np.exp(np.clip(logits - lmax, -30.0, 0.0))
                p_full = ex / max(float(ex.sum()), 1e-12)
                nll = float(-np.log(max(float(p_full[tgt]), 1e-9)))
                ce_full += nll

                top16 = np.argpartition(-logits, 16)[:16]
                top16 = top16[np.argsort(-logits[top16])]
                sm_map = {int(idx): float(p_full[int(idx)]) for idx in top16 if int(idx) > 0}
                sm_sum = sum(sm_map.values())
                if sm_sum > 1e-12:
                    sm_map = {a: p / sm_sum for a, p in sm_map.items()}
                ca_only = [int(idx) for idx in top16[:8]]
                base_map = {int(a): float(p) for a, p in zip(acts, probs)}
                n_full += 1

                def _acc(stats, ranked):
                    if ranked[0] == tgt: stats[0] += 1
                    if tgt in ranked: stats[1] += 1
                    if bundle.class_policy.sym_to_class_[ranked[0]] == bundle.class_policy.sym_to_class_[tgt]: stats[2] += 1
                    if float(np.dot(emb_norm[ranked[0]], emb_norm[tgt])) >= 0.45: stats[3] += 1
                    if float(np.dot(emb_cn[ranked[0]], emb_cn[tgt])) >= 0.20: stats[4] += 1

                _acc(ssm_full, ca_only)

                in_v2k = (1 <= tgt < 2000)
                if in_v2k:
                    n_v2k += 1
                    ce_v2k_raw += nll
                    p2k_sum = max(float(p_full[1:2000].sum()), 1e-12)
                    ce_v2k_norm += float(-np.log(max(float(p_full[tgt]) / p2k_sum, 1e-9)))
                    _acc(ssm_v2k, ca_only)

                lag = i - last_seen.get(tgt, -10**9)
                if rare_mask[tgt] and 33 <= lag <= 384:
                    rr_ce_sum += nll
                    rr_t1 += int(ca_only[0] == tgt)
                    rr_t8 += int(tgt in ca_only)
                    rr_n += 1

                prev1 = int(ctx[-1]) if len(ctx) else 0
                union_keys = set(base_map)
                for a, sp_p in sm_map.items():
                    if a in union_keys:
                        continue
                    p_loc, _ = ctrl_base._local_pair_prob_and_lift(prev1, a) if prev1 > 0 else (0.0, 0.0)
                    if p_loc > 0.0 or sp_p >= 0.03:
                        union_keys.add(a)

                for al in blend_alphas:
                    if al <= 0.0:
                        ranked = acts[:8]
                    else:
                        scored = [(a, (1.0 - al) * base_map.get(a, 0.0) + al * sm_map.get(a, 0.0)) for a in union_keys]
                        scored.sort(key=lambda kv: kv[1], reverse=True)
                        ranked = [a for a, _ in scored[:8]]
                    _acc(blend_full[float(al)], ranked)
                    if in_v2k:
                        _acc(blend_v2k[float(al)], ranked)

                last_seen[tgt] = i

            bundle.temp_memory.observe(tgt, source="external")
            slow = bundle.class_policy.update_slow_state(slow, tgt)

        print(f"\n=== {label} ev[5000:11000] (n_full={n_full}, n_v2k={n_v2k}, {time.perf_counter()-t_eval:.1f}s) ===", flush=True)
        if n_full:
            print(
                f"  [Full V=4096] SSM-only CE={ce_full/n_full:.4f} PPL={np.exp(ce_full/n_full):.2f} | "
                f"RareRepeat(33..384, n={rr_n}) PPL={np.exp(rr_ce_sum/max(rr_n,1)):.2f} "
                f"Top1={100*rr_t1/max(rr_n,1):.2f}% Top8={100*rr_t8/max(rr_n,1):.2f}%",
                flush=True,
            )
            print(f"  [Full V=4096] SSM-only Top1/Top8/SameCls/SemAlign/CenteredSem={np.round(100*np.asarray(ssm_full)/n_full, 2)}", flush=True)
            for al in blend_alphas:
                print(f"  [Full V=4096] UnionBlend alpha={al:.2f} Top1/Top8/SameCls/SemAlign/CenteredSem={np.round(100*np.asarray(blend_full[float(al)])/n_full, 2)}", flush=True)
        if n_v2k:
            print(
                f"  [V=2000 Sub n={n_v2k}] SSM-only CE_raw={ce_v2k_raw/n_v2k:.4f} (PPL={np.exp(ce_v2k_raw/n_v2k):.2f}) | "
                f"CE_v2k_norm={ce_v2k_norm/n_v2k:.4f} (PPL={np.exp(ce_v2k_norm/n_v2k):.2f})",
                flush=True,
            )
            print(f"  [V=2000 Sub n={n_v2k}] SSM-only Top1/Top8/SameCls/SemAlign/CenteredSem={np.round(100*np.asarray(ssm_v2k)/n_v2k, 2)}", flush=True)
            for al in blend_alphas:
                print(f"  [V=2000 Sub n={n_v2k}] UnionBlend alpha={al:.2f} Top1/Top8/SameCls/SemAlign/CenteredSem={np.round(100*np.asarray(blend_v2k[float(al)])/n_v2k, 2)}", flush=True)
        return blend_full, n_full

    dev_stats, dev_n = evaluate_blend(dev, alphas, "Development alpha selection @257M")
    best_al = max(
        alphas,
        key=lambda al: 0.5 * dev_stats[float(al)][0] + 0.3 * dev_stats[float(al)][1] + 0.2 * dev_stats[float(al)][2],
    )
    print(f"\n  Frozen alpha selected on @257M dev only: {best_al:.2f} (n={dev_n})", flush=True)

    evaluate_blend(ev, [best_al], "Blind quality report @260M")

    # 6. Save V=4096 P0-1+P0-2 checkpoint
    bundle.metadata["union_blend_alpha_dev"] = float(best_al)
    bundle.metadata["union_blend_selection_offset"] = int(DEV_OFFSET)
    bundle.metadata["ssm_architecture"] = "V4096_2L_H768_FW64SlowFloor_DualChunk"
    sv = bundle.save(ckpt_out, compresslevel=1)
    print(f"Saved V=4096 P0-1+P0-2 checkpoint: {sv} | RSS={current_rss_bytes()/(1024**2):.1f} MB", flush=True)

    # 7. P1/P2/P3 100-char blind rollouts at @260M
    cf = bundle.temp_memory.circuit_field
    if cf is not None and not getattr(cf, "gpu_enabled", False):
        bundle.temp_memory.enable_circuit_field(use_gpu=True)
        cf = bundle.temp_memory.circuit_field

    wiki_sp = 0
    for j in range(len(ev) - 180):
        if decode_tokens(ev[j : j + 6]) == "大上海时代广":
            wiki_sp = j
            break
    test_list = [("P1(立冬/日历)", 16367), ("P2(个税/BI数据)", 19122)]
    if wiki_sp > 0:
        test_list.append(("P3(Wiki-上海广场)", wiki_sp))

    print(f"\n=== 64M V=4096 P0-1+P0-2 Blind Rollouts @260M (frozen alpha={best_al:.2f}, depth=2, beam=5) ===", flush=True)
    for p_idx, (label, sp_pos) in enumerate(test_list, 1):
        p_toks = [int(x) for x in ev[sp_pos : sp_pos + 60]]
        gt_toks = [int(x) for x in ev[sp_pos + 60 : sp_pos + 160]]
        ptree = bundle.build_prompt_stage_tree(p_toks)
        ctrl_g = bundle.make_controller(use_arbitrator=True, stage_tree=ptree, lookahead_depth=2, lookahead_beam=5)
        ctrl_g.stateful_weight = float(best_al)
        ctrl_g.self_stateful_beta = 0.0
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
            acts, probs, _ = ctrl_g.decide_with_lookahead(seq[-8:], hg, k=8, depth=2, beam_width=5)
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
            f"3gram_rep={100*rep3:.1f}% | spaces={gs.count(' ')} | cos(p)={cp_c:.3f} cos(gt)={cg_c:.3f}"
        )
        print(f"  PROMPT: {decode_tokens(p_toks)}")
        print(f"  TEXT  : {gs}", flush=True)

    if cf is not None:
        cf.close()
    print(f"\nTotal wall time: {time.perf_counter() - t_global:.1f}s", flush=True)


if __name__ == "__main__":
    main()
