"""Train & Blind-Evaluate 64M P0-1 + P0-2 LearnableStateSpacePolicy:
  - P0-1: Fast-Weight Gated Delta-Rule Associative State (fw_dim=64, learned lam/eta/read-gate o_t)
  - P0-2A: Dual-Stream Multi-Granularity Input (Atom E[sym] + HierarchicalChunkMemory E_chunk[chk])
  - P0-2B: RX 9070 XT VRAM Scale-Up (H=768, 2-Layer Residual RC, T=96, B=256, 2 epochs = 128M tokens)

Protocol:
  - Train on [0, 64M), save to NEW checkpoint checkpoints/wiki_zhihu_64m_v2000_p01_p02.pkl.gz.
  - Select UnionBlend alpha strictly on @257M dev.
  - Report frozen-alpha blind metrics & P1/P2/P3 rollouts once on sealed @260M ev[5000:11000].
"""
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
ckpt_in = os.path.join(ROOT, "checkpoints", "wiki_zhihu_64m_v2000.pkl.gz")
ckpt_out = os.path.join(ROOT, "checkpoints", "wiki_zhihu_64m_v2000_p01_p02.pkl.gz")
TARGET_N = 64_000_000
DEV_OFFSET = 257_000_000
EVAL_OFFSET = 260_000_000
EVAL_LEN = 20_000


def ngram_repeat_rate(seq, n_g=3):
    grams = [tuple(seq[j : j + n_g]) for j in range(len(seq) - n_g + 1)]
    return 1.0 - (len(set(grams)) / float(max(len(grams), 1)))


def main():
    t_global = time.perf_counter()
    bundle = StructuralSubstrateBundle.load(ckpt_in)
    vocab = bundle.vocab
    itos = bundle.itos
    VOCAB_V = len(itos)
    cm = bundle.chunk_memory
    c_vocab = cm.n_multi_chunks + 1
    emb_raw = bundle.class_policy.feature_map.embeddings_
    emb_norm = emb_raw / np.maximum(np.linalg.norm(emb_raw, axis=1, keepdims=True), 1e-8)
    emb_cn = bundle.class_policy.emb_centered_norm_
    priors = bundle.class_policy.feature_map.global_probs_
    rare_mask = (priors > 0) & (priors <= np.median(priors[1:]))
    rare_mask[0] = False

    def encode_text(text: str) -> np.ndarray:
        vget = vocab.get
        return np.fromiter((vget(ch, 0) for ch in text), dtype=np.int32, count=len(text))

    def read_encoded_slice(offset: int, length: int) -> np.ndarray:
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            remaining = offset
            while remaining > 0:
                chunk = f.read(min(remaining, 16_000_000))
                if not chunk:
                    raise RuntimeError(f"Corpus ended before offset {offset:,}")
                remaining -= len(chunk)
            text = f.read(length)
        if len(text) != length:
            raise RuntimeError(f"Expected {length:,} chars at offset {offset:,}, got {len(text):,}")
        return encode_text(text)

    def decode_tokens(seq) -> str:
        return "".join(itos[int(i)] if 0 <= int(i) < len(itos) else "?" for i in seq)

    print(
        f"Loaded 64M symbol bundle (V={VOCAB_V}, D={emb_raw.shape[1]}, chunk_vocab={c_vocab}) "
        f"in {time.perf_counter()-t_global:.1f}s | RSS={current_rss_bytes()/(1024**2):.1f} MB",
        flush=True,
    )

    # 1. Encode 64M stream & align self-discovered HierarchicalChunkMemory stream
    toks_64m = np.empty(TARGET_N, dtype=np.int32)
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        pos = 0
        while pos < TARGET_N:
            take = min(16_000_000, TARGET_N - pos)
            txt = f.read(take)
            if not txt:
                break
            enc = encode_text(txt)
            del txt
            toks_64m[pos : pos + len(enc)] = enc
            pos += len(enc)
            del enc
        toks_64m = toks_64m[:pos]

    t_cm = time.perf_counter()
    chks_64m = cm.match_stream_chunks(toks_64m, slice_len=4_000_000)
    print(
        f"Aligned 64M dual-stream chunks in {time.perf_counter()-t_cm:.1f}s | "
        f"chunk_hit={100.0*np.mean(chks_64m>0):.1f}% | RSS={current_rss_bytes()/(1024**2):.1f} MB",
        flush=True,
    )

    dev = read_encoded_slice(DEV_OFFSET, EVAL_LEN)
    ev = read_encoded_slice(EVAL_OFFSET, EVAL_LEN)
    gc.collect()

    # 2. Train P0-1 + P0-2 SSM (H=768, L=2, fw_dim=64, dual-stream chunks, T=96, B=256, 2 epochs)
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
        f"\n=== Training 64M P0-1+P0-2 SSM (V={VOCAB_V}, C={c_vocab}, D=64, H=768x2, FW=64, T=96, B=256, 2 epochs) ===",
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
        f"Finished 64M P0-1+P0-2 GPU BPTT in {train_sec:.1f}s ({ktok_s:.1f} ktok/s) | "
        f"backend={ssm.train_backend} | active_VRAM~{vram_peak + 488.3:.1f} MB | "
        f"params={ssm.param_bytes()/1024**2:.2f} MB | total_tokens={ssm.total_tokens_trained:,} | "
        f"final_loss={ssm.last_loss_:.4f} | RSS={current_rss_bytes()/(1024**2):.1f} MB",
        flush=True,
    )

    # 3. Inspect Layer-1 / Layer-2 tau spectra + Fast-Weight gate distribution on @257M dev
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

    # 4. Strict @257M dev alpha selection -> single frozen @260M sealed blind report
    alphas = [0.00, 0.45, 0.55, 0.65, 0.75, 0.85]

    def evaluate_blend(tokens: np.ndarray, blend_alphas: list, label: str):
        ctrl_base = bundle.make_controller(use_arbitrator=True)
        ctrl_base.stateful_policy = None
        blend_stats = {float(al): [0, 0, 0, 0, 0] for al in blend_alphas}
        ssm_stats = [0, 0, 0, 0, 0]
        ssm_ce_sum = 0.0
        n_eval = 0
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
                ssm_ce_sum += nll

                top16 = np.argpartition(-logits, 16)[:16]
                top16 = top16[np.argsort(-logits[top16])]
                sm_map = {int(idx): float(p_full[int(idx)]) for idx in top16 if int(idx) > 0}
                sm_sum = sum(sm_map.values())
                if sm_sum > 1e-12:
                    sm_map = {a: p / sm_sum for a, p in sm_map.items()}
                ca_only = [int(idx) for idx in top16[:8]]
                base_map = {int(a): float(p) for a, p in zip(acts, probs)}
                n_eval += 1

                if ca_only[0] == tgt: ssm_stats[0] += 1
                if tgt in ca_only: ssm_stats[1] += 1
                if bundle.class_policy.sym_to_class_[ca_only[0]] == bundle.class_policy.sym_to_class_[tgt]: ssm_stats[2] += 1
                if float(np.dot(emb_norm[ca_only[0]], emb_norm[tgt])) >= 0.45: ssm_stats[3] += 1
                if float(np.dot(emb_cn[ca_only[0]], emb_cn[tgt])) >= 0.20: ssm_stats[4] += 1

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
                    st = blend_stats[float(al)]
                    if ranked[0] == tgt: st[0] += 1
                    if tgt in ranked: st[1] += 1
                    if bundle.class_policy.sym_to_class_[ranked[0]] == bundle.class_policy.sym_to_class_[tgt]: st[2] += 1
                    if float(np.dot(emb_norm[ranked[0]], emb_norm[tgt])) >= 0.45: st[3] += 1
                    if float(np.dot(emb_cn[ranked[0]], emb_cn[tgt])) >= 0.20: st[4] += 1

                last_seen[tgt] = i

            bundle.temp_memory.observe(tgt, source="external")
            slow = bundle.class_policy.update_slow_state(slow, tgt)

        print(f"\n=== {label} ev[5000:11000] (n={n_eval}, {time.perf_counter()-t_eval:.1f}s) ===", flush=True)
        if n_eval:
            print(
                f"  SSM-only CE={ssm_ce_sum/n_eval:.4f} PPL={np.exp(ssm_ce_sum/n_eval):.2f} | "
                f"RareRepeat(33..384, n={rr_n}) PPL={np.exp(rr_ce_sum/max(rr_n,1)):.2f} "
                f"Top1={100*rr_t1/max(rr_n,1):.2f}% Top8={100*rr_t8/max(rr_n,1):.2f}%",
                flush=True,
            )
            vals = np.asarray(ssm_stats, dtype=np.float64) / n_eval
            print(f"  SSM-only Top1/Top8/SameCls/SemAlign/CenteredSem={np.round(100*vals, 2)}", flush=True)
            for al in blend_alphas:
                vals = np.asarray(blend_stats[float(al)], dtype=np.float64) / n_eval
                print(f"  UnionBlend alpha={al:.2f} Top1/Top8/SameCls/SemAlign/CenteredSem={np.round(100*vals, 2)}", flush=True)
        return blend_stats, n_eval

    dev_stats, dev_n = evaluate_blend(dev, alphas, "Development alpha selection @257M")
    best_al = max(
        alphas,
        key=lambda al: 0.5 * dev_stats[float(al)][0] + 0.3 * dev_stats[float(al)][1] + 0.2 * dev_stats[float(al)][2],
    )
    print(f"\n  Frozen alpha selected on @257M dev only: {best_al:.2f} (n={dev_n})", flush=True)

    evaluate_blend(ev, [best_al], "Blind quality report @260M")

    # 5. Save to NEW checkpoint (preserves original wiki_zhihu_64m_v2000.pkl.gz)
    bundle.metadata["union_blend_alpha_dev"] = float(best_al)
    bundle.metadata["union_blend_selection_offset"] = int(DEV_OFFSET)
    bundle.metadata["ssm_architecture"] = "2L_H768_FW64_DualChunk"
    sv = bundle.save(ckpt_out, compresslevel=1)
    print(f"Saved P0-1+P0-2 checkpoint: {sv} | RSS={current_rss_bytes()/(1024**2):.1f} MB", flush=True)

    # 6. P1/P2/P3 100-char blind rollouts at @260M with frozen alpha
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

    print(f"\n=== 64M P0-1+P0-2 Blind Rollouts @260M (frozen alpha={best_al:.2f}, depth=2, beam=5) ===", flush=True)
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
