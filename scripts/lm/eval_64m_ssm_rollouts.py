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

corpus_path = r"C:\workspace\code\data\wiki_zhihu\corpus.txt"
ckpt_64m = os.path.join(ROOT, "checkpoints", "wiki_zhihu_64m_v2000.pkl.gz")
EVAL_OFFSET = 260_000_000
EVAL_LEN = 20_000

bundle = StructuralSubstrateBundle.load(ckpt_64m)
vocab = bundle.vocab
itos = bundle.itos
emb_raw = bundle.class_policy.feature_map.embeddings_
emb_cn = bundle.class_policy.emb_centered_norm_

def encode_text(text: str) -> np.ndarray:
    vget = vocab.get
    return np.fromiter((vget(ch, 0) for ch in text), dtype=np.int32, count=len(text))

def decode_tokens(seq) -> str:
    return "".join(itos[int(i)] if 0 <= int(i) < len(itos) else "?" for i in seq)

def ngram_repeat_rate(seq, n_g=3):
    grams = [tuple(seq[j:j + n_g]) for j in range(len(seq) - n_g + 1)]
    return 1.0 - (len(set(grams)) / float(max(len(grams), 1)))

with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
    rem = EVAL_OFFSET
    while rem > 0:
        rd = f.read(min(rem, 16_000_000))
        if not rd:
            break
        rem -= len(rd)
    ev_text = f.read(EVAL_LEN)
ev = encode_text(ev_text)

wiki_sp = 0
for j in range(len(ev) - 180):
    if decode_tokens(ev[j:j + 6]) == "大上海时代广":
        wiki_sp = j
        break

test_list = [("P1(立冬/日历)", 16367), ("P2(个税/BI数据)", 19122)]
if wiki_sp > 0:
    test_list.append(("P3(Wiki-上海广场)", wiki_sp))

# Re-enable circuit field GPU cleanly via enable_circuit_field(use_gpu=True) if needed
cf = bundle.temp_memory.circuit_field
if cf is not None and not getattr(cf, "gpu_enabled", False):
    bundle.temp_memory.enable_circuit_field(use_gpu=True)
    cf = bundle.temp_memory.circuit_field

print(
    f"Loaded 64M SSM checkpoint | sp={type(bundle.stateful_policy).__name__} "
    f"H={bundle.stateful_policy.hidden_dim} | cf_gpu={getattr(cf, 'gpu_enabled', False)}",
    flush=True,
)

for alpha in (0.45,):
    print(f"\n=== 64M LearnableStateSpacePolicy (H=512, T=48 BPTT) + UnionBlend (alpha={alpha:.2f}, self_stateful_beta=0.0) ===", flush=True)
    for p_idx, (label, sp_pos) in enumerate(test_list, 1):
        p_toks = [int(x) for x in ev[sp_pos:sp_pos + 60]]
        gt_toks = [int(x) for x in ev[sp_pos + 60:sp_pos + 160]]
        ptree = bundle.build_prompt_stage_tree(p_toks)
        ctrl_g = bundle.make_controller(
            use_arbitrator=True, stage_tree=ptree, lookahead_depth=2, lookahead_beam=5,
        )
        ctrl_g.stateful_weight = alpha
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

if cf is not None:
    cf.close()
