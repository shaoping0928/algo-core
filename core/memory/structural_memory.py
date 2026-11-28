"""Task-agnostic Higher-Order Structural Memory:
  1. ``HierarchicalChunkMemory`` — Recursive MDL/PMI chunk induction (macro-options /
     motifs / multi-symbol phrases) with chunk-to-chunk transition & intra-chunk
     completion memory.
  2. ``StructuralFrameMemory`` — Directional non-adjacent open->close dependency
     discovery (key->door, brackets, cause->effect frames) with an online
     pushdown working register and learned gap-timing curves.

Zero language rules, zero word segmenters, and zero task priors: both modules
operate purely on discrete symbol/action ID streams.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparse_index import SparseFeatureActionIndex


class HierarchicalChunkMemory:
    """Multi-scale recursive chunk induction + chunk-level transition & completion memory."""

    def __init__(
        self,
        vocab_size: int,
        *,
        max_levels: int = 2,
        max_chunks_per_level: int = 1200,
        min_pair_count: int = 20,
        min_pmi: float = 2.0,
        max_chunk_len: int = 4,
        chunk_ctx_len: int = 2,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.max_levels = max(int(max_levels), 1)
        self.max_chunks_per_level = max(int(max_chunks_per_level), 16)
        self.min_pair_count = max(int(min_pair_count), 3)
        self.min_pmi = float(min_pmi)
        self.max_chunk_len = max(int(max_chunk_len), 2)
        self.chunk_ctx_len = max(int(chunk_ctx_len), 1)

        # Chunk IDs: 0..vocab_size-1 are atomic symbols; >= vocab_size are discovered chunks
        self.n_total_units: int = self.vocab_size
        self.chunk_to_atoms_: Dict[int, Tuple[int, ...]] = {
            i: (i,) for i in range(self.vocab_size)
        }
        self.atoms_to_chunk_: Dict[Tuple[int, ...], int] = {
            (i,): i for i in range(self.vocab_size)
        }
        self.level_merges_: List[Dict[Tuple[int, int], int]] = []
        # Prefix index: atom tuple prefix -> list of (next_atom, full_chunk_id)
        self.prefix_to_chunks_: Dict[Tuple[int, ...], List[Tuple[int, int]]] = defaultdict(list)
        self.chunk_counts_: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self.chunk_index: Optional[SparseFeatureActionIndex] = None
        self._fitted: bool = False

    def _merge_stream_once(
        self,
        stream: np.ndarray,
        pair_map: Dict[Tuple[int, int], int],
    ) -> np.ndarray:
        """Fast greedy left-to-right non-overlapping merge for one hierarchy level."""
        if not pair_map or len(stream) < 2:
            return stream
        s_list = stream.tolist()
        n = len(s_list)
        out: List[int] = []
        i = 0
        get = pair_map.get
        while i < n:
            if i + 1 < n:
                merged = get((s_list[i], s_list[i + 1]))
                if merged is not None:
                    out.append(merged)
                    i += 2
                    continue
            out.append(s_list[i])
            i += 1
        return np.asarray(out, dtype=np.int32)

    def fit(self, tokens: Sequence[int]) -> "HierarchicalChunkMemory":
        """Recursively discover high-PMI structural chunks and fit chunk-level transitions."""
        arr = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        if len(arr) < 4:
            return self

        self.chunk_to_atoms_ = {i: (i,) for i in range(self.vocab_size)}
        self.atoms_to_chunk_ = {(i,): i for i in range(self.vocab_size)}
        self.level_merges_ = []
        next_cid = self.vocab_size
        cur_stream = arr

        for _lvl in range(self.max_levels):
            n_cur = len(cur_stream)
            if n_cur < 4:
                break
            n_units = next_cid
            u_counts = np.bincount(cur_stream, minlength=n_units).astype(float)
            u_counts[0] = 0.0
            tot_u = max(float(u_counts.sum()), 1.0)
            p_u = (u_counts + 1.0) / (tot_u + float(n_units))

            # Vectorized adjacent pair counting (skip <UNK>=0)
            valid = (cur_stream[:-1] > 0) & (cur_stream[1:] > 0)
            if not np.any(valid):
                break
            a_arr = cur_stream[:-1][valid].astype(np.int64)
            b_arr = cur_stream[1:][valid].astype(np.int64)
            stride_base = int(n_units)
            codes = a_arr * stride_base + b_arr
            uniq_codes, code_cnts = np.unique(codes, return_counts=True)
            keep = code_cnts >= self.min_pair_count
            if not np.any(keep):
                break

            inv_pairs = 1.0 / max(float(len(a_arr)), 1.0)
            scored = []
            for code_val, cnt_val in zip(uniq_codes[keep].tolist(), code_cnts[keep].tolist()):
                u = int(code_val // stride_base)
                v = int(code_val % stride_base)
                atoms_uv = self.chunk_to_atoms_[u] + self.chunk_to_atoms_[v]
                if len(atoms_uv) > self.max_chunk_len:
                    continue
                if atoms_uv in self.atoms_to_chunk_:
                    continue
                pmi = float(np.log((float(cnt_val) * inv_pairs) / max(float(p_u[u] * p_u[v]), 1e-14)))
                if pmi >= self.min_pmi:
                    # MDL compression gain proxy: frequency * positive PMI
                    mdl_gain = float(cnt_val) * pmi
                    scored.append((mdl_gain, u, v, atoms_uv))

            if not scored:
                break
            scored.sort(key=lambda item: item[0], reverse=True)
            pair_map: Dict[Tuple[int, int], int] = {}
            for _gain, u, v, atoms_uv in scored[: self.max_chunks_per_level]:
                cid = next_cid
                next_cid += 1
                pair_map[(u, v)] = cid
                self.chunk_to_atoms_[cid] = atoms_uv
                self.atoms_to_chunk_[atoms_uv] = cid

            if not pair_map:
                break
            self.level_merges_.append(pair_map)
            cur_stream = self._merge_stream_once(cur_stream, pair_map)

        self.n_total_units = next_cid
        self.chunk_counts_ = np.bincount(cur_stream, minlength=self.n_total_units).astype(float)

        # Build atom-prefix -> (next_atom, chunk_id) lookup for intra-chunk completion
        self.prefix_to_chunks_ = defaultdict(list)
        for cid in range(self.vocab_size, self.n_total_units):
            atoms = self.chunk_to_atoms_[cid]
            for pref_len in range(1, len(atoms)):
                pref = atoms[:pref_len]
                nxt_a = atoms[pref_len]
                self.prefix_to_chunks_[pref].append((nxt_a, cid))

        # Fit higher-order SparseFeatureActionIndex over compressed chunk stream
        self.chunk_index = SparseFeatureActionIndex(
            tuple(range(self.n_total_units)), prior_weight=0.0,
        )
        self.chunk_index.fit_token_stream(cur_stream, ctx_len=self.chunk_ctx_len)
        self._fitted = True
        return self

    def parse_sequence(self, tokens: Sequence[int]) -> List[int]:
        """Parse an atomic symbol sequence into discovered multi-scale chunk IDs."""
        cur = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        for pair_map in self.level_merges_:
            cur = self._merge_stream_once(cur, pair_map)
        return cur.tolist()

    @staticmethod
    def _chunk_keys(c_ctx: Sequence[int]) -> List[Tuple[int, ...]]:
        if not c_ctx:
            return []
        keys: List[Tuple[int, ...]] = [(-1, int(c_ctx[-1]))]
        for order in range(2, len(c_ctx) + 1):
            keys.append((10 + order,) + tuple(int(x) for x in c_ctx[-order:]))
        return keys

    def predict_next_atoms(
        self,
        recent_tokens: Sequence[int],
        *,
        k: int = 16,
        top_chunks: int = 24,
    ) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
        """Predict next atomic symbol via BOTH inter-chunk transitions and intra-chunk completions."""
        if not self._fitted or self.chunk_index is None or not recent_tokens:
            return [], np.zeros(0, dtype=float), dict(intra_mass=0.0, inter_mass=0.0)

        tail = [int(x) for x in recent_tokens[-24:]]
        chunks = self.parse_sequence(tail)
        if not chunks:
            return [], np.zeros(0, dtype=float), dict(intra_mass=0.0, inter_mass=0.0)

        score_map: Dict[int, float] = {}
        inter_mass = 0.0
        intra_mass = 0.0
        has_ho_inter = False
        has_ho_intra = False

        # 1. Inter-chunk boundary transition: P(next_chunk | prev_chunks) -> entry atom
        c_ctx = chunks[-self.chunk_ctx_len :]
        c_keys = self._chunk_keys(c_ctx)
        if len(c_ctx) >= 2 and c_keys[-1] in self.chunk_index._postings:
            has_ho_inter = True
        c_cand, c_probs = self.chunk_index.propose_with_probs(
            None, k=top_chunks, sparse_keys=c_keys,
        )
        for cid, pr in zip(c_cand, c_probs):
            ci = int(cid)
            p_val = float(pr)
            if ci <= 0 or p_val <= 0.0:
                continue
            entry_atom = self.chunk_to_atoms_[ci][0]
            if entry_atom > 0:
                score_map[entry_atom] = score_map.get(entry_atom, 0.0) + p_val
                inter_mass += p_val

        # 2. Intra-chunk completion: check if trailing 1..(max_chunk_len-1) atoms
        #    form a prefix of a multi-atom chunk conditioned on the chunk BEFORE that prefix!
        max_pref = min(self.max_chunk_len - 1, len(tail))
        for pref_len in range(1, max_pref + 1):
            pref = tuple(tail[-pref_len:])
            if any(x <= 0 for x in pref):
                continue
            conts = self.prefix_to_chunks_.get(pref)
            if not conts:
                continue
            # Context immediately preceding this open prefix
            before_tail = tail[:-pref_len]
            if before_tail:
                before_chunks = self.parse_sequence(before_tail)
                b_ctx = before_chunks[-self.chunk_ctx_len :]
                b_cand, b_probs = self.chunk_index.propose_with_probs(
                    None, k=max(top_chunks, 32), sparse_keys=self._chunk_keys(b_ctx),
                )
                b_prob_map = {int(c): float(p) for c, p in zip(b_cand, b_probs)}
            else:
                b_prob_map = {}

            tot_c_prior = max(float(sum(self.chunk_counts_[cid] for _, cid in conts)), 1.0)
            pref_bonus = 1.0 + 0.45 * float(pref_len - 1)
            for nxt_atom, cid in conts:
                if nxt_atom <= 0:
                    continue
                p_ctx_c = b_prob_map.get(cid, 0.0)
                if p_ctx_c > 0.0:
                    w_add = pref_bonus * p_ctx_c
                    has_ho_intra = True
                elif pref_len >= 2:
                    # Only apply unconditioned prefix prior when >=2 consecutive atoms
                    # already lock onto a 3-4 atom chunk (prevents 1-atom false completions)
                    w_add = 0.10 * pref_bonus * (float(self.chunk_counts_[cid]) / tot_c_prior)
                    has_ho_intra = True
                else:
                    w_add = 0.0
                if w_add > 1e-8:
                    score_map[nxt_atom] = score_map.get(nxt_atom, 0.0) + w_add
                    intra_mass += w_add

        if not score_map:
            return [], np.zeros(0, dtype=float), dict(intra_mass=0.0, inter_mass=0.0, chunk_conf=0.0)

        # Higher-order structural confidence in [0, 1]
        is_multi_atom_prev = (len(self.chunk_to_atoms_.get(chunks[-1], ())) >= 2)
        chunk_conf = 0.15
        if has_ho_inter:
            chunk_conf += 0.45
        if has_ho_intra:
            chunk_conf += 0.35
        if is_multi_atom_prev:
            chunk_conf += 0.25
        chunk_conf = float(np.clip(chunk_conf, 0.0, 1.0))

        ranked = sorted(score_map.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[: max(int(k), 1)]
        acts = [a for a, _ in ranked]
        probs = np.asarray([p for _, p in ranked], dtype=float)
        p_sum = float(probs.sum())
        if p_sum > 0.0:
            probs /= p_sum
        return acts, probs, dict(intra_mass=intra_mass, inter_mass=inter_mass, chunk_conf=chunk_conf)

    def top_discovered_chunks(self, itos: Sequence[str], top_n: int = 20) -> List[Tuple[str, int]]:
        """Inspect the highest-count discovered multi-symbol chunks."""
        rows = []
        for cid in range(self.vocab_size, self.n_total_units):
            cnt = int(self.chunk_counts_[cid])
            if cnt <= 0:
                continue
            atoms = self.chunk_to_atoms_[cid]
            text = "".join(itos[a] if 0 <= a < len(itos) else "?" for a in atoms)
            rows.append((cnt, text))
        rows.sort(reverse=True)
        return [(txt, cnt) for cnt, txt in rows[:top_n]]


@dataclass
class FrameSpec:
    """Learned directional non-adjacent dependency ``opener ... [gap] ... closer``."""

    opener: int
    closer: int
    pmi: float
    asymmetry: float
    mean_gap: float
    std_gap: float
    strength: float


class StructuralFrameMemory:
    """Discovers directional non-adjacent frames (open->close / key->lock) with an online register."""

    def __init__(
        self,
        vocab_size: int,
        *,
        min_gap: int = 2,
        max_gap: int = 18,
        min_pair_count: int = 15,
        min_pmi: float = 1.6,
        min_asymmetry: float = 0.62,
        max_closers_per_opener: int = 6,
        max_active_frames: int = 12,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.min_gap = max(int(min_gap), 1)
        self.max_gap = max(int(max_gap), self.min_gap + 1)
        self.min_pair_count = max(int(min_pair_count), 3)
        self.min_pmi = float(min_pmi)
        self.min_asymmetry = float(min_asymmetry)
        self.max_closers_per_opener = max(int(max_closers_per_opener), 1)
        self.max_active_frames = max(int(max_active_frames), 2)

        self.frames_by_opener_: Dict[int, List[FrameSpec]] = {}
        self._active: List[Tuple[int, int]] = []  # list of (opener_sym, age)
        self._fitted: bool = False

    def fit(
        self,
        tokens: Sequence[int],
        *,
        max_samples: int = 250000,
    ) -> "StructuralFrameMemory":
        """Discover directional non-adjacent dependencies ``opener -> closer`` + gap timing."""
        arr = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        n = len(arr)
        if n <= self.max_gap + 2:
            return self

        counts = np.bincount(arr, minlength=self.vocab_size).astype(float)
        counts[0] = 0.0
        tot = max(float(counts.sum()), 1.0)
        p_glob = (counts + 1.0) / (tot + float(self.vocab_size))

        # Count directional pairs (a -> b) within [min_gap, max_gap] + gap moments
        fwd_cnt: Dict[Tuple[int, int], int] = defaultdict(int)
        fwd_gap_sum: Dict[Tuple[int, int], float] = defaultdict(float)
        fwd_gap_sq: Dict[Tuple[int, int], float] = defaultdict(float)
        opener_trials: Dict[int, int] = defaultdict(int)

        span = n - self.max_gap - 1
        stride = max(span // max(int(max_samples), 1000), 1)
        g_lo = self.min_gap
        g_hi = self.max_gap
        arr_list = arr.tolist()

        for t in range(0, span, stride):
            a = arr_list[t]
            if a <= 0:
                continue
            opener_trials[a] += 1
            seen_b = set()
            for d in range(g_lo, g_hi + 1):
                b = arr_list[t + d]
                if b <= 0 or b == a or b in seen_b:
                    continue
                seen_b.add(b)
                key = (a, b)
                fwd_cnt[key] += 1
                fd = float(d)
                fwd_gap_sum[key] += fd
                fwd_gap_sq[key] += fd * fd

        # Probability that symbol b appears at least once in a random window of width W
        w_len = float(g_hi - g_lo + 1)
        by_opener: Dict[int, List[FrameSpec]] = defaultdict(list)
        for (a, b), cnt in fwd_cnt.items():
            if cnt < self.min_pair_count:
                continue
            rev = fwd_cnt.get((b, a), 0)
            asym = float(cnt) / float(cnt + rev)
            if asym < self.min_asymmetry:
                continue
            mean_g = fwd_gap_sum[(a, b)] / float(cnt)
            var_g = max(fwd_gap_sq[(a, b)] / float(cnt) - mean_g * mean_g, 0.25)
            std_g = float(np.sqrt(var_g))
            n_a = float(opener_trials.get(a, 1))
            p_cond = float(cnt) / max(n_a, 1.0)
            # Effective timing window reflects how tightly b concentrates around mean_gap
            w_eff = min(max(2.0 * std_g + 1.0, 1.0), w_len)
            p_null = 1.0 - float(np.power(max(1.0 - float(p_glob[b]), 1e-9), w_eff))
            pmi = float(np.log(max(p_cond, 1e-12) / max(p_null, 1e-12)))
            if pmi < self.min_pmi:
                continue
            strength = pmi * asym * min(float(np.log1p(cnt)) / 5.0, 1.5)
            by_opener[a].append(FrameSpec(
                opener=a,
                closer=b,
                pmi=pmi,
                asymmetry=asym,
                mean_gap=mean_g,
                std_gap=std_g,
                strength=strength,
            ))

        self.frames_by_opener_ = {}
        for a, specs in by_opener.items():
            specs.sort(key=lambda sp: sp.strength, reverse=True)
            self.frames_by_opener_[a] = specs[: self.max_closers_per_opener]

        self._fitted = True
        return self

    def reset(self) -> None:
        self._active.clear()

    def observe(self, symbol: int) -> None:
        """Advance online structural frame register by one observed/emitted symbol."""
        sym = int(symbol)
        # 1. Age existing frames and pop any that are closed by ``sym`` or expired
        next_active: List[Tuple[int, int]] = []
        for op, age in self._active:
            new_age = age + 1
            if new_age > self.max_gap:
                continue
            specs = self.frames_by_opener_.get(op, ())
            if any(sp.closer == sym for sp in specs):
                # Frame closed cleanly -> pop!
                continue
            next_active.append((op, new_age))
        self._active = next_active

        # 2. If ``sym`` is a learned directional opener, push it onto the register
        if sym > 0 and sym in self.frames_by_opener_:
            # Refresh if same opener already active
            self._active = [(op, age) for op, age in self._active if op != sym]
            self._active.append((sym, 0))
            if len(self._active) > self.max_active_frames:
                self._active = self._active[-self.max_active_frames :]

    def predict_closers_for_history(
        self,
        recent_history: Optional[Sequence[int]] = None,
    ) -> Dict[int, float]:
        """Compute active expected closer scores from either the live register or a rolled history."""
        if not self._fitted:
            return {}
        if recent_history is not None:
            # Stateless replay over trailing max_gap+2 symbols (safe for lookahead branching!)
            reg: List[Tuple[int, int]] = []
            tail = [int(x) for x in recent_history[-(self.max_gap + 2) :]]
            for sym in tail:
                nxt_reg: List[Tuple[int, int]] = []
                for op, age in reg:
                    na = age + 1
                    if na > self.max_gap:
                        continue
                    if any(sp.closer == sym for sp in self.frames_by_opener_.get(op, ())):
                        continue
                    nxt_reg.append((op, na))
                reg = nxt_reg
                if sym > 0 and sym in self.frames_by_opener_:
                    reg = [(op, age) for op, age in reg if op != sym]
                    reg.append((sym, 0))
                    if len(reg) > self.max_active_frames:
                        reg = reg[-self.max_active_frames :]
        else:
            reg = self._active

        if not reg:
            return {}

        scores: Dict[int, float] = {}
        for op, age in reg:
            # Next symbol will land at gap = age + 1
            next_gap = float(age + 1)
            if next_gap < float(self.min_gap) or next_gap > float(self.max_gap):
                continue
            for sp in self.frames_by_opener_.get(op, ()):
                sigma = max(sp.std_gap, 1.5)
                timing = float(np.exp(-0.5 * ((next_gap - sp.mean_gap) / sigma) ** 2))
                val = sp.strength * (0.25 + 0.75 * timing)
                if val > scores.get(sp.closer, 0.0):
                    scores[sp.closer] = val
        return scores

    def top_frames(self, itos: Sequence[str], top_n: int = 20) -> List[Tuple[str, str, float, float, float]]:
        """Inspect strongest discovered directional frames: (opener, closer, pmi, asym, mean_gap)."""
        all_specs = [sp for specs in self.frames_by_opener_.values() for sp in specs]
        all_specs.sort(key=lambda sp: sp.strength, reverse=True)
        out = []
        for sp in all_specs[:top_n]:
            o_s = itos[sp.opener] if 0 <= sp.opener < len(itos) else "?"
            c_s = itos[sp.closer] if 0 <= sp.closer < len(itos) else "?"
            out.append((o_s, c_s, round(sp.pmi, 2), round(sp.asymmetry, 2), round(sp.mean_gap, 1)))
        return out
