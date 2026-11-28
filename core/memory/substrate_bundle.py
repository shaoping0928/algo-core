"""Parallel Multi-Core Substrate Builder & Reusable Checkpoint Persistence.

Unifies all structural memory reserves into a single reusable ``StructuralSubstrateBundle``:
  - **Multi-core CPU acceleration**: exploits Python 3.14t free-threading (no-GIL)
    via ``ThreadPoolExecutor`` to build independent reserves concurrently AND
    shard heavy ``SparseFeatureActionIndex`` streams across worker threads.
  - **Hard memory budget governance**: integrates ``MemoryBudgetGovernor``
    (default 16 GB ceiling), progressively strengthening forgetting & eviction
    as RSS approaches the limit.
  - **Reusable checkpoint persistence**: atomic gzip-compressed save/load so
    trained multi-million-step substrates can be reloaded in seconds for
    multi-dimensional testing without re-training.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import gzip
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from class_policy import ClassLevelSequencePolicy
from compression_tree import HierarchicalStageTree, LearnedDrillDownPolicy, SelfDiscoveredRoleExtractor
from context_decision import ContextDrivenDecisionController, LearnedClassContextGate
from context_integrator import ContextualIntegrator
from hierarchical_arbitrator import HierarchicalOnlineArbitrator
from memory_governor import MemoryBudgetGovernor, current_rss_bytes
from recency_kernel import LearnedRecencyLagKernel
from sequence_stage import HierarchicalSequenceStageModel
from structural_memory import HierarchicalChunkMemory, StructuralFrameMemory
from temp_context import TemporaryContextMemory
from virtual_node_dag import BidirectionalVirtualNodeDAG


@dataclass
class StructuralSubstrateBundle:
    """Serializable container holding all trained structural memory reserves & online learners."""

    vocab_size: int
    class_policy: ClassLevelSequencePolicy
    virtual_dag: BidirectionalVirtualNodeDAG
    temp_memory: TemporaryContextMemory
    recency_kernel: LearnedRecencyLagKernel
    chunk_memory: HierarchicalChunkMemory
    frame_memory: StructuralFrameMemory
    learned_gate: LearnedClassContextGate
    arbitrator: HierarchicalOnlineArbitrator
    stage_model: Optional[HierarchicalSequenceStageModel] = None
    role_extractor: Optional[SelfDiscoveredRoleExtractor] = None
    drill_policy: Optional[LearnedDrillDownPolicy] = None
    context_integrator: Optional[ContextualIntegrator] = None
    vocab: Optional[Dict[str, int]] = None
    itos: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit_parallel(
        cls,
        tokens: Sequence[int],
        *,
        vocab_size: int,
        vocab: Optional[Dict[str, int]] = None,
        itos: Optional[List[str]] = None,
        n_workers: Optional[int] = None,
        max_memory_bytes: int = 16 * (1024 ** 3),
        n_classes: int = 280,
        n_singletons: int = 28,
        feature_dim: int = 32,
        max_chunks_per_level: int = 1500,
        n_chunk_classes: int = 160,
        build_stage_model: bool = True,
        stage_slice: int = 400_000,
        post_fit_noise_prune: bool = True,
        noise_prune_pressure: float = 0.22,
        seed: int = 42,
    ) -> "StructuralSubstrateBundle":
        """Build all multi-layer memory reserves concurrently across CPU cores under memory governance."""
        t_start = time.perf_counter()
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, vocab_size - 1)
        n_toks = len(arr)
        cpu_n = max(os.cpu_count() or 1, 1)
        total_workers = min(max(int(n_workers), 1), cpu_n) if n_workers is not None else min(cpu_n, 16)
        # Allocate inner shard threads to heavy index builders alongside outer concurrent tasks
        inner_workers = max(min(total_workers // 3, 6), 1)
        outer_workers = min(6, total_workers)
        gil_enabled = bool(getattr(sys, "_is_gil_enabled", lambda: True)())
        governor = MemoryBudgetGovernor(max_bytes=max_memory_bytes)

        timings: Dict[str, float] = {}
        min_trig = max(min(n_toks // 60_000, 60), 5)
        min_pair = max(min(n_toks // 250_000, 15), 3)
        min_cooc = max(min(n_toks // 400_000, 8), 2)
        tcm_stride = 4 if n_toks >= 2_000_000 else (2 if n_toks >= 500_000 else 1)
        min_chunk_pair = max(min(n_toks // 150_000, 25), 5)
        min_frame_pair = max(min(n_toks // 150_000, 25), 5)

        fit_min_ho = 2 if (post_fit_noise_prune and n_toks >= 500_000) else 1

        def _fit_class_policy():
            t = time.perf_counter()
            obj = ClassLevelSequencePolicy(
                vocab_size=vocab_size, n_classes=n_classes, n_singletons=n_singletons,
                feature_dim=feature_dim, window=3, seed=seed,
            ).fit(arr, n_workers=inner_workers, min_high_order_count=fit_min_ho)
            timings["class_policy"] = round(time.perf_counter() - t, 2)
            return obj

        def _fit_dag():
            t = time.perf_counter()
            obj = BidirectionalVirtualNodeDAG(
                vocab_size=vocab_size, min_trigger_count=min_trig,
                min_pair_count=min_pair, min_lift=1.5,
            ).fit_sequence(arr)
            timings["virtual_dag"] = round(time.perf_counter() - t, 2)
            return obj

        def _fit_tcm():
            t = time.perf_counter()
            obj = TemporaryContextMemory(
                vocab_size=vocab_size, assoc_window=12, top_assoc_per_sym=24,
                min_cooc=min_cooc, max_compounds=1500, min_compound_pmi=2.5,
                corroboration_singleton_scale=0.22,
            ).fit_associations(arr, stride=tcm_stride, max_tokens=n_toks)
            timings["temp_memory"] = round(time.perf_counter() - t, 2)
            return obj

        def _fit_chunks():
            t = time.perf_counter()
            obj = HierarchicalChunkMemory(
                vocab_size=vocab_size, max_levels=2, max_chunks_per_level=max_chunks_per_level,
                min_pair_count=min_chunk_pair, min_pmi=2.2, max_chunk_len=4, chunk_ctx_len=2,
                enable_chunk_classes=True, n_chunk_classes=n_chunk_classes,
                n_singleton_chunks=min(32, max(vocab_size // 10, 4)),
                chunk_class_ctx_len=3, min_schema_count=max(min_chunk_pair // 2, 4), seed=seed,
            ).fit(arr, n_workers=inner_workers, min_high_order_count=fit_min_ho)
            timings["chunk_memory"] = round(time.perf_counter() - t, 2)
            return obj

        def _fit_frames():
            t = time.perf_counter()
            obj = StructuralFrameMemory(
                vocab_size=vocab_size, min_gap=2, max_gap=18,
                min_pair_count=min_frame_pair, min_pmi=1.8, min_asymmetry=0.65,
                max_closers_per_opener=5,
            ).fit(arr, max_samples=250_000)
            timings["frame_memory"] = round(time.perf_counter() - t, 2)
            return obj

        def _fit_stage():
            if not build_stage_model or n_toks < 200:
                return None, None, None
            t = time.perf_counter()
            sl = arr[: min(n_toks, max(int(stage_slice), 500))]
            sm = HierarchicalSequenceStageModel(
                vocab_size=vocab_size, feature_dim=feature_dim, window=3,
                n_regimes=min(20, max(vocab_size // 20, 4)),
                min_dwell=5, hysteresis=1, ctx_len=3, chunk_len=min(2000, max(len(sl) // 4, 60)), seed=seed,
            ).fit(sl)
            sample_n = min(len(sl), 30_000)
            chain, _, _ = sm.compress_to_stage_chain(sl[:sample_n])
            surp = sm._compute_local_surprises(sl[:sample_n])
            rext = SelfDiscoveredRoleExtractor(n_roles=4).fit_from_spans(
                [(sl[st.start : st.end + 1], surp[st.start : st.end + 1], st.mean_state) for st in chain],
                sm.feature_map.embeddings_, sm.feature_map.salience_,
            )
            dpol = LearnedDrillDownPolicy(step_cost=0.01)
            timings["stage_model"] = round(time.perf_counter() - t, 2)
            return sm, rext, dpol

        # Phase 1: Parallel independent reserve fitting across CPU cores
        if outer_workers > 1:
            with ThreadPoolExecutor(max_workers=outer_workers) as ex:
                f_pol = ex.submit(_fit_class_policy)
                f_dag = ex.submit(_fit_dag)
                f_tcm = ex.submit(_fit_tcm)
                f_cm = ex.submit(_fit_chunks)
                f_fm = ex.submit(_fit_frames)
                f_st = ex.submit(_fit_stage)
                pol = f_pol.result()
                dag = f_dag.result()
                tcm = f_tcm.result()
                cm = f_cm.result()
                fm = f_fm.result()
                stage_model, role_ext, drill_pol = f_st.result()
        else:
            pol = _fit_class_policy()
            dag = _fit_dag()
            tcm = _fit_tcm()
            cm = _fit_chunks()
            fm = _fit_frames()
            stage_model, role_ext, drill_pol = _fit_stage()

        # Phase 2: Fit unbiased recency-lag saturation kernel & chunk-level ContextualIntegrator concurrently
        def _fit_recency():
            t = time.perf_counter()
            obj = LearnedRecencyLagKernel(
                vocab_size=vocab_size, n_classes=pol.n_classes,
            ).fit(arr, sym_to_class=pol.sym_to_class_, local_index=pol.local_index, max_samples=120_000)
            timings["recency_kernel"] = round(time.perf_counter() - t, 2)
            return obj

        def _fit_ctx_integrator():
            t = time.perf_counter()
            parsed = cm.parse_stream(arr)
            ci = ContextualIntegrator(
                n_units=cm.n_total_units,
                vocab_size=vocab_size,
                forward_window=48,
                top_targets_per_cue=48,
                min_cooc=max(min_cooc, 3),
                min_pmi=0.85,
            ).fit_unit_stream(
                parsed,
                chunk_to_atoms=cm.chunk_to_atoms_,
                stride=max(tcm_stride, 2),
                max_centers=600_000,
            )
            timings["context_integrator"] = round(time.perf_counter() - t, 2)
            return ci

        if outer_workers > 1:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_rec = ex.submit(_fit_recency)
                f_ci = ex.submit(_fit_ctx_integrator)
                rec_kern = f_rec.result()
                ctx_int = f_ci.result()
        else:
            rec_kern = _fit_recency()
            ctx_int = _fit_ctx_integrator()

        tcm.attach_context_integrator(ctx_int, cm)

        # Phase 3: Optional built-in tail-noise pruning (cuts ~80% count=1 high-order noise
        #          while preserving or slightly improving held-out accuracy) + 16GB governance
        noise_evicted = 0
        if post_fit_noise_prune and n_toks >= 50_000 and float(noise_prune_pressure) > 0.0:
            p_np = float(np.clip(noise_prune_pressure, 0.0, 0.5))
            noise_evicted += pol.prune_under_pressure(p_np)
            noise_evicted += cm.prune_under_pressure(p_np)
            noise_evicted += dag.prune_under_pressure(p_np)
            import gc as _gc
            _gc.collect()
        mem_report = governor.govern([pol, cm, dag, tcm, ctx_int])
        total_wall = round(time.perf_counter() - t_start, 2)

        bundle = cls(
            vocab_size=int(vocab_size),
            class_policy=pol,
            virtual_dag=dag,
            temp_memory=tcm,
            recency_kernel=rec_kern,
            chunk_memory=cm,
            frame_memory=fm,
            learned_gate=LearnedClassContextGate(pol.n_classes, lr=1.2, init_logit=0.0),
            arbitrator=HierarchicalOnlineArbitrator(n_regimes=3, lr=0.45, episode_window=350),
            stage_model=stage_model,
            role_extractor=role_ext,
            drill_policy=drill_pol,
            context_integrator=ctx_int,
            vocab=vocab,
            itos=list(itos) if itos is not None else None,
            metadata=dict(
                train_tokens=int(n_toks),
                vocab_size=int(vocab_size),
                cpu_count=int(cpu_n),
                workers_used=int(total_workers),
                gil_enabled=gil_enabled,
                wall_seconds=total_wall,
                component_timings=timings,
                post_fit_noise_evicted=int(noise_evicted),
                memory_governor=mem_report,
                max_memory_gb=round(max_memory_bytes / (1024.0 ** 3), 2),
                rss_mb_after_fit=round(current_rss_bytes() / (1024.0 * 1024.0), 1),
            ),
        )
        return bundle

    def build_prompt_stage_tree(self, prompt_tokens: Sequence[int]) -> Optional[HierarchicalStageTree]:
        """Compress an episode prompt into a live ``HierarchicalStageTree`` if stage_model is present."""
        if self.stage_model is None or self.role_extractor is None or len(prompt_tokens) < 8:
            return None
        p_arr = np.clip(np.asarray(prompt_tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        chain, _, _ = self.stage_model.compress_to_stage_chain(p_arr)
        if not chain:
            return None
        surp = self.stage_model._compute_local_surprises(p_arr)
        return HierarchicalStageTree.build_from_stage_chain(
            chain, p_arr, surp,
            self.stage_model.feature_map.embeddings_,
            self.stage_model.feature_map.salience_,
            self.role_extractor,
            branch_factor=3,
        )

    def make_controller(
        self,
        *,
        use_arbitrator: bool = True,
        stage_tree: Optional[HierarchicalStageTree] = None,
        lookahead_depth: int = 1,
        lookahead_beam: int = 6,
    ) -> ContextDrivenDecisionController:
        """Instantiate a unified ``ContextDrivenDecisionController`` wired to all reserves."""
        return ContextDrivenDecisionController(
            self.class_policy,
            self.virtual_dag,
            self.temp_memory,
            stage_tree=stage_tree,
            drill_policy=self.drill_policy,
            active_inject_weight=0.35,
            bridge_weight=0.20,
            content_class_gate=True,
            learned_class_gate=self.learned_gate,
            lookahead_depth=lookahead_depth,
            lookahead_beam=lookahead_beam,
            recency_kernel=self.recency_kernel,
            recency_strength=0.85,
            chunk_memory=self.chunk_memory,
            chunk_weight=0.38,
            frame_memory=self.frame_memory,
            frame_weight=0.15,
            arbitrator=self.arbitrator if use_arbitrator else None,
            context_integrator=self.context_integrator,
        )

    def govern_memory(self, max_bytes: int = 16 * (1024 ** 3), *, rss_bytes: Optional[int] = None) -> dict:
        """Run memory-pressure governance across all memory reserves in this bundle."""
        gov = MemoryBudgetGovernor(max_bytes=max_bytes)
        targets = [self.class_policy, self.chunk_memory, self.virtual_dag, self.temp_memory]
        if self.context_integrator is not None:
            targets.append(self.context_integrator)
        rep = gov.govern(targets, rss_bytes=rss_bytes)
        self.metadata["last_memory_governor"] = rep
        return rep

    def ingest_stream_chunk(
        self,
        tokens: Sequence[int],
        *,
        n_workers: int = 8,
        max_memory_bytes: int = 16 * (1024 ** 3),
        min_high_order_count: int = 2,
    ) -> Dict[str, Any]:
        """Incrementally ingest an additional macro-chunk in parallel under strict memory governance."""
        t0 = time.perf_counter()
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        nw_each = max(int(n_workers) // 2, 1)

        def _ingest_chunks_and_ctx():
            parsed = self.chunk_memory.parse_stream(arr)
            self.chunk_memory.ingest_chunk(
                arr, n_workers=nw_each, min_high_order_count=min_high_order_count,
            )
            if self.context_integrator is not None:
                self.context_integrator.ingest_unit_chunk(parsed, stride=4, max_centers=300_000)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(
                self.class_policy.ingest_chunk, arr,
                n_workers=nw_each, min_high_order_count=min_high_order_count,
            )
            f2 = ex.submit(_ingest_chunks_and_ctx)
            f1.result()
            f2.result()
        self.metadata["train_tokens"] = int(self.metadata.get("train_tokens", 0)) + len(arr)
        mem_rep = self.govern_memory(max_bytes=max_memory_bytes)
        return dict(
            chunk_tokens=len(arr),
            total_tokens=int(self.metadata["train_tokens"]),
            seconds=round(time.perf_counter() - t0, 2),
            memory=mem_rep,
        )

    def save(self, path: str, *, compresslevel: int = 1) -> Dict[str, Any]:
        """Atomically serialize this trained bundle to a reusable compressed checkpoint file."""
        t0 = time.perf_counter()
        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = abs_path + ".tmp"
        with gzip.open(tmp_path, "wb", compresslevel=int(compresslevel)) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, abs_path)
        size_mb = os.path.getsize(abs_path) / (1024.0 * 1024.0)
        return dict(
            path=abs_path,
            size_mb=round(size_mb, 2),
            save_seconds=round(time.perf_counter() - t0, 2),
        )

    @classmethod
    def load(cls, path: str) -> "StructuralSubstrateBundle":
        """Load a previously saved ``StructuralSubstrateBundle`` checkpoint."""
        abs_path = os.path.abspath(path)
        with gzip.open(abs_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected StructuralSubstrateBundle in {abs_path}, got {type(obj).__name__}")
        return obj
