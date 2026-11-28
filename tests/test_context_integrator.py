import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for dp, _, _ in os.walk(os.path.join(ROOT, "core")):
    if "__pycache__" not in dp:
        sys.path.insert(0, dp)

from context_integrator import ContextualIntegrator
from substrate_bundle import StructuralSubstrateBundle


class ContextualIntegratorTests(unittest.TestCase):
    def test_multi_cue_intersection_disambiguation_and_no_amnesia(self):
        # Synthetic two-domain stream where cue 10 is ambiguous between domain 1 (11 -> 20,21)
        # and domain 2 (12 -> 30,31), separated by long stretches of neutral filler (1..5).
        rng = np.random.RandomState(42)
        seq = []
        for _ in range(120):
            # Domain 1 episode: cues 10, 11 followed 15-35 steps later by targets 20, 21
            seq.extend([10, 1, 2, 11] + rng.randint(1, 6, size=18).tolist() + [20, 3, 21, 4, 22])
            # Domain 2 episode: cues 10, 12 followed 15-35 steps later by targets 30, 31
            seq.extend([10, 2, 3, 12] + rng.randint(1, 6, size=18).tolist() + [30, 4, 31, 1, 32])
        arr = np.asarray(seq, dtype=np.int32)

        ci = ContextualIntegrator(
            n_units=40, vocab_size=40, forward_window=36, min_cooc=5, min_pmi=0.4,
        ).fit_unit_stream(arr, stride=1, max_centers=50_000)

        # 1. Domain 1 context followed by 40 neutral filler steps -> early cues MUST survive!
        ci.reset()
        ci.observe_units([10, 11] + [1, 2, 3, 4] * 10, source="external")
        llr1 = ci.contextual_atom_llr()
        self.assertGreater(llr1.get(20, 0.0), llr1.get(30, 0.0))
        self.assertGreater(llr1.get(20, 0.0), 0.5)

        # 2. Domain 2 context disambiguates cue 10 toward target 30 instead of 20!
        ci.reset()
        ci.observe_units([10, 12] + [1, 2, 3, 4] * 10, source="external")
        llr2 = ci.contextual_atom_llr()
        self.assertGreater(llr2.get(30, 0.0), llr2.get(20, 0.0))

        # 3. Realization adaptation: once target 30 is self-emitted repeatedly, its drive drops
        #    while unexpressed domain-2 target 31 stays elevated!
        s30_before = llr2[30]
        s31_before = llr2.get(31, 0.0)
        for _ in range(3):
            ci.observe_units([30], source="self")
        llr2_after = ci.contextual_atom_llr()
        self.assertLess(llr2_after.get(30, 0.0), s30_before)
        self.assertGreater(llr2_after.get(31, 0.0), llr2_after.get(30, 0.0) * 0.8)
        self.assertGreater(s31_before, 0.0)

    def test_bundle_integrates_context_across_long_boundary(self):
        # Construct a 2-topic stream with distinct episodes separated by neutral gaps,
        # where both topics share the exact same local boundary [2, 3, 4, 5] before
        # diverging to topic-A continuation (20, 21) vs topic-B continuation (30, 31).
        rng = np.random.RandomState(7)
        seq = []
        for ep in range(160):
            # Inter-episode gap of neutral tokens (1..4) so Topic A & Topic B do not bleed
            seq.extend(rng.randint(1, 5, size=55).tolist())
            if ep % 2 == 0:
                seq.extend([10, 11, 1, 12, 13, 2, 3, 4, 1, 2, 3, 4, 5, 20, 21, 6, 7])
            else:
                seq.extend([14, 15, 1, 16, 17, 2, 3, 4, 1, 2, 3, 4, 5, 30, 31, 6, 7])
        arr = np.asarray(seq, dtype=np.int32)

        bundle = StructuralSubstrateBundle.fit_parallel(
            arr, vocab_size=40, n_workers=2, n_classes=12, n_singletons=4,
            feature_dim=8, max_chunks_per_level=32, n_chunk_classes=8,
            build_stage_model=False, post_fit_noise_prune=False, seed=42,
        )
        self.assertIsNotNone(bundle.context_integrator)
        ctrl = bundle.make_controller(use_arbitrator=False)

        # Prompt A ends with identical local context [2, 3, 4, 5] as Prompt B!
        prompt_a = [10, 11, 1, 12, 13, 2, 3, 4, 1, 2, 3, 4, 5]
        prompt_b = [14, 15, 1, 16, 17, 2, 3, 4, 1, 2, 3, 4, 5]

        bundle.temp_memory.reset()
        ha = np.zeros(8, dtype=float)
        for s in prompt_a:
            bundle.temp_memory.observe(s, source="external")
            ha = bundle.class_policy.update_slow_state(ha, s)
        acts_a, _, _ = ctrl.decide_distribution(prompt_a[-4:], ha, k=4)

        bundle.temp_memory.reset()
        hb = np.zeros(8, dtype=float)
        for s in prompt_b:
            bundle.temp_memory.observe(s, source="external")
            hb = bundle.class_policy.update_slow_state(hb, s)
        acts_b, _, _ = ctrl.decide_distribution(prompt_b[-4:], hb, k=4)

        # Identical last 4 tokens [2, 3, 4, 5], yet long-range integrated context selects 20 vs 30!
        self.assertEqual(acts_a[0], 20)
        self.assertEqual(acts_b[0], 30)


if __name__ == "__main__":
    unittest.main()
