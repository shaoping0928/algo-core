import unittest

import numpy as np

from milestone_graph import (
    boundary_axis,
    boundary_keys,
    boundary_scale,
    boundary_scores,
    frame_keys,
    frame_project,
    frame_project_axes,
    milestone_coverage,
    milestone_memory,
    milestone_statistics,
    next_milestone,
    pooled_boundary_model,
    quantile_cuts,
    support_bins,
    segment_boundaries,
    slow_baseline_jumps,
    strength_boundaries,
    transition_signatures,
)


def sawtooth(period, cycles, amplitude=1.0, noise=0.01, seed=0):
    """A periodic stream that ramps down and resets once per period."""
    rng = np.random.RandomState(seed)
    n = int(period) * int(cycles)
    ramp = -amplitude * (np.arange(n) % int(period)) / float(period)
    return (ramp + rng.normal(0.0, noise, n)).reshape(-1, 1)


def ramped_sawtooth(period, cycles):
    """A reset signal riding on an unbounded timer coordinate.

    The timer ramps one unit per frame, so its leaky-baseline step is a
    constant - which is exactly what used to pin the raw boundary statistic.
    """
    n = int(period) * int(cycles)
    signal = (-(np.arange(n) % int(period)) / float(period)).reshape(-1, 1)
    timer = np.arange(n, dtype=float).reshape(-1, 1)
    return np.concatenate([signal, timer], axis=1)


class RampingCoordinateTests(unittest.TestCase):
    def test_raw_statistic_cannot_see_the_reset(self):
        # The motivating failure: a raw max over coordinates is dominated by
        # the ramping timer, so a reset frame is no larger than an ordinary
        # frame and no threshold on this statistic could ever isolate it.
        x = ramped_sawtooth(period=100, cycles=4)
        raw = slow_baseline_jumps(x)
        ordinary = float(np.median(raw[2:]))
        for reset in (100, 200, 300):
            self.assertAlmostEqual(float(raw[reset]), ordinary, places=6)

    def test_per_coordinate_scale_finds_the_resets(self):
        x = ramped_sawtooth(period=100, cycles=4)
        self.assertEqual(list(segment_boundaries(x)), [100, 200, 300])

    def test_scale_is_zero_safe_on_a_flat_coordinate(self):
        x = np.zeros((50, 2))
        scale = boundary_scale(x)
        np.testing.assert_allclose(scale, 0.0)
        np.testing.assert_allclose(boundary_scores(x), 0.0)

    def test_tracker_reproduces_the_offline_boundaries(self):
        from milestone_graph import MilestoneTracker

        x = ramped_sawtooth(period=50, cycles=4)
        model = pooled_boundary_model([x])
        frame = boundary_axis([x], threshold=model["threshold"],
                              scale=model["scale"])
        tracker = MilestoneTracker(
            model["threshold"], frame, scale=model["scale"]
        )
        opened = [i for i in range(len(x)) if tracker.observe(x[i])]
        # Frame 0 opens the first milestone; the resets open the rest.
        self.assertEqual(opened, [0, 50, 100, 150])

    def test_pooled_model_finds_resets_in_each_episode(self):
        # Episode seams are not phase resets; pooling must not let them raise
        # the threshold past a real reset.
        streams = [sawtooth(period=50, cycles=2, noise=0.0) for _ in range(3)]
        model = pooled_boundary_model(streams)
        for stream in streams:
            found = segment_boundaries(
                stream, threshold=model["threshold"], scale=model["scale"]
            )
            self.assertEqual(list(found), [50])


class ConceptIdentityTests(unittest.TestCase):
    def test_frame_coordinates_are_standardised(self):
        # A coordinate with large units must not supply the axis merely
        # because of its scale, and the axis must not be the time ramp.
        n = 40
        time = np.linspace(0.0, 1.0, n)
        signal = np.sin(np.arange(n) * 0.7)
        states = np.column_stack([time, 100.0 * signal])
        boundaries = np.asarray([10, 20, 30])
        frame = boundary_axis([states], boundaries_list=[boundaries], bins=2)
        # sigma absorbs the 100x difference in units.
        np.testing.assert_allclose(
            frame["sigma"], states[boundaries].std(axis=0), rtol=1e-6
        )
        self.assertGreater(
            float(frame["sigma"][1] / frame["sigma"][0]), 50.0
        )
        # The axis carries the non-time signal, not the time ramp.
        self.assertLess(abs(float(frame["axis"][0])), 0.2)
        self.assertGreater(abs(float(frame["axis"][1])), 0.8)

    def test_frame_keys_are_a_coarse_concept(self):
        # Many boundary states, only a couple of identities: a concept, not a
        # per-milestone fingerprint.
        rng = np.random.RandomState(0)
        states = rng.normal(size=(61, 4))
        boundaries = np.arange(10, 600, 10)
        frame = boundary_axis([states], boundaries_list=[boundaries], bins=2)
        keys = frame_keys(states, boundaries, frame)
        self.assertLessEqual(len(set(keys)), 2)

    def test_identity_axis_is_orthogonal_to_time(self):
        # A timer that ramps with the episode must not supply the identity:
        # the projection is orthogonalised against progress.  (Residual
        # correlation of the *projection* with progress can come from the
        # signal itself, so the guarantee tested is the axis, not the sample
        # correlation.)
        n = 240
        timer = np.linspace(0.0, 6.0, n)
        rng = np.random.RandomState(4)
        signal = rng.normal(size=n)
        states = np.column_stack([timer, signal])
        boundaries = np.arange(20, n, 12)
        frame = boundary_axis([states], boundaries_list=[boundaries], bins=2)
        self.assertLess(
            abs(float(frame["axis"] @ frame["time_dir"])), 1e-6
        )
        self.assertLess(abs(float(frame["axis"][0])), 0.5)

    def test_support_gate_raises_the_bin_count(self):
        projection = np.concatenate([
            np.linspace(0.0, 0.2, 20),
            np.linspace(0.4, 0.6, 20),
            np.linspace(0.8, 1.0, 20),
        ])
        cuts, count = support_bins(projection, min_support=10, max_bins=6)
        keys = np.searchsorted(cuts, projection)
        self.assertGreaterEqual(count, 3)
        self.assertTrue(np.all(np.bincount(keys, minlength=count) >= 10))

    def test_frame_projection_axes_returns_all_directions(self):
        states = np.random.RandomState(2).normal(size=(20, 3))
        frame = boundary_axis(
            [states], boundaries_list=[np.arange(5, 20, 5)], bins=2
        )
        projections = frame_project_axes(frame, states)
        self.assertEqual(projections.shape, (20, frame["axes"].shape[0]))
        np.testing.assert_allclose(
            projections[:, 0], frame_project(frame, states)
        )

    def test_frame_projection_is_scale_invariant(self):
        rng = np.random.RandomState(1)
        states = rng.normal(size=(51, 3))
        boundaries = np.arange(5, 500, 5)
        base = boundary_axis([states], boundaries_list=[boundaries], bins=2)
        # Rescaling a coordinate must not move any milestone's identity:
        # sigma absorbs the change of units.
        scaled = states.copy()
        scaled[:, 1] *= 1000.0
        other = boundary_axis([scaled], boundaries_list=[boundaries], bins=2)
        np.testing.assert_allclose(
            frame_keys(states, boundaries, base),
            frame_keys(scaled, boundaries, other),
        )


class TaskAgnosticGuardTests(unittest.TestCase):
    def test_milestone_layer_names_no_task_quantity(self):
        # The layer must read the stream, not this task: no coordinate name,
        # world constant, progress counter or environment module may appear.
        import inspect

        import milestone_graph
        import substrate_nodes

        tokens = ("BIRD_X", "GRAVITY", "FLAP", "SPEED", "SPACING", "GAP",
                  "pipe", "flappy", "latent", "passed")
        source = inspect.getsource(milestone_graph)
        for token in tokens:
            self.assertNotIn(token, source)
        # substrate_nodes legitimately imports the worlds; only the milestone
        # functions have to stay task-free.
        for name in ("_milestone_model_from_transitions",
                     "_milestone_goals_from_transitions"):
            text = inspect.getsource(getattr(substrate_nodes, name))
            for token in tokens:
                self.assertNotIn(token, text)


class MilestoneLookaheadTests(unittest.TestCase):
    def test_walks_two_stages_and_prices_bad_mass(self):
        from milestone_graph import milestone_lookahead

        table = [
            dict(count=9, key_from=[0], key_to=[1], mean_duration=10.0,
                 mean_feedback=10.0),
            dict(count=1, key_from=[0], key_to=[2], mean_duration=10.0,
                 mean_feedback=-10.0),
            dict(count=10, key_from=[1], key_to=[1], mean_duration=10.0,
                 mean_feedback=10.0),
            dict(count=10, key_from=[2], key_to=[2], mean_duration=10.0,
                 mean_feedback=-10.0),
        ]
        walk = milestone_lookahead(
            table, (0,), key_feedback={(1,): 1.0, (2,): -1.0}, steps=2
        )
        self.assertAlmostEqual(walk["duration"], 20.0)
        self.assertAlmostEqual(walk["per_frame_feedback"], 0.8)
        self.assertAlmostEqual(walk["risk"], 0.1)
        self.assertEqual(
            sorted(stage for stage, _mass in walk["keys"]), [(1,), (2,)]
        )

    def test_empty_table_predicts_nothing(self):
        from milestone_graph import milestone_lookahead

        walk = milestone_lookahead([], (0,), steps=2)
        self.assertEqual(walk["keys"], [])
        self.assertEqual(walk["duration"], 0.0)
        self.assertEqual(walk["per_frame_feedback"], 0.0)
        self.assertEqual(walk["cost"], 0.0)

    def test_terminal_stage_keeps_its_mass(self):
        from milestone_graph import milestone_lookahead

        table = [
            dict(count=1, key_from=[0], key_to=[1], mean_duration=5.0,
                 mean_feedback=5.0),
        ]
        walk = milestone_lookahead(
            table, (0,), steps=2, key_cost={(1,): 1.0}
        )
        self.assertAlmostEqual(walk["cost"], 1.0)
        self.assertEqual(
            sorted(stage for stage, _mass in walk["keys"]), [(1,)]
        )

    def test_lookahead_cost_is_graded_over_the_frontier(self):
        from milestone_graph import milestone_lookahead

        table = [
            dict(count=3, key_from=[0], key_to=[1], mean_duration=10.0,
                 mean_feedback=10.0),
            dict(count=1, key_from=[0], key_to=[2], mean_duration=10.0,
                 mean_feedback=-10.0),
        ]
        walk = milestone_lookahead(
            table, (0,), steps=1, key_cost={(1,): 0.0, (2,): 1.0}
        )
        self.assertAlmostEqual(walk["cost"], 0.25)


class SlowBaselineJumpTests(unittest.TestCase):
    def test_jump_series_is_zero_on_a_constant_stream(self):
        x = np.full((50, 3), 0.4)
        jumps = slow_baseline_jumps(x)
        np.testing.assert_allclose(jumps, 0.0)

    def test_jump_series_marks_the_reset(self):
        # The period must be long enough that the ordinary per-frame drift
        # stays far below the reset: a reset of amplitude 1 over a period p
        # must exceed k * (drift / beta) = k * (1/p) / beta.
        x = sawtooth(period=100, cycles=2, noise=0.0)
        jumps = slow_baseline_jumps(x)
        self.assertAlmostEqual(float(jumps[0]), 0.0)
        # The reset lands at index 100 and is far above every other step.
        self.assertGreater(float(jumps[100]), 10.0 * float(jumps[1:100].max()))

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            slow_baseline_jumps(np.zeros(5))
        with self.assertRaises(ValueError):
            slow_baseline_jumps(np.zeros((5, 2)), beta=0.0)
        with self.assertRaises(ValueError):
            slow_baseline_jumps(np.zeros((5, 2)), beta=1.5)


class SegmentBoundariesTests(unittest.TestCase):
    def test_constant_stream_has_no_boundaries(self):
        self.assertEqual(len(segment_boundaries(np.full((200, 2), 0.3))), 0)

    def test_one_boundary_per_cycle(self):
        cycles = 20
        x = sawtooth(period=50, cycles=cycles)
        boundaries = segment_boundaries(x)
        self.assertLessEqual(abs(len(boundaries) - (cycles - 1)), 1)
        gaps = np.diff(boundaries)
        if len(gaps):
            self.assertLess(abs(float(np.median(gaps)) - 50), 2)

    def test_boundary_count_is_invariant_to_the_time_scale(self):
        # The same number of cycles at twice the period: the number of
        # milestones must not move, only their duration.
        cycles = 20
        fast = segment_boundaries(sawtooth(period=50, cycles=cycles))
        slow = segment_boundaries(sawtooth(period=100, cycles=cycles))
        self.assertLessEqual(abs(len(fast) - len(slow)), 1)
        fast_gap = float(np.median(np.diff(fast)))
        slow_gap = float(np.median(np.diff(slow)))
        self.assertAlmostEqual(slow_gap / fast_gap, 2.0, delta=0.2)

    def test_action_like_transients_do_not_create_milestones(self):
        # A transient that perturbs one coordinate for a single frame is not
        # a phase reset; the tail statistic must not turn it into a boundary.
        cycles = 20
        clean = sawtooth(period=50, cycles=cycles)
        noisy = clean.copy()
        noisy[::7] += 0.20
        base = len(segment_boundaries(clean))
        with_transients = len(segment_boundaries(noisy))
        self.assertLessEqual(with_transients, base + 1)

    def test_rejects_bad_parameters(self):
        x = sawtooth(period=50, cycles=3)
        with self.assertRaises(ValueError):
            segment_boundaries(x, k=0.0)
        with self.assertRaises(ValueError):
            segment_boundaries(x, quantile=1.0)
        with self.assertRaises(ValueError):
            segment_boundaries(x, nms=-1)


class StrengthBoundaryTests(unittest.TestCase):
    def test_clock_is_detected_from_the_stream_not_from_sparse_boundaries(self):
        # Two episodes, one boundary state each: the old boundary-based clock
        # test skipped every episode (fewer than three boundary states) and
        # silently let the clock back into the identity axis.
        streams = [
            ramped_sawtooth(period=10, cycles=3),
            ramped_sawtooth(period=12, cycles=3),
        ]
        frame = boundary_axis(streams, boundaries_list=[[10], [12]])
        self.assertEqual(np.flatnonzero(frame["time_dir"]).tolist(), [1])
        self.assertAlmostEqual(float(abs(frame["axis"][1])), 0.0, places=6)

    def test_strength_rise_with_a_new_future_makes_a_boundary(self):
        # The node's strength steps up at frame 40 and the action-conditioned
        # future changes there: the product has exactly one peak.
        strength = np.concatenate([np.full(40, 5.0), np.full(40, 50.0)])
        signatures = np.vstack([
            np.tile([0.0, 1.0], (40, 1)),
            np.tile([1.0, 0.0], (40, 1)),
        ])
        self.assertEqual(
            strength_boundaries(strength, signatures).tolist(), [40]
        )

    def test_strength_rise_alone_is_not_a_boundary(self):
        strength = np.concatenate([np.full(40, 5.0), np.full(40, 50.0)])
        signatures = np.tile([0.0, 1.0], (80, 1))
        self.assertEqual(len(strength_boundaries(strength, signatures)), 0)

    def test_future_change_alone_is_not_a_boundary(self):
        strength = np.full(80, 5.0)
        signatures = np.vstack([
            np.tile([0.0, 1.0], (40, 1)),
            np.tile([1.0, 0.0], (40, 1)),
        ])
        self.assertEqual(len(strength_boundaries(strength, signatures)), 0)

    def test_requires_aligned_inputs(self):
        with self.assertRaises(ValueError):
            strength_boundaries(np.zeros(3), np.zeros((4, 2)))


class TransitionSignatureTests(unittest.TestCase):
    def test_signature_is_action_conditioned_and_ordered(self):
        # A model that advances the state by the action: the a=1 rollout must
        # sit above the a=0 rollout, and both stay inside their own slot.
        from milestone_graph import transition_signatures

        def predict(states, history):
            return states + float(history[0, 0])

        states = np.array([[0.0], [10.0]])
        signatures = transition_signatures(
            states, predict, lag=1, horizon=2
        )
        self.assertEqual(signatures.shape, (2, 4))
        # a=0 slots are the untouched state; a=1 accumulates 1 then 2.
        np.testing.assert_allclose(signatures[0], [0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(signatures[1], [10.0, 10.0, 11.0, 12.0])


class SignatureIdentityTests(unittest.TestCase):
    def test_pooled_clusters_recover_two_movement_styles(self):
        from milestone_graph import signature_clusters, signature_keys

        rng = np.random.RandomState(0)
        style_a = rng.normal(0.0, 0.01, (30, 3)) + np.array([0.0, 1.0, 0.0])
        style_b = rng.normal(0.0, 0.01, (30, 3)) + np.array([1.0, 0.0, 0.0])
        clusters = signature_clusters([style_a, style_b], min_support=3)
        self.assertEqual(clusters["k"], 2)
        keys_a = set(signature_keys(style_a[:4], clusters["centers"]))
        keys_b = set(signature_keys(style_b[:4], clusters["centers"]))
        self.assertEqual(len(keys_a), 1)
        self.assertEqual(len(keys_b), 1)
        self.assertNotEqual(keys_a, keys_b)

    def test_support_gate_keeps_one_identity_when_nothing_can_split(self):
        from milestone_graph import signature_clusters

        points = np.array([[0.0], [0.0], [0.0], [1.0]])
        clusters = signature_clusters([points], min_support=3)
        self.assertEqual(clusters["k"], 1)

    def test_nearest_stage_key_maps_a_state_to_its_nearest_known_stage(self):
        from milestone_graph import nearest_stage_key

        frame = dict(center=np.zeros(2), sigma=np.ones(2))
        states = np.array([[0.0, 0.0], [10.0, 10.0]])
        keys = [(0,), (1,)]
        self.assertEqual(
            nearest_stage_key(states, keys, frame, [9.9, 10.1]), (1,)
        )
        self.assertEqual(
            nearest_stage_key(states, keys, frame, [0.1, -0.1]), (0,)
        )
        self.assertIsNone(nearest_stage_key(np.zeros((0, 2)), keys, frame,
                                            [0.0, 0.0]))

    def test_signature_stage_risk_uses_the_nearest_cluster(self):
        from milestone_graph import signature_stage_risk

        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        risks = {(0,): 0.2, (1,): 0.9}
        values = signature_stage_risk(
            np.array([[9.0, 9.0], [1.0, 0.0]]), centers, risks
        )
        np.testing.assert_allclose(values, [0.9, 0.2])

    def test_memory_accepts_an_explicit_identity(self):
        x = np.arange(40.0).reshape(-1, 1)
        identity = [(1,), (2,), (1,), (2,)]
        memory = milestone_memory(
            x, boundaries=[10, 20, 30], identity=identity
        )
        self.assertEqual(memory["keys"], identity)
        self.assertEqual(len(memory["durations"]), 4)


class MilestoneStatisticsTests(unittest.TestCase):
    def test_durations_cover_the_stream_and_feedback_is_summed(self):
        cycles = 10
        x = sawtooth(period=40, cycles=cycles)
        feedback = np.column_stack([
            np.ones(len(x)),                      # alive every frame
            np.where(np.arange(len(x)) % 40 == 39, 1.0, 0.0),
        ])
        stats = milestone_statistics(x, feedback=feedback)
        self.assertEqual(int(stats["durations"].sum()), len(x))
        self.assertEqual(int(stats["count"]), len(stats["durations"]))
        self.assertEqual(int(stats["feedback"].shape[0]), stats["count"])
        # Column 0 is the frame count of the stretch, by construction.
        np.testing.assert_allclose(stats["feedback"][:, 0],
                                   stats["durations"].astype(float))

    def test_explicit_boundaries_are_used_as_given(self):
        x = np.zeros((10, 1))
        stats = milestone_statistics(x, boundaries=np.asarray([3, 7]))
        np.testing.assert_array_equal(stats["durations"], [3, 4, 3])

    def test_feedback_length_must_match(self):
        with self.assertRaises(ValueError):
            milestone_statistics(
                np.zeros((10, 1)), feedback=np.zeros((9, 1)),
                boundaries=np.asarray([3]),
            )


class MilestoneMemoryTests(unittest.TestCase):
    def test_keys_are_invariant_to_a_monotone_rescale(self):
        # Rescaling a dimension (which is what a change of the world's time
        # scale does to a distance-like coordinate) must not move the
        # identity, because the cut points are read from the stream itself.
        rng = np.random.RandomState(0)
        x = rng.normal(size=(200, 2))
        boundaries = np.asarray([50, 100])
        base = boundary_keys(x, boundaries, bins=2)
        scaled = x.copy()
        scaled[:, 0] *= 100.0
        self.assertEqual(base, boundary_keys(scaled, boundaries, bins=2))

    def test_memory_counts_transitions_and_durations(self):
        x = np.zeros((40, 1))
        x[10, 0] = 10.0
        x[20, 0] = -10.0
        x[30, 0] = 10.0
        memory = milestone_memory(x, boundaries=np.asarray([10, 20, 30]))
        np.testing.assert_array_equal(memory["durations"], [10, 10, 10, 10])
        # The starting state of each milestone alternates across the median.
        self.assertEqual(len(set(memory["keys"])), 2)
        pairs = {
            (tuple(row["key_from"]), tuple(row["key_to"])): row["count"]
            for row in memory["transitions"]
        }
        self.assertEqual(sum(pairs.values()), 3)
        self.assertEqual(len(pairs), 2)
        self.assertAlmostEqual(
            float(memory["transitions"][0]["mean_duration"]), 10.0
        )

    def test_memory_feedback_is_the_stretch_total(self):
        x = np.zeros((40, 1))
        memory = milestone_memory(
            x, feedback=np.ones((40, 1)), boundaries=np.asarray([10, 30])
        )
        np.testing.assert_array_equal(
            memory["durations"], [10, 20, 10]
        )
        np.testing.assert_allclose(memory["feedback"], [10.0, 20.0, 10.0])

    def test_rejects_bad_bins(self):
        with self.assertRaises(ValueError):
            quantile_cuts(np.zeros((5, 2)), bins=1)
        with self.assertRaises(ValueError):
            quantile_cuts(np.zeros(5), bins=2)

    def test_memory_accepts_a_shared_cut_frame(self):
        # The identity is a common alphabet only if every stream is read
        # through the same cut points.  Passing `cuts` must take precedence
        # over the stream's own quantiles, so two streams with different
        # marginals still speak the same keys.
        x = np.zeros((40, 1))
        x[10, 0] = 10.0
        x[20, 0] = -10.0
        shared = quantile_cuts(np.zeros((10, 1)), bins=2)  # median = 0
        memory = milestone_memory(
            x, boundaries=np.asarray([10, 20]),
            cuts=shared,
        )
        expected = boundary_keys(
            x, np.asarray([10, 20]), bins=2, cuts=shared
        )
        self.assertEqual(memory["keys"], expected)

    def test_next_milestone_names_the_frequent_successor(self):
        # keys alternate 0 -> 1 (twice) and 1 -> 0 (once); the prediction
        # from key 0 must be key 1 and must carry the recorded outcome.
        x = np.zeros((40, 1))
        x[10, 0] = 10.0
        x[20, 0] = -10.0
        x[30, 0] = 10.0
        memory = milestone_memory(
            x, feedback=np.ones((40, 1)),
            boundaries=np.asarray([10, 20, 30]),
        )
        prediction = next_milestone(memory, memory["keys"][0])
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction["count"], 2)
        self.assertAlmostEqual(prediction["mean_duration"], 10.0)
        self.assertAlmostEqual(prediction["mean_feedback"], 10.0)
        self.assertIsNone(next_milestone(memory, (9,)))


class MilestoneCoverageTests(unittest.TestCase):
    def test_full_coverage_when_every_identity_was_learned(self):
        report = milestone_coverage(
            [(0, 1), (0, 1), (1, 0)], [(0, 1), (1, 0)]
        )
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["covered"], 3)
        self.assertAlmostEqual(report["fraction"], 1.0)
        self.assertEqual(report["unseen"], [])

    def test_unseen_identities_are_named_not_silently_skipped(self):
        # A stream from a different policy visits identities the memory never
        # saw; that must be reported, which is the difference between an
        # abstention and a silent skip.
        report = milestone_coverage(
            [(0, 1), (1, 1), (1, 1), (0, 0)], [(0, 1)]
        )
        self.assertEqual(report["covered"], 1)
        self.assertAlmostEqual(report["fraction"], 1.0 / 4.0)
        self.assertEqual(report["unseen"], [[0, 0], [1, 1]])

    def test_empty_stream_reports_zero(self):
        report = milestone_coverage([], [(0, 1)])
        self.assertEqual(report["total"], 0)
        self.assertAlmostEqual(report["fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
