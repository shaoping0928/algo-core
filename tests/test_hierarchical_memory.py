"""The hierarchical memory levels and the variable relation registry."""
import unittest

import numpy as np

from hierarchical_memory import HierarchicalMemory
from memory_tree import MemoryTree
from variable_registry import VariableRegistry


class MemoryTreeTests(unittest.TestCase):
    def test_clusters_and_queries(self):
        rng = np.random.RandomState(3)
        tree = MemoryTree(
            2, max_fanout=3, leaf_capacity=8, merge_radius=1.0
        )
        for center in (np.asarray([0.0, 0.0]), np.asarray([5.0, 0.0]),
                       np.asarray([0.0, 5.0])):
            for _ in range(12):
                tree.insert(center + rng.normal(0.0, 0.1, 2))
        self.assertGreaterEqual(len(tree.leaves), 1)
        hit = tree.query(np.asarray([5.05, 0.02]))
        self.assertIsNotNone(hit)
        self.assertLess(hit["distance"], 0.5)
        self.assertAlmostEqual(float(hit["leaf"].mean[0]), 5.0, places=1)

    def test_split_increases_granularity(self):
        rng = np.random.RandomState(5)
        tree = MemoryTree(
            2, max_fanout=4, leaf_capacity=6,
            merge_radius=5.0, split_variance=0.05,
        )
        for _ in range(48):
            tree.insert(rng.uniform(0.0, 4.0, 2))
        self.assertGreater(tree.splits, 0)

    def test_persistence_compares_distinct_memories(self):
        rng = np.random.RandomState(7)
        tree = MemoryTree(
            2, max_fanout=3, leaf_capacity=8, merge_radius=0.8
        )
        for _ in range(10):
            tree.insert(np.asarray([1.0, 0.0]) + rng.normal(0.0, 0.02, 2))
        for _ in range(10):
            tree.insert(np.asarray([1.0, 0.04]) + rng.normal(0.0, 0.02, 2))
        for _ in range(4):
            tree.insert(np.asarray([0.0, 1.0]) + rng.normal(0.0, 0.02, 2))
        persistent = tree.persistence(np.asarray([1.0, 0.0]))
        temporary = tree.persistence(np.asarray([0.0, 1.0]))
        self.assertGreater(persistent["fraction"], temporary["fraction"])

    def test_parent_summary_is_lossy_and_expandable(self):
        rng = np.random.RandomState(13)
        tree = MemoryTree(
            3, max_fanout=2, leaf_capacity=6, merge_radius=0.7,
            keyword_dim=2,
        )
        centers = (
            np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 1.0, 0.0]),
            np.asarray([1.0, 2.0, 0.0]), np.asarray([1.0, 3.0, 0.0]),
            np.asarray([1.0, 4.0, 0.0]),
        )
        for center in centers:
            for _ in range(8):
                tree.insert(center + rng.normal(0.0, 0.05, 3))
        parents = [
            node for node in tree.nodes.values()
            if not node.leaf and node.keywords is not None
        ]
        self.assertTrue(parents)
        parent = parents[0]
        expanded = tree.expand(parent.node_id)
        self.assertGreaterEqual(len(expanded), 2)
        self.assertTrue(all(item["error"] < 0.5 for item in expanded))
        keywords = tree.keyword_persistence(parent.node_id)
        self.assertTrue(keywords)
        self.assertTrue(all("persistent" in item for item in keywords))

    def test_registry_registers_summary_keywords(self):
        rng = np.random.RandomState(17)
        tree = MemoryTree(
            3, max_fanout=2, leaf_capacity=6, merge_radius=0.7,
            keyword_dim=2,
        )
        for center in (np.asarray([0.0, 0.0, 0.0]),
                       np.asarray([1.0, 1.0, 0.0]),
                       np.asarray([2.0, 2.0, 0.0])):
            for _ in range(8):
                tree.insert(center + rng.normal(0.0, 0.05, 3))
        registry = VariableRegistry()
        registry.register_tree(tree)
        registry.register_summaries(tree)
        self.assertTrue(any(
            key.startswith("kw:") for key in registry.variables
        ))
        self.assertTrue(any(
            kind == "summarizes"
            for _source, _target, kind, _weight in registry.relations
        ))


class VariableRegistryTests(unittest.TestCase):
    def test_relations_shape_credibility_and_learning_rate(self):
        registry = VariableRegistry(base_prior=4.0, parent_gain=1.0)
        registry.add("root", "root", np.zeros(1), support=100.0)
        registry.add("child", "context", np.zeros(1), support=5.0)
        registry.add("orphan", "context", np.zeros(1), support=5.0)
        registry.relate("child", "root", "part_of")
        self.assertGreater(
            registry.credibility("child"), registry.credibility("orphan")
        )
        self.assertLess(
            registry.learning_rate("child"),
            registry.learning_rate("orphan"),
        )

    def test_fast_learning_rate_tracks_parent_and_persistence(self):
        registry = VariableRegistry(base_prior=4.0)
        registry.add("parent", "context", np.zeros(1), support=80.0)
        registry.add("fast", "context", np.zeros(1), support=2.0,
                     timescale="fast", metadata={"persistence": 0.0})
        registry.relate("fast", "parent", "tracks")
        high = registry.learning_rate("fast")
        registry.variables["fast"].metadata["persistence"] = 0.9
        low = registry.learning_rate("fast")
        self.assertGreater(high, 0.0)
        self.assertLess(low, high)

    def test_support_slows_a_slow_variable(self):
        registry = VariableRegistry(base_prior=4.0)
        registry.add("v", "context", np.zeros(1), support=1.0)
        first = registry.learning_rate("v")
        registry.update("v", np.ones(1), weight=20.0)
        self.assertLess(registry.learning_rate("v"), first)


class HierarchicalMemoryTests(unittest.TestCase):
    def _feed(self, memory, episode, context, count, done=True):
        for step in range(count):
            feedback = np.asarray([1.0, 0.0])
            error = 0.01
            if step > 8 and step % 12 == 0:
                feedback = np.asarray([0.0, 1.0])
                error = 1.0
            memory.observe(
                observation=np.asarray([context[0], context[1], step * 0.01]),
                feedback=feedback, error=error, context=context,
                done=(done and step == count - 1), episode=episode,
            )

    def test_builds_all_levels_and_persistent_vs_temporary(self):
        memory = HierarchicalMemory(
            context_dim=2, observation_dim=3, window=2,
            salience_z=0.5, phase_split=1.0,
        )
        self._feed(memory, 0, np.asarray([0.0, 0.0]), 40)
        self._feed(memory, 1, np.asarray([3.0, 0.0]), 40)
        self.assertGreater(len(memory.frames), 0)
        self.assertGreater(len(memory.events), 0)
        self.assertGreater(len(memory.phases), 0)
        self.assertEqual(len(memory.episodes), 2)
        snapshot = memory.snapshot()
        self.assertIn("root", snapshot["registry"])
        self.assertTrue(any(
            key.startswith("level:") for key in snapshot["registry"]
        ))
        persistent = memory.persistence(np.asarray([1.0, 0.0]))
        temporary = memory.persistence(np.asarray([0.0, 1.0]))
        self.assertGreater(persistent["score"], temporary["score"])
        self.assertGreater(persistent["score"], 0.0)

    def test_phase_closes_when_context_switches(self):
        memory = HierarchicalMemory(
            context_dim=2, observation_dim=3, window=2,
            salience_z=0.5, phase_split=0.5, max_phase_gap=100,
        )
        self._feed(memory, 0, np.asarray([0.0, 0.0]), 20, done=False)
        self._feed(memory, 0, np.asarray([4.0, 0.0]), 20, done=True)
        self.assertGreaterEqual(len(memory.phases), 1)

    def test_phase_summary_expands_into_an_approximate_flow(self):
        memory = HierarchicalMemory(
            context_dim=2, observation_dim=3, window=2,
            salience_z=0.5, phase_split=100.0, max_phase_gap=1000,
            keyword_dim=1,
        )
        for step in range(30):
            context = np.asarray([step * 0.1, 0.0])
            feedback = np.asarray([1.0, 0.0])
            error = 0.01
            if step > 8 and step % 8 == 0:
                feedback = np.asarray([0.0, 1.0])
                error = 1.0
            memory.observe(
                observation=np.asarray([context[0], context[1], step * 0.01]),
                feedback=feedback, error=error, context=context,
                done=(step == 29), episode=0,
            )
        self.assertGreaterEqual(len(memory.phases), 1)
        summary = memory.phase_summary(0)
        self.assertIn("mean", summary)
        self.assertGreaterEqual(summary["residual"], 0.0)
        expansion = memory.expand_phase(0)
        self.assertGreaterEqual(len(expansion["events"]), 1)
        for item in expansion["events"]:
            self.assertTrue(np.isfinite(item["error"]))


if __name__ == "__main__":
    unittest.main()
