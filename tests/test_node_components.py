import unittest

from node_components import ComponentBank
from node_graph import ASSOCIATION, CONCEPT, NodeGraph, TRANSITION_EDGE


class ComponentBankTests(unittest.TestCase):
    def make(self, **kwargs):
        graph = NodeGraph()
        base = graph.add_node(CONCEPT, "base")
        bank = ComponentBank(graph, **kwargs)
        return graph, base, bank

    def test_component_is_runtime_and_sign_nodes_are_concepts(self):
        graph, base, bank = self.make()
        component_id = bank.bind(base, "difference")

        self.assertNotIn(component_id, graph.nodes)
        self.assertEqual(len(bank.sign_nodes[component_id]), 3)
        for sign, node_id in bank.sign_nodes[component_id].items():
            node = graph.nodes[node_id]
            self.assertEqual(node.kind, CONCEPT)
            self.assertEqual(node.role, "component_state")
            self.assertEqual(node.component_id, component_id)
            self.assertEqual(node.sign, sign)

    def test_difference_emits_activate_hold_inhibit(self):
        graph, base, bank = self.make()
        component_id = bank.bind(base, "difference", smoothing=1.0)

        signs = [
            bank.step({base: 0.0})[component_id],
            bank.step({base: 1.0})[component_id],
            bank.step({base: 1.0})[component_id],
            bank.step({base: 0.0})[component_id],
        ]
        self.assertEqual(signs, [0, 1, 0, -1])
        self.assertEqual(
            bank.active_sign_nodes(),
            [bank.sign_nodes[component_id][-1]],
        )

    def test_threshold_rounds_to_three_states(self):
        graph, base, bank = self.make()
        component_id = bank.bind(
            base, "difference", smoothing=1.0,
            threshold=0.2, hysteresis=0.0,
        )
        signs = [
            bank.step({base: value})[component_id]
            for value in (0.0, 0.2, 0.4, 0.7, 0.7, 0.4)
        ]
        # differences: 0, 0.2, 0.2, 0.3, 0.0, -0.3
        self.assertEqual(signs, [0, 0, 0, 1, 0, -1])

    def test_smoothing_prevents_chatter(self):
        graph, base, bank = self.make()
        component_id = bank.bind(
            base, "difference", smoothing=0.25,
            threshold=0.2, hysteresis=0.0,
        )
        signs = [
            bank.step({base: value})[component_id]
            for value in (0.4, 0.0, 0.4, 0.0, 0.4, 0.0)
        ]
        self.assertEqual(signs, [0, 0, 0, 0, 0, 0])

    def test_hysteresis_holds_between_thresholds(self):
        graph, base, bank = self.make(threshold=0.4, hysteresis=0.2)
        component_id = bank.bind(base, "difference", smoothing=1.0)

        signs = [
            bank.step({base: 0.0})[component_id],
            bank.step({base: 0.7})[component_id],   # difference 0.7 -> +
            bank.step({base: 1.2})[component_id],   # difference 0.5 -> hold +
            bank.step({base: 1.2})[component_id],   # difference 0.0 -> 0
        ]
        self.assertEqual(signs, [0, 1, 1, 0])

    def test_sign_nodes_form_association_transition_and_memory(self):
        graph, base, bank = self.make()
        component_id = bank.bind(base, "difference", smoothing=1.0)
        bank.step({base: 1.0})
        plus = bank.sign_nodes[component_id][1]
        bank.step({base: 1.0})
        hold = bank.sign_nodes[component_id][0]

        associations = [
            key for key in graph.edges
            if key[2] == ASSOCIATION and {key[0], key[1]} == {base, plus}
        ]
        self.assertTrue(associations)
        self.assertIn((plus, hold, TRANSITION_EDGE), graph.edges)
        memory = bank.add_memory()
        self.assertIn(hold, graph.nodes[memory].members)
        self.assertIn((memory, hold, "membership"), graph.edges)

    def test_valence_updates_the_active_sign_node(self):
        graph, base, bank = self.make()
        component_id = bank.bind(base, "difference")
        bank.step({base: 1.0})
        active = bank.active_sign_nodes()
        bank.observe_valence(pain=1.0, dopamine=0.25, rate=0.5)

        node = graph.nodes[active[0]]
        self.assertAlmostEqual(node.value, 0.5)
        self.assertAlmostEqual(node.dopamine, 0.125)
        self.assertEqual(node.count, 1)

    def test_state_round_trip_preserves_graph_and_components(self):
        graph, base, bank = self.make()
        component_id = bank.bind(base, "difference")
        bank.step({base: 1.0})
        bank.step({base: 0.0})
        bank.observe_valence(pain=1.0, dopamine=0.0, rate=0.5)
        memory = bank.add_memory()
        state = bank.state_dict()

        restored = ComponentBank(NodeGraph()).load_state_dict(state)
        self.assertEqual(restored.specs.keys(), bank.specs.keys())
        self.assertEqual(restored.previous_sign, bank.previous_sign)
        self.assertIn(memory, restored.graph.nodes)
        self.assertEqual(
            restored.graph.snapshot(), bank.graph.snapshot()
        )
        for sign, node_id in restored.sign_nodes[component_id].items():
            self.assertAlmostEqual(
                restored.graph.nodes[node_id].value,
                bank.graph.nodes[node_id].value,
            )

    def test_capacitor_and_resistor_bind_and_step(self):
        graph, base, bank = self.make()
        capacitor = bank.bind(base, "capacitor", params=(0.5,))
        resistor = bank.bind(base, "resistor", params=(0.5,), component_id="r")
        result = bank.step({base: 1.0})
        self.assertEqual(set(result), {capacitor, resistor})
        self.assertIn(result[capacitor], (-1, 0, 1))
        self.assertIn(result[resistor], (-1, 0, 1))


if __name__ == "__main__":
    unittest.main()
