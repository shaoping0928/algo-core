import unittest
from types import SimpleNamespace

from stage_option_search_probe import summarize


class StageOptionSearchProbeTests(unittest.TestCase):
    def test_conditional_options_are_summarized_separately(self):
        option_a = ((0, 1),)
        option_b = ((1, 1),)
        plans = (
            SimpleNamespace(
                options=(option_a,),
                known_mass=1.0,
                outcome_mass=(("terminal", 0.2), ("open", 0.8)),
                expected_elapsed=4.0,
                elapsed_distribution=((4.0, 1.0),),
            ),
            SimpleNamespace(
                options=(option_b,),
                known_mass=1.0,
                outcome_mass=(("terminal", 0.8), ("open", 0.2)),
                expected_elapsed=8.0,
                elapsed_distribution=((8.0, 1.0),),
            ),
        )
        result = SimpleNamespace(
            counts={},
            leaves=(SimpleNamespace(outcome="terminal"),),
            unresolved=(),
            complete=True,
            plans=lambda: plans,
        )

        report = summarize([result], horizon=6)

        self.assertEqual(report["option_sequences"], 2)
        self.assertAlmostEqual(report["option_terminal_rate_mean"], 0.5)
        self.assertAlmostEqual(report["option_elapsed_mean"], 6.0)
        self.assertAlmostEqual(report["within_horizon"], 0.5)


if __name__ == "__main__":
    unittest.main()
