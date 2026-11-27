"""Raw arithmetic facts -> associative memory -> generic relation search.

The text format matches the nanoGPT arithmetic prompts (`a+b=` / `a*b=`), but the learner is
not shown worked solutions. An external oracle supplies only final answers.

Phase 1 stores observed one-digit input/output associations in a generic key-value memory.
Phase 2 stores raw three-digit equations in the same memory, then searches those retrieved
examples with a generic numeric signal basis (raw columns, simple transforms and pairwise
products). The formula is not supplied. The basis is supplied, so results establish learning
only within that hypothesis class.

This isolates the proposed memory -> strategy-search path. It is not a char-level nanoGPT run:
the input adapter parses the raw equation into two numeric operand signals and an operator tag.
"""
import numpy as np

from substrate_search import basis, greedy_determine


class AssociativeMemory:
    """Exact episodic key/value store; stores observations, not an arithmetic procedure."""

    def __init__(self):
        self.values = {}
        self.samples = {}

    def observe(self, op, a, b, result, phase):
        self.values[(op, int(a), int(b))] = int(result)
        self.samples[(phase, op, int(a), int(b))] = int(result)

    def recall(self, op, a, b):
        return self.values.get((op, int(a), int(b)))

    def examples(self, op, phase):
        return [(a, b, y) for (p, o, a, b), y in self.samples.items()
                if p == phase and o == op]


def oracle(op, a, b):
    return a + b if op == "+" else a * b


def fit_rule(op, examples):
    d = {
        "a": np.asarray([a for a, _b, _y in examples], dtype=float),
        "b": np.asarray([b for _a, b, _y in examples], dtype=float),
        "target": np.asarray([y for _a, _b, y in examples], dtype=float),
    }
    chosen, coef, residual, _scale = greedy_determine(
        d, "target", ["a", "b"], nonlinear=True, k_max=4, holdout=0.3)
    return chosen, coef, residual


def predict_rule(chosen, coef, pairs):
    d = {
        "a": np.asarray([a for a, _b in pairs], dtype=float),
        "b": np.asarray([b for _a, b in pairs], dtype=float),
    }
    B = basis(d, ["a", "b"], nonlinear=True)
    X = np.column_stack([np.ones(len(pairs))] + [B[name] for name in chosen])
    return X @ coef


def main():
    rng = np.random.RandomState(1337)
    memory = AssociativeMemory()
    fact_examples = {}
    form_examples = {}

    # Memory stage: all 100 one-digit associations per operator.
    for op in ("+", "*"):
        rows = [(a, b, oracle(op, a, b)) for a in range(10) for b in range(10)]
        rng.shuffle(rows)
        fact_examples[op] = rows
        for a, b, y in rows:
            memory.observe(op, a, b, y, phase="facts")

    # Form-search stage: additional raw three-digit equations only. A fixed input scale avoids
    # making candidate signals nearly collinear just because digit lengths differ. These are
    # experience records, not worked solutions; memory stores the outcomes and search sees rows.
    for op in ("+", "*"):
        rows = []
        for _ in range(3000):
            a = int(rng.randint(100, 1000))
            b = int(rng.randint(100, 1000))
            y = oracle(op, a, b)
            rows.append((a, b, y))
            memory.observe(op, a, b, y, phase="form")
        form_examples[op] = rows

    print("input examples (no worked steps):")
    print("  3+7=10")
    print("  4*6=24")
    print("memory: 100 one-digit facts/operator + 3000 raw 3-digit observations/operator")
    print("stored keys total:", len(memory.values))

    for op in ("+", "*"):
        rows = fact_examples[op]
        hits = sum(memory.recall(op, a, b) == y for a, b, y in rows)
        print("\noperator %s: one-digit memory recall %d/%d" % (op, hits, len(rows)))

        rows = memory.examples(op, phase="form")
        chosen, coef, residual = fit_rule(op, rows)
        print("  search-selected signals: %s" % chosen)
        active = [name for name, weight in zip(["bias"] + chosen, coef)
                  if abs(weight) > 1e-8]
        print("  nonzero terms: %s" % active)
        print("  coefficients (intercept first): %s" % np.array2string(coef, precision=6))
        print("  raw-example validation residual: %.3e" % residual)

        # Fresh, unmemorized operands, including lengths below/above the form-search scale.
        rng_test = np.random.RandomState(9000 + (op == "*"))
        out = []
        for digits in (2, 3, 4, 6):
            lo = 10 ** (digits - 1)
            hi = 10**digits
            pairs = []
            while len(pairs) < 128:
                a = int(rng_test.randint(lo, hi))
                b = int(rng_test.randint(lo, hi))
                if memory.recall(op, a, b) is None:
                    pairs.append((a, b))
            pred = predict_rule(chosen, coef, pairs)
            correct = sum(int(round(p)) == oracle(op, a, b)
                          for (a, b), p in zip(pairs, pred))
            memory_hits = sum(memory.recall(op, a, b) is not None for a, b in pairs)
            out.append((digits, correct, len(pairs), memory_hits))
        print("  fresh operands: digits  exact      memory hits")
        for digits, correct, n, mem_hits in out:
            print("                %2d   %3d/%-3d    %3d/%d"
                  % (digits, correct, n, mem_hits, n))

        # Negative control: shuffled form labels should not support an exact compact rule.
        shuffled = list(rows)
        ys = [x[2] for x in shuffled]
        rng.shuffle(ys)
        decoy = [(a, b, y) for (a, b, _), y in zip(shuffled, ys)]
        _bad_terms, _bad_coef, bad_resid = fit_rule(op, decoy)
        print("  shuffled-label control residual: %.3e (must be nonzero)" % bad_resid)

    print("\nBoundary:")
    print("  memory provides observed one-digit correspondences.")
    print("  search selects a relation from a generic basis; it does not invent the basis.")
    print("  the parser's numeric operands and the candidate signal vocabulary are scaffolding.")
    print("  this tests memory + relation search, not raw-character perception or the Flappy sandbox.")


if __name__ == "__main__":
    main()
