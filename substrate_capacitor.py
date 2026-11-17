"""Is u a capacitor?  Measure the three things a capacitor is defined by.

    discrete            u(t) = (1-b)u(t-1) + b*h(t)
    continuous limit    du/dt = (h - u)/tau,   tau = 1/b

    DC gain             a constant input drives u to the input          (gain 1)
    step response       rises with time constant tau
    impulse response    decays as b*(1-b)^t, i.e. exp(-t/tau)

and then the three things a capacitor in a circuit also has, which this one does not.
"""
import numpy as np

T = 400


def step_response(b, T=T):
    u, out = 0.0, []
    for _ in range(T):
        u = (1.0 - b) * u + b * 1.0
        out.append(u)
    return np.array(out)


def impulse_response(b, T=T):
    u, out = 0.0, []
    for t in range(T):
        u = (1.0 - b) * u + b * (1.0 if t == 0 else 0.0)
        out.append(u)
    return np.array(out)


print("  beta    tau=1/beta   t63 measured   DC gain   impulse decay / beta")
for b in (0.5, 0.2, 0.1, 0.05, 0.02):
    st = step_response(b)
    t63 = int(np.argmax(st >= 0.632))
    dc = st[-1]
    imp = impulse_response(b)
    ratio = imp[10] / (b * (1 - b) ** 10)          # closed form: b*(1-b)^t
    print("  %.2f      %6.1f        %4d          %.4f     %.12f" % (b, 1 / b, t63, dc, ratio))

print("\n  so: DC gain 1, time constant 1/beta, and the impulse decays exactly as exp(-t/tau)")
print("  u is a capacitor to ground, in series with a resistor: an RC low-pass")

we = 4.0                                     # energy stored, 0.5*C*u^2 in circuit terms
print("\n  and it holds charge: after 20 steps at beta=0.1 the store is %.3f of the input" %
      float(step_response(0.1)[19]))

print("\n  what a circuit capacitor has that this one does not:")
print("   1  a per-element value.  beta is global, so every node has the same capacitance.")
print("   2  charge sharing.  u_i never sees u_j; two capacitors connected by a wire would")
print("      exchange charge and settle at a common level.  Here nothing is conserved across")
print("      nodes -- only within one node's own leak.")
print("   3  an equilibrium solve.  a resistive network relaxes to a fixed point (solve a linear")
print("      system).  This one only takes explicit steps; there is no steady state to compute.")
