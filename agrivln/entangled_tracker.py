# AgriVLN - ENTANGLED subtask tracker (order between subtasks)
#
# Same votes as QubitTracker(strict=True): every VLM claim about subtask k adds +delta or -delta
# to that subtask's start angle theta_k or done angle phi_k (clamped to [0, pi]).
# Difference: the qubits are no longer independent. One circuit with 2 qubits per subtask:
#
#   s1 : RY(theta_1)
#   s2 : CRY(theta_2) controlled by s1        "subtask 2 can only have started if subtask 1 has"
#   s3 : CRY(theta_3) controlled by s2
#   dk : CRY(phi_k)   controlled by sk        "subtask k can only be done if it has started"
#
# The circuit is rebuilt from the current angles at every step and measured:
#   P(sk = 1) = belief that subtask k has started   (now depends on all earlier subtasks)
#   P(dk = 1) = belief that subtask k is done
# Kept state: done if P(dk) > 0.5, else doing if P(sk) > 0.5, else pending.
#
# classical=True computes the same beliefs with the chain rule (no circuit):
#   P(s1) = sin^2(theta_1/2),  P(sk) = P(s{k-1}) * sin^2(theta_k/2),  P(dk) = P(sk) * sin^2(phi_k/2)
# Because every control is measured in the same basis, the two give identical numbers: the
# entanglement encodes a dependency, and a classical chain encodes the same dependency.

import math

from agrivln.quantum_tracker import _rot

try:
    import pennylane as qml
    BACKEND = "pennylane"
except ImportError:
    qml = None
    BACKEND = "numpy"


def _s2(a):
    return math.sin(a / 2) ** 2


class EntangledTracker:

    def __init__(self, n_subtasks, delta=math.pi / 6, classical=False):
        self.n, self.delta = n_subtasks, delta
        self.classical = classical or qml is None
        self.theta = [0.0] * n_subtasks
        self.phi = [0.0] * n_subtasks
        self.state = ["pending"] * n_subtasks
        if not self.classical:
            wires = [f"s{k}" for k in range(n_subtasks)] + [f"d{k}" for k in range(n_subtasks)]
            dev = qml.device("default.qubit", wires=wires)

            @qml.qnode(dev)
            def circuit(theta, phi):
                qml.RY(theta[0], wires="s0")
                for k in range(1, n_subtasks):
                    qml.CRY(theta[k], wires=[f"s{k - 1}", f"s{k}"])
                for k in range(n_subtasks):
                    qml.CRY(phi[k], wires=[f"s{k}", f"d{k}"])
                return [qml.probs(wires=w) for w in wires]
            self._circuit = circuit

    def beliefs(self):
        if self.classical:
            ps, prev = [], 1.0
            for t in self.theta:
                prev = prev * _s2(t)
                ps.append(prev)
            pd = [p * _s2(f) for p, f in zip(ps, self.phi)]
            return ps, pd
        out = self._circuit(self.theta, self.phi)
        ps = [float(o[1]) for o in out[:self.n]]
        pd = [float(o[1]) for o in out[self.n:]]
        return ps, pd

    def update(self, index, said):
        """Same voting rule as QubitTracker(strict=True); returns (kept, p_started, p_done)."""
        if index is not None and 0 <= index < self.n and self.state[index] != "done":
            j, o, d = index, said, self.delta
            if o == "done" and self.state[j] == "doing":
                self.phi[j] = _rot(self.phi[j], d)
            elif o in ("done", "doing"):
                self.theta[j] = _rot(self.theta[j], d)
                if o == "doing" and self.state[j] == "doing":
                    self.phi[j] = _rot(self.phi[j], -d)
            else:
                self.theta[j] = _rot(self.theta[j], -d)
                self.phi[j] = _rot(self.phi[j], -d)
        ps, pd = self.beliefs()
        for k in range(self.n):
            if self.state[k] == "done":
                continue                                   # done stays done, as in QubitTracker
            self.state[k] = "done" if pd[k] > 0.5 else ("doing" if ps[k] > 0.5 else "pending")
        return list(self.state), [round(p, 3) for p in ps], [round(p, 3) for p in pd]

    def all_done(self):
        return all(s == "done" for s in self.state)

    def draw(self):
        if self.classical:
            return "(classical chain, no circuit)"
        return qml.draw(self._circuit, decimals=2)(self.theta, self.phi)
