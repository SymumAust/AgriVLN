# AgriVLN - Qubit-based subtask state tracker
#
# Each subtask has two qubits:
#   start qubit  |0> = pending, |1> = doing   (evidence that the subtask has started)
#   done  qubit  |0> = not yet, |1> = done    (evidence that it is finished; strict mode only)
# Each time step the VLM talks about ONE subtask in its <state> line. Only that subtask's
# qubits are rotated (RY by +/- delta) toward what the VLM said; the other subtasks are left
# unchanged (v3 fix: being "not mentioned" is no longer counted as evidence against).
# The state kept is the more likely one.
#
# strict=False : "done" is accepted at once.
# strict=True  : "done" also needs evidence over several steps, and a jump
#                pending -> done counts only as evidence for "doing".
# delta = pi with strict=False reproduces the original AgriVLN behaviour exactly.

import math

try:
   import pennylane as qml
   _dev = qml.device("default.qubit", wires=1)

   @qml.qnode(_dev)
   def _probs(theta):
      qml.RY(theta, wires=0)
      return qml.probs(wires=0)

   def prob_one(theta):
      return float(_probs(theta)[1])

   BACKEND = "pennylane"
except ImportError:  # same maths without PennyLane: P(|1>) = sin^2(theta/2)
   def prob_one(theta):
      return math.sin(theta / 2) ** 2

   BACKEND = "numpy"


def _rot(angle, step):
   return min(max(angle + step, 0.0), math.pi)


class QubitTracker:

   def __init__(self, n_subtasks, delta=math.pi / 3, strict=False, classical=False):
      # classical=True: same votes, but the belief is a plain vote share (angle / pi)
      # instead of a qubit probability sin^2(angle / 2). Both cross 0.5 at the same angle,
      # so the decisions are identical; this is the classical control.
      self.prob = (lambda a: a / math.pi) if classical else prob_one
      self.delta = delta
      self.strict = strict
      self.theta = [0.0] * n_subtasks      # start qubits
      self.phi = [0.0] * n_subtasks        # done qubits
      self.state = ["pending"] * n_subtasks

   def update(self, index, said):
      """index: 0-based subtask the VLM talked about (None = it said nothing usable).
      said: the state the VLM says that subtask is in now ('pending'/'doing'/'done').
      Returns (kept_states, p_doing, p_done)."""
      if index is not None and 0 <= index < len(self.state) and self.state[index] != "done":
         j, o, d = index, said, self.delta
         if not self.strict:
            if o == "done":
               self.state[j] = "done"
            else:
               self.theta[j] = _rot(self.theta[j], d if o == "doing" else -d)
               self.state[j] = "doing" if self.prob(self.theta[j]) > 0.5 else "pending"
         else:
            if o == "done" and self.state[j] == "doing":
               self.phi[j] = _rot(self.phi[j], d)
            elif o in ("done", "doing"):          # pending -> done jump = evidence for doing only
               self.theta[j] = _rot(self.theta[j], d)
               if o == "doing" and self.state[j] == "doing":
                  self.phi[j] = _rot(self.phi[j], -d)
            else:                                 # "pending"
               self.theta[j] = _rot(self.theta[j], -d)
               self.phi[j] = _rot(self.phi[j], -d)
            if self.state[j] == "doing" and self.prob(self.phi[j]) > 0.5:
               self.state[j] = "done"
            else:
               self.state[j] = "doing" if self.prob(self.theta[j]) > 0.5 else "pending"
      p_doing = [1.0 if s == "done" else round(self.prob(t), 3) for s, t in zip(self.state, self.theta)]
      p_done = [1.0 if s == "done" else round(self.prob(f), 3) for s, f in zip(self.state, self.phi)]
      return list(self.state), p_doing, p_done

   def all_done(self):
      return all(s == "done" for s in self.state)
