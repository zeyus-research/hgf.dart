"""``get_update_sequence`` mutates the list it iterates over.

The final loop moves the ef-state posteriors (and ``categorical_state_update``)
to the end with ``update_sequence.remove(step)`` + ``append(step)`` inside
``for step in update_sequence``. Python's list iterator advances by index, so
the element after each moved step is skipped, and when the iterator reaches a
moved step again it moves it again. The deferred steps don't end up in their
original order: with ef nodes 0, 1, 2 they come out as below.
"""

from _common import steps
from pyhgf.model import Network

for n in (2, 3):
    net = Network().add_nodes(kind="ef-state", n_nodes=n, learning="hgf-1")
    deferred = [i for i, name in steps(net) if name == "posterior_update_exponential_family_dynamic"]
    print(f"{n} ef nodes: deferred posteriors run for nodes {deferred}")

# The same loop on plain labels; 'D*' are the steps to defer.
seq = ["D0", "a", "D1", "b", "D2", "c"]
for step in seq:
    if step.startswith("D"):
        seq.remove(step)
        seq.append(step)
print("loop as written:", seq)

seq = ["D0", "a", "D1", "b", "D2", "c"]
seq = [s for s in seq if not s.startswith("D")] + [s for s in seq if s.startswith("D")]
print("stable partition:", seq)
