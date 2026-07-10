# DQN — replacing the table with a network

Unlike the other three entries here, this one isn't a blow-by-blow account of
a debugging session — DQN's implementation predates this project's habit of
writing things up as they happened. What follows is what it does and why,
written after the fact, for the same reason the other docs exist: so the
design decisions are somewhere other than in the diff.

## Why move on from Q-learning at all

[Q-learning](01-q-learning.md) worked, but its ceiling was structural: a
table indexed by discretized state has size `bins^dimensions`, and coverage
of that table gets worse, not better, as the grid gets finer — the states
that most need precision (near the failure boundary) are the ones a decent
policy visits least. The fix isn't a better table. It's not using a table.

## The core swap

`QLearningAgent.getQValues(state: string)` returns a row out of a
`Map<string, number[]>`. `DQNAgent.getQValues(state: number[])` returns the
output of a forward pass through a small neural network (two 64-unit hidden
layers, ReLU, same `DenseLayer`/`NeuralNetwork` building blocks every other
agent in this project uses). Same shape of answer — one Q-value per
action — computed instead of looked up. The state is passed in as its raw
continuous `number[]` form; there's no discretization step at all, so the
`bins^dimensions` problem doesn't exist here by construction.

That swap alone doesn't work if you just start backpropagating on every
step, though — it introduces two new problems tabular Q-learning never had,
because a table has no notion of "nearby" states interfering with each
other and a network very much does.

## Problem 1: correlated samples

Consecutive frames of one episode are extremely similar to each other —
almost the same state, again and again. Training a network on a stream of
highly correlated samples pulls it hard in whatever direction that one
trajectory is going, then hard in a different direction on the next
trajectory, which is a recipe for oscillation instead of convergence.
`ReplayBuffer` (`src/lib/ReplayBuffer.ts`) exists to break that correlation:
every transition gets stored, and `DQNAgent.replay()` samples a random batch
of 32 from up to 50,000 stored transitions before each training step —
turning a highly-correlated stream into something closer to the i.i.d. data
SGD actually assumes.

## Problem 2: chasing a moving target

The TD target for Q-learning is `reward + gamma * max(Q(nextState))`. If
`Q` is the same network you're actively updating, then every gradient step
changes the target you're trying to hit on the very next step — the network
is chasing its own tail. `DQNAgent` keeps a second, frozen copy
(`targetBrain`) that only gets synced to match the live network
(`brain`) every `targetUpdateFreq` (200) training steps. The TD target is
computed from this frozen copy, so it stays still long enough for the live
network to actually converge toward it before it moves again.

## Exploration: still epsilon-greedy

Q-learning's later iterations moved to UCB exploration (see
[01-q-learning.md](01-q-learning.md)) because count-based confidence bonuses
are cheap to compute from a table's visit counts. A network doesn't expose
per-state visit counts the same way, so DQN keeps the simpler, standard
approach instead: epsilon-greedy, starting fully random (`epsilon = 1.0`)
and decaying multiplicatively (`× 0.9995` per episode) down to a floor
(`epsilonMin = 0.01`), so the agent always keeps a small amount of randomness
rather than ever fully committing to what it currently believes is best.

## What carried over unchanged

Because every algorithm in this project targets the same `Task` interface,
none of the physics, reward shaping, or episode-termination logic needed to
be rewritten for DQN — `DoublePendulumTask`/`SinglePendulumTask` are shared
verbatim with Q-learning, PPO, and REINFORCE. The only things that changed
between the tabular and neural agent are inside `DQNAgent` itself:
state representation (string key → raw vector), the lookup itself (map →
forward pass), and the two stabilizers above.

## Where this sits in the lineup

DQN gets discrete actions right — no discretization of the *state*, still a
fixed menu of *actions* to choose from (see `THRUST_LEVELS` in the worker
wiring). The next problem was continuous *actions*: precisely how hard to
push, not just which of seven preset thrust levels to pick. That's what
[REINFORCE](03-reinforce.md) takes on next, at the cost of a new instability
DQN never had to deal with.
