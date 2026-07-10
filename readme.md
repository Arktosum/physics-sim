<div align="center">

# rl-physics-sim

**A physics engine and four reinforcement-learning algorithms, built entirely from scratch in TypeScript.**

Tabular Q-learning &rarr; DQN &rarr; REINFORCE &rarr; PPO, each trained live in your browser on a pendulum-balancing task, with no physics library and no ML library underneath any of it.

[**Live demos**](#-demos) · [**Getting started**](#-getting-started) · [**Architecture**](#-architecture) · [**The journey**](#-the-journey)

</div>

---

## Overview

This project is a small research lab you can run in a browser tab. A hand-built 2D physics engine simulates a cart balancing one or two pendulums, and four different reinforcement-learning algorithms are trained against it — side by side, so you can watch each one's strengths and failure modes directly rather than take them on faith.

Nothing is pre-recorded and nothing is a pre-trained checkpoint played back: every demo spins up a real training loop the moment you open the page, running on a dedicated Web Worker so the UI never blocks. You can save a trained model to disk mid-run and load it back in later.

## ✨ Features

- **4 algorithms, 2 tasks, 8 demos** — tabular Q-learning, DQN, REINFORCE, and PPO, each trained on both a single pendulum and a double pendulum, so you can compare how a technique holds up as the problem gets harder.
- **Trains live, in-browser, off the main thread** — every demo runs its training loop on a Web Worker via `postMessage`, so the physics render and UI stay smooth no matter how hard the agent is training.
- **Live Demo mode** — flip a toggle to switch from the noisy training rollout (exploration included) to a second environment stepped in real time by the policy's deterministic, noise-free action. This is the honest answer to "what has it actually learned so far."
- **Save / Load** — every agent serializes its weights (or its Q-table) to JSON. Save triggers a browser download; Load reads a file back in and resumes from exactly that point.
- **A from-scratch physics engine** — Verlet integration + Position-Based Dynamics, the same family of technique behind cloth and rope simulations. No physics library.
- **A from-scratch neural network library** — dense layers, ReLU, batched and single-sample forward/backward passes, all hand-rolled on top of a small `Matrix` class. No ML library.
- **A real, documented debugging history** — every non-obvious bug, benchmark, and design decision made while building this is written up in [`docs/journey/`](docs/journey/index.html), in the order it actually happened.

## 🎮 Demos

Open [`index.html`](index.html) for the full showcase, or jump straight to any demo:

| Algorithm | Idea | Single Pendulum | Double Pendulum |
|---|---|---|---|
| **Q-Learning** | A lookup table over discretized states, tuned with UCB exploration and left-right symmetry augmentation. | [`q-learning-single.html`](q-learning-single.html) | [`q-learning-double.html`](q-learning-double.html) |
| **DQN** | A neural network replaces the table — continuous state input, no discretization, stabilized with a replay buffer and a target network. | [`dqn-single.html`](dqn-single.html) | [`dqn-double.html`](dqn-double.html) |
| **REINFORCE** | Vanilla policy gradients with a continuous Gaussian policy. Simple, and genuinely unstable on a chaotic system — watch the double pendulum demo's policy occasionally collapse. | [`reinforce-single.html`](reinforce-single.html) | [`reinforce-double.html`](reinforce-double.html) |
| **PPO** | The fix for REINFORCE's instability: a clipped trust-region objective plus GAE. The most capable agent here — the double pendulum demo can learn to balance for 60+ seconds. | [`ppo-single.html`](ppo-single.html) | [`ppo-double.html`](ppo-double.html) |

Each demo page shows the live simulation next to a metrics panel (score history, algorithm-specific health signals like KL divergence or exploration coverage, and an action-distribution chart), plus Live Demo and Save/Load controls.

## 🚀 Getting started

**Requirements:** Node.js 18+.

```bash
git clone https://github.com/Arktosum/rl-physics-sim.git
cd rl-physics-sim
npm install
npm run dev
```

Then open the printed local URL and click into `index.html` — every demo trains from scratch the moment its page loads. No GPU, no external services, no API keys.

### Other scripts

```bash
npm run build     # type-checks with tsc, then produces a static dist/ with all 14 pages
npm run preview   # serve the production build locally to sanity-check it
```

`npm run build` is genuinely required if you want to deploy this — Vite only bundles `index.html` by default, so [`vite.config.ts`](vite.config.ts) explicitly lists every HTML entry point (8 demos + the landing page + 5 journey pages).

## 🧠 Architecture

```
src/
  physics/     Vector, Constraint, Environment, Actuator — the physics engine core
  state/       PointMass — Verlet-integrated point mass
  renderer/    CanvasRenderer — draws physics state to a <canvas>
  tasks/       Task interface, SinglePendulumTask, DoublePendulumTask
  agents/      QLearningAgent, DQNAgent, ReinforceAgent, PPOAgent
  lib/         Matrix, NeuralNetwork, DenseLayer, ReLULayer — the NN library
  training/    One Trainer per algorithm, plus a shared Worker harness
    workers/   The 8 Web Workers that actually run training
  ui/          MetricsPanel — one config-driven panel, not 4 hand-built ones
  scenes/      Main-thread entry points, one per demo page
```

**The physics engine.** A `PointMass` is integrated with Verlet integration: instead of tracking velocity directly, it keeps its previous position and derives velocity implicitly from the difference between old and current position. Rods between points are `Constraint`s solved with Position-Based Dynamics — each constraint nudges connected points back to the correct distance, weighted by inverse mass, so heavy points don't get shoved around by light ones. An `Actuator` applies horizontal force to the cart based on whatever action the current agent chose.

**The agent/task boundary.** Every algorithm targets the same `Task` interface (`reset()` / `step()` / a normalized `number[]` state), so the physics, reward shaping, and episode-termination logic are written once per task and reused by all four algorithms — not duplicated four times.

**The Worker harness.** All 8 demos share one `runTrainingWorker()` function ([`src/training/workers/workerHarness.ts`](src/training/workers/workerHarness.ts)) that owns the actual plumbing: throttled `postMessage` for metrics/frames, worker-stall detection, the Save/Load message protocol, and the real-time Live Demo eval loop. Each worker only has to define its own `tick()`, `buildMetrics()`, and `buildFrame()` — the training logic differs meaningfully per algorithm (a discrete Q-table vs. a replay buffer vs. an on-policy rollout buffer), so that part is deliberately *not* shared.

**The metrics panel.** Every demo's sidebar is built at runtime by one config-driven `MetricsPanel` class from a per-algorithm config object ([`src/ui/metricsConfigs/`](src/ui/metricsConfigs/)) rather than four separate hand-authored DOM/UI classes — new stats get added in exactly one place, not mirrored by hand across multiple files.

## 📖 The Journey

[`docs/journey/`](docs/journey/index.html) is a from-the-trenches write-up of how this was actually built, in order — including the wrong turns:

1. **[Q-Learning](docs/journey/01-q-learning.html)** — gets surprisingly far with UCB exploration and symmetry augmentation, then hits a real structural wall: the `bins^dimensions` blowup of discretizing a continuous state space.
2. **[DQN](docs/journey/02-dqn.html)** — replaces the table with a network, and picks up the two stabilizers that make learning from one converge: a replay buffer and a frozen target network.
3. **[REINFORCE](docs/journey/03-reinforce.html)** — tackles continuous actions, but vanilla policy gradients turn out to be mathematically unstable on a chaotic system: a single lucky episode can send the policy into permanent overconfidence.
4. **[PPO](docs/journey/04-ppo.html)** — the fix (a trust-region / clipped objective), followed by a much longer investigation into *why the browser stuttered*: a Worker migration, a genuine GAE bootstrap bug, and a real learning-rate regression that mini-batching quietly introduced — caught by building a second, simpler task specifically to sanity-check against.

## 🛠️ Tech stack

- **TypeScript** — strict, no `any` beyond a handful of deliberate weight-serialization casts.
- **Vite** — dev server + multi-page static build, zero framework.
- **Web Workers** — one per training run, communicating over `postMessage`.
- **Canvas 2D** — all rendering, no WebGL.
- No physics library, no ML/RL library, no UI framework. Everything under `src/physics/`, `src/lib/`, and `src/agents/` is hand-written.

## 📄 License

No license file yet — all rights reserved by default. Open an issue if you'd like to use this and I'll sort one out.






