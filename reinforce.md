# REINFORCE Scene — Planned UI & Diagnostics Overhaul

Status: **planning / not yet implemented.** This is the agreed shape of the next
edit pass on `ReinforcePendulum_main.ts` and `ReinforceTrainer.ts`, written down
before touching code so we can sign off on the plan first.

---

## 1. Clean canvas, DOM sidebar for everything else

**Problem today:** `drawDiagnostics()` draws text/charts straight onto the
same `<canvas>` the physics renders to, via `ctx.fillText`/`ctx.fillRect`
layered on top of `renderer.render(task.env)`.

**Plan:** Canvas becomes environment-only. `renderer.render(task.env)` is the
only thing that touches it. All stats, charts, and controls move into a DOM
sidebar — extending the existing `uiContainer` div (currently just holds the
Speed/Save/Load controls) rather than inventing a second UI system.

- Numeric readouts → plain DOM elements (`<div>`/`<span>`), updated per frame
  or per episode as appropriate — no need to route them through canvas
  drawing at all.
- Charts → small dedicated `<canvas>` elements living in the sidebar, one per
  chart, each with their own draw call. Keeps the main sim canvas free of any
  drawing code that isn't `CanvasRenderer`.

## 2. Fast-forward stays mechanically the same, one caveat

`trainer.tick()`'s loop-until-`timeBudgetMs`-then-`setTimeout(0)` shape is
unchanged and still the thing to crank up for speed.

**Caveat to keep in mind, not something to fix:** DQN's training cost is
smooth (fixed batch of 32 every 4 steps). REINFORCE's is lumpy — `learn()`
only fires once per episode and costs proportional to that episode's length,
all synchronously inside one `doOneStep()` call. Longer episodes can make a
single training burst overshoot `timeBudgetMs` more than DQN ever did. Just a
different rhythm to expect, not a bug.

## 3. Render-pause toggle for fast-forwarding

New sidebar button: **"Pause Rendering"** (or similar).

- Toggling it OFF skips `renderer.render()` and all sidebar redraw work
  inside `renderLoop()`, freeing the main thread almost entirely for
  `trainer.tick()`'s training loop.
- `requestAnimationFrame` itself keeps running at a trickle underneath (just
  polling the toggle state, maybe cheaply updating an episode counter) rather
  than being cancelled outright — stopping `rAF` completely would need a
  separate mechanism to know when to resume drawing later.
- `trainer.tick()` is untouched either way — it was already fully decoupled
  from rendering, so this toggle only affects the render side.

## 4. Diagnostics — full rethink, not a port of the DQN panel

Starting from scratch rather than reusing the old panel's metric list, since
several of those (epsilon, per-action Q-values) have no REINFORCE equivalent.

### Outcome metrics (kept, same role as before)
- **Score per episode + moving average** — ground truth of "is it improving."
- **Survival time** — task-specific proxy, less sensitive to reward-shaping
  quirks than raw score.

### REINFORCE-specific health metrics (new)
- **Policy std / entropy over time.** For a Gaussian: `entropy = 0.5 * log(2πe · std²)`.
  This is *the* signal DQN never had — exploration is now baked into the
  network's own output rather than an external epsilon schedule. Collapsing
  to ~0 early means the policy stopped exploring before it was actually good;
  staying high forever means it's never converging. Gets its own chart.
- **Critic (baseline) loss.** Same idea as DQN's loss chart, but now it's
  regression against real returns `G_t` instead of Q-values. Should trend
  down smoothly if the baseline is learning.
- **Average |Advantage| magnitude.** Should shrink as the Critic gets better
  at predicting returns. Large or noisy late in training = stability warning.
- **Gradient clip rate.** % of Actor updates in the last episode where
  `clipGrad()` in `ReinforceAgent.learn()` actually clamped something. Direct
  visibility into whether the exploding-gradient issue we already fixed once
  is creeping back, far more actionable than the old placeholder "actor loss"
  metric (sum of absolute gradients — dropped entirely, wasn't principled).

### Raw action distribution (new — the fun one)
A rolling histogram of the last N *clamped* actions actually sent to the
environment (N ~100–200, ring buffer).

- Bucket `[-1, 1]` into a fixed number of bins (e.g. 21 bins, width 0.1).
- Redraw as a small bar chart in the sidebar, rebuilt each episode (or every
  K steps if we want it live mid-episode).
- Purpose: this is the most direct possible view into *what the policy is
  actually doing*, independent of what the Critic thinks or what the loss
  numbers say. A healthy learning policy should visibly narrow and shift this
  distribution over training; a stuck one will show it planted on a single
  bin (over-exploited) or flat across all bins (never converging /
  effectively still random).

### Dropped from the old DQN panel entirely
- Epsilon / "Chaos %" — doesn't exist here, exploration is continuous std, not
  an external schedule.
- Per-action Q-value bar chart — no discrete action set anymore to bar-chart.
- "MSE Loss" framed as a single DQN-style number — replaced by the
  Critic-loss / Advantage-magnitude pair above, which map onto what's
  actually being optimized here.

---



# REINFORCE Architecture — Diagnostic Report 

**Conclusion:** Vanilla REINFORCE is mathematically insufficient for the Double Pendulum environment due to inherent instability. Transition to PPO required.

---

## 1. Observed Symptoms (The Collapse)

During the live training of the `ReinforceAgent` on the `DoublePendulumTask`, the agent consistently exhibited a rapid and fatal mathematical collapse. The newly implemented DOM UI dashboard successfully captured the exact mechanics of this failure in real-time.

### Symptom A: Action Distribution "Wall" Slam
* **Observation:** The action histogram began as a healthy, spread-out Bell curve. After a random fluctuation where the agent survived slightly longer, the histogram violently collapsed.
* **Result:** All actions became permanently glued to either `-1.0` or `+1.0`. The agent effectively held a single direction key down forever.

### Symptom B: The -1.581 Entropy Floor
* **Observation:** The Policy Entropy metric plummeted and locked exactly at `-1.581`.
* **Mathematical Proof:** In our code, we hardcoded `LOG_STD_MIN = -3.0` to prevent division by zero. This clamps the standard deviation ($\sigma$) at `Math.exp(-3.0) ≈ 0.0497`.
* **Calculation:** The entropy of a continuous Gaussian is `0.5 * ln(2 * π * e * σ²)`. Plugging in our floor: `0.5 * ln(2 * π * e * (0.0497)²) = -1.581`.
* **Meaning:** The network became so violently overconfident in a single action that it tried to shrink its uncertainty ($\sigma$) to absolute zero, hitting our mathematical safety net and getting permanently stuck.

### Symptom C: Score Flatline
* **Observation:** The moving average score flatlined near `0`. 
* **Meaning:** The agent was surviving for exactly ~20 frames (earning +20 points) before immediately failing the boundary/angle constraints (taking the -20 penalty). $20 - 20 = 0$.

---

## 2. Root Cause Analysis: Catastrophic Overconfidence

The collapse is a textbook example of the fundamental flaw in Vanilla Policy Gradients when applied to chaotic, continuous control environments.

1. **The Spark:** By pure luck, the Actor executes a random twitch that keeps the pendulum alive for a few extra milliseconds.
2. **The Critique:** The Critic compares this survival to its baseline and issues a massive positive **Advantage** score.
3. **The Exploding Gradient:** Because REINFORCE lacks a "speed limit," the Actor receives this massive Advantage and immediately alters its entire neural weight structure to favor that single twitch.
4. **The Coma:** The Actor pushes its Mean ($\mu$) to infinity (which gets clamped to `1` or `-1` by the `tanh` activation) and its Standard Deviation ($\sigma$) to zero. It becomes completely rigid, deaf, and blind to new data.

---

## 3. The Resolution: Transition to PPO

Tuning hyperparameters (like lowering the Actor's learning rate to `0.00005`) merely delays this inevitable collapse. The Double Pendulum is too chaotic; an overconfident "jerk" will eventually happen.

To solve this, the architecture must be upgraded to **Proximal Policy Optimization (PPO)**. 

### PPO Implementation Requirements:
1. **The Trust Region (Clipping):** We must abandon the raw REINFORCE loss function. The new Actor loss function must calculate the probability ratio between the *old* policy and the *new* policy.
2. **The 20% Speed Limit:** We must apply PPO's clipping mechanism (`clip(ratio, 0.8, 1.2)`) to ensure the Actor can never change its action probability by more than 20% in a single update, regardless of how large the Advantage score is.
3. **On-Policy Batches:** The training loop will shift from "learn at the end of every episode" to "play for N steps, learn on that exact batch, and immediately delete it."

By enforcing monotonic, constrained improvement, PPO will prevent the entropy from collapsing and allow the agent to safely discover the microscopic adjustments required to balance the double pendulum.
