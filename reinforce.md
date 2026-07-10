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

## Open questions before implementation

- Bin count/width for the action histogram — 21 bins @ 0.1 width assumed
  above, easy to change.
- Whether the action histogram rebuilds every episode or has its own shorter
  rolling window independent of episode boundaries (matters more once
  episodes get long).
- Sidebar chart sizing/layout once everything moves off-canvas — will sketch
  once the metric list above is confirmed.