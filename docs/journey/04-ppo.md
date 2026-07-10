# The PPO Performance Journey — from "it stutters" to knowing exactly why

This one didn't start as an algorithm problem. The PPO agent trained fine; the
page just stuttered. What follows is the investigation, in order, with the
actual numbers at each step rather than guesses.

## 0. The starting architecture

`PPOTrainer.tick()` ran on the main thread, in a `while (performance.now() -
start < timeBudgetMs)` loop, `setTimeout(0)`-rescheduling itself — the same
shape DQN and REINFORCE already used. The difference was what shared that
thread with it: a 60fps `requestAnimationFrame` render loop, fighting the
trainer for CPU time every frame.

## 1. Move training off the main thread

**Fix:** the entire training loop moved into a dedicated Web Worker
(`ppo.worker.ts`). `PPOTrainer`, `PPOAgent`, `DoublePendulumTask` all run
there unmodified — none of them ever touched the DOM to begin with. The main
thread's job shrank to: draw whatever physics snapshot the worker last
posted, and read whatever metrics it last reported.

- `CanvasRenderer.render()` now accepts a structural `RenderableEnvironment`
  (plain data) as well as a live `Environment`, since `postMessage`'s
  structured-clone can't send class instances with their prototypes intact.
- `PPODOMUI` reads from a `PPOTrainerLike` interface instead of a concrete
  `PPOTrainer` — a plain-data mirror object on the main thread, updated via
  `Object.assign` from `'metrics'` messages, stands in for the live trainer
  it used to read directly.
- The render toggle now also tells the *worker* to stop serializing frames,
  not just tells the main thread to stop drawing them.

This alone should have fixed it. It didn't — the animation still stuttered,
just as visibly.

## 2. A real bug, found along the way: GAE bootstrap value

While touching this code: `PPOAgent.learn()` always assumed `lastValue = 0`
for the state after the last stored transition — correct only if the episode
actually *ended* there. But the rollout buffer fills on a fixed `HORIZON`
(2048 steps), so most cuts are mid-episode truncations, not falls. Treating
every truncation as a terminal state systematically biased the advantage
estimates downward.

**Fix:** `PPOTrainer` now captures `lastNextState` right before any
episode-reset overwrites `currentState`, and passes
`agent.getValue(lastNextState)` into `learn()` as a real bootstrap value. The
existing `nextNonTerminal` gate in the GAE loop already zeroes it out
correctly on genuine terminal steps, so this is safe either way — no
"was it actually done" branch needed. Standard PPO practice; this
implementation was just missing it.

## 3. Why it was still stuttering: the worker was blocking itself

Moving training to a worker didn't remove the freeze — it just moved *which*
thread froze. `PPOAgent.learn()` runs `epochs × length` samples
(3 × 2048 = 6144) through two networks, fully synchronously, with zero
yields. While that ran, the worker's own `setInterval` timers (frame @30fps,
metrics @10Hz) couldn't fire either — so the physics preview would freeze in
place for the entire duration, then jump.

**Fix:** `learn()` became `async`, yielding to the event loop periodically
during the epoch loop (originally every 512 samples, later reworked — see
§5). This let the worker's own timers interleave with training instead of
being shut out for a solid block.

`PPOTrainer` needed restructuring to support this properly: `doOneStep()` now
only *flags* `needsTraining = true` when the horizon is hit, instead of
calling `train()` inline mid-loop; `tick()`'s physics while-loop breaks early
on that flag and `await`s `train()` from a clean point before rescheduling
itself.

## 4. Instrumentation instead of more guessing

Before optimizing further, we added a live "Performance" panel to the DOM UI
instead of continuing to speculate:

- **Steps/sec** — physics throughput, computed by the worker from
  `totalSteps` deltas.
- **Last / Avg Train Time** — wall-clock duration of `learn()`, tracked with
  an exponential moving average.
- **Worker Max Stall (1s)** — the worst gap between the worker's own 33ms
  heartbeat timer firing, over the last second. If the worker's event loop is
  genuinely blocked, *every* timer on that thread delays by the same amount —
  so this is a direct read on "was the worker frozen," independent of cause.
- **Render Max Stall (1s)** — same idea on the main thread's
  `requestAnimationFrame` cadence. Since no training math runs there anymore,
  it should track ~16ms regardless of what the worker does; if it spikes
  independently, the bottleneck moved to the main thread instead.

**What it showed:** Render stall stayed ~8ms (smooth) the whole time — the
thread separation from §1 was working correctly. Worker stall hitched up to
~190ms, and **average train time was ~1150ms**. The chunked yielding from §3
was doing its job (breaking one long block into ~190ms pieces instead of one
1150ms freeze) — but the sim still visibly paused for over a second every
~2048 steps, because physics genuinely can't advance while `learn()` is
running (correct, deliberate behavior for on-policy PPO — you don't collect
new rollout under a stale policy mid-update). The chunking made the freeze
*interruptible*; it didn't make it *shorter*. That required actually reducing
how long `learn()` takes.

## 5. Cutting real cost, benchmarked at each step

From here on, every change was benchmarked directly (`tsx` script running
`agent.learn()` against a realistic 2048-sample buffer, JIT-warmed, averaged
over multiple runs) rather than assumed:

| Step | Change | `learn()` time |
|---|---|---|
| — | baseline | ~1400ms |
| A | **Buffer reuse**: `Matrix`/`DenseLayer`/`NeuralNetwork` allocated a fresh `Float32Array`-backed scratch matrix on *every* forward/backward call — ~20-30 tiny allocations × 6144 samples per `learn()`, flooding the GC. Every layer now preallocates its scratch matrices once and reuses them. Shared by DQN and REINFORCE too (same lib classes), verified unaffected via smoke tests. | (rolled into row below) |
| B | **Drop the redundant actor forward.** `learn()` called `decodeActorOutput()` (a full actor forward pass) to get mean/std, then `trainWithGradient()` immediately forwarded the *same* input again before backpropagating. Added `NeuralNetwork.backwardWithGradient()`, which skips straight to backward using the cache the first forward already populated. | ~1200ms |
| C | **Matmul inner-loop micro-opt.** `Matrix.dotInto()`/`transposeInto()` were recomputing `i * cols` and `k * cols` index arithmetic on every iteration of the hot triple loop; hoisted out and replaced repeated multiplication with an accumulator. | (included in B's measurement) |
| D | **Mini-batching.** The big one. PPO here was doing single-sample SGD — 6144 individual one-column matmuls per `learn()` call, the worst possible shape for a JS engine (thousands of tiny function calls instead of a few wide ones). Reworked the epoch loop to process 64-sample mini-batches instead: gather states into an `(inputSize, 64)` matrix, one batched forward + backward per minibatch. This is also just... standard PPO — Stable-Baselines3's default `batch_size` is 64. We were doing a simplification, not the canonical algorithm. | ~680ms |
| E | **Shrink the network.** Hidden width dropped 64 → 32 in both the Actor and Critic. The 64×64 middle Dense layer was ~85% of the network's forward/backward FLOPs (4096 weights vs. 512 + 128 for the input/output layers combined) — by far the dominant cost, and unlike A-D this one actually reduces total floating-point work instead of just doing the same work more efficiently. 32 hidden units is still generous for this task's 8-dimensional state space. | **~363ms** |

Step D needed real new plumbing, scoped deliberately to avoid touching
DQN/REINFORCE's existing per-sample paths:
- `Matrix.addBroadcastColumn()` / `sumRowsInto()` — batched bias add/gradient
  need to broadcast a `(rows, 1)` bias across every column, and sum a
  `(rows, batchWidth)` gradient back down to one column.
- `Layer` interface gained *optional* `forwardBatch?`/`backwardBatch?`.
  `DenseLayer` implements them (with their own lazily-sized scratch buffers,
  independent of the single-sample ones `act()`/`getValue()` still use every
  physics step). `ReLULayer` didn't need a batched variant at all — it's
  already elementwise/shape-agnostic, and its existing lazy-resize scratch
  handles a batch matrix with no code change.
- `NeuralNetwork.predictBatch()` / `backwardBatchWithGradient()` route to the
  batched methods when a layer has them, falling back to the single-sample
  ones otherwise — so nothing about `DQNAgent`/`ReinforceAgent`'s training
  changed.
- The PPO-clipping math itself (ratio, clip, log-prob derivatives) stayed a
  per-sample scalar loop *inside* each minibatch — there's no useful way to
  express that part as a matrix op. What got batched was the two expensive
  parts: the forward pass producing mean/std for the whole minibatch at
  once, and the backward pass applying all 64 samples' gradients in a single
  averaged update instead of 64 sequential ones.

Verified via a smoke test that deliberately used a buffer size *not* a
multiple of 64 (2048 + 37), to exercise the ragged-last-minibatch path, plus
confirmed `act()`/`getValue()` still behave correctly when interleaved with
batched `learn()` calls, plus DQN/REINFORCE smoke tests to confirm the shared
`lib/` classes weren't broken for them. Step E re-ran the same smoke test
against the smaller 32-wide network — no NaNs, `act()`/save-load still fine.

## Where this stands

**~3.85x faster than the original** (1400ms → 363ms per `learn()` call), on
top of the worker migration that was always going to be necessary regardless
of train speed. Two of these changes are more than just speed wins:

- **Mini-batching (D)** is also a **correctness/convergence improvement**,
  not just a speed one — 64-sample averaged updates are the standard PPO
  recipe; 2048 individual single-sample updates per epoch was noisier than
  intended.
- **Shrinking the network (E)** is a genuine trade-off, not a free win —
  worth watching whether score/survival plateaus meaningfully lower than
  before with only 32 hidden units.

Combined with mini-batching dropping weight updates per epoch from 2048 to
32, it's worth keeping an eye on KL divergence and clip fraction in the
Performance panel after these changes — if training looks stalled, the
learning rate may want retuning upward to compensate for fewer, larger-batch
steps on a smaller network.

**Why steps A-D only bought ~2x, and E bought another ~2x on top:**
everything through step C was an *overhead* problem — allocation churn,
redundant work, function-call count. Mini-batching (D) attacks the same
category (fewer, wider calls instead of many tiny ones) but doesn't change
the total floating-point work: same network, same number of multiply-adds
either way. Step E is the first change in this list that actually reduces
FLOPs rather than just doing the same FLOPs more efficiently — which is why
it moved the needle by roughly as much as A-D combined. The instrumentation
panel added in §4 stays in the UI specifically so the next slowdown —
whatever it turns out to be — gets diagnosed with real numbers again instead
of another round of guessing.

## If more speed is needed later

The remaining levers, roughly in order of effort:
1. **Shrink the workload further** — fewer epochs, smaller `HORIZON`, or an
   even narrower hidden layer. Free, but trades away sample
   efficiency/convergence quality further.
2. **WASM with SIMD** for the matmul core — same algorithm, a runtime that
   can actually vectorize the inner loop.
3. **GPU-backed ops** (WebGL/WebGPU, or a real tensor library) — the biggest
   possible win, since this workload (many small batched matmuls) is exactly
   what GPUs are good at. Also the largest engineering lift of the three.

---

# Part 2 — From "it trains" to "it actually balances" (30s → 42s → past a minute)

Performance work made `learn()` fast. It didn't make the *policy* good — that
turned out to need a separate pass, on the actual RL side rather than the
engineering side. Same rule as Part 1: real numbers over guesses at every
step, verified before trusted.

## 6. The environment was fighting the agent

Three changes to `DoublePendulumTask`, prompted by watching the live "Actor
Distribution" panel (added specifically to make this kind of thing visible
instead of invisible):

- **Thrust ceiling raised 1500N → 5000N.** The panel showed the actor pinned
  at `±1500N` — its mean output saturating at the actuator's hard cap — which
  is a direct visual tell that the policy *wants* more control authority than
  the environment allows. No amount of training fixes that; the network only
  ever learns "what fraction of the fixed max," never a different max.
- **Fall condition relaxed from a fixed angle to a physical one.** Was
  `|angle| > 0.8 rad` (~46°) — an arbitrary cliff. Now: episode ends when the
  tip (`pole2`) actually swings down to or past the cart's own height
  (`pole2.position.y >= cart.position.y`). More realistic, and critically,
  much more forgiving — a wide, fast swing that would previously have been
  instant death now has room to recover, which is what the agent needs to
  actually *learn* recovery instead of only ever seeing near-vertical states.
- **`MAX_EPISODE_STEPS` raised 2000 → 4000** (~64s), to give recovery arcs
  under the new, more forgiving fall condition room to actually finish
  playing out. Left `HORIZON` (2048) untouched deliberately — an episode
  outliving a training pause was already handled correctly by the GAE
  bootstrap fix from Part 1 §2, so there was no need to inflate training cost
  just to "keep pace."

## 7. A real bug, caught by testing on an easier problem first

Before trusting any of this on the double pendulum, we built a second,
much simpler scene — `SinglePendulumTask` (renamed from the pre-existing,
unused `CartPoleTask`) — specifically to sanity-check the pipeline somewhere
convergence should be fast and obvious. Good thing: it wasn't.

A headless training run (`tsx` script, no browser, same `PPOAgent`/
`PPORolloutBuffer` calls `PPOTrainer` makes internally) showed `kl≈0.0000`
and `clip≈0%` for **over 1000 episodes straight** — the actor was not
learning, at all, just slowly enough on the double pendulum's noisier reward
signal to look like "hard task" instead of "broken."

**Root cause:** the mini-batching work in Part 1 §5 correctly averaged the
gradient across each 64-sample batch — standard practice — but nothing
compensated the learning rate for going from 6144 full-strength single-sample
updates to 96 averaged ones. Effective step size per unit of data had quietly
dropped far more than "slower."

**Fix, verified empirically before committing to it:** scaled
`actorLearningRate`/`criticLearningRate` up 16x (`0.0005→0.008`,
`0.001→0.016`) — a conservative step short of the full "linear scaling rule"
(which would suggest the full 64x, matching batch size). Re-ran the same
headless harness: score climbed from ~48 to a peak of ~235 over 60 rounds,
`clip` moved into a healthy 2-15% band, `kl` showed real, growing policy
movement instead of a flat zero. This is the fix that actually mattered —
everything in §6 gave the agent room to learn; this is what made it capable
of learning at all again after §5's batching change.

## 8. The result

With §6 and §7 together: **survival time passed 60 seconds**, climbing from
"stuck around 2s" before this pass, through 31s, 42s, and past a minute — at
only ~8000 episodes in. Confirmed as a real result, not a fluke: `HORIZON`
and the training pipeline were untouched by any of this, `MAX_EPISODE_STEPS`
was raised specifically to stop capping legitimately long runs, and every
change was verified independently (learning-rate fix on the single pendulum,
then observed working on the double pendulum) before being trusted.

## 9. New capability, not just a faster/better agent

Two features added once there was finally something worth watching closely:

- **Live Demo mode.** A second physics environment now runs inside the
  Worker continuously, in real time (~60Hz, decoupled from training's
  run-as-fast-as-possible loop), driven by `PPOAgent.actGreedy()` — the
  policy's mean output with zero sampling noise. Toggling it switches the
  canvas between the noisy training rollout (what the agent is *trying*,
  exploration included) and this clean deterministic view (what it actually
  *thinks* is best). Tracks its own separate survival stats
  (`evalSurvivalSeconds` / `evalMaxSurvivalSeconds`), since training's score
  includes exploration noise degrading it — this is a truer read on "how good
  is the policy right now."
- **Save / Load.** `PPOAgent.toJSON()`/`loadJSON()` already existed but had
  no UI. Wired the existing Worker `'save'`/`'load'` message handlers up to
  actual buttons: Save triggers a browser download of the actor+critic
  weights as JSON; Load reads a file back in. Both round-trip through the
  Worker via `postMessage`, same as everything else crossing that boundary.

Both scenes (`ppo.html` double pendulum, `single-pendulum.html` single
pendulum) got all of this, since `PPODOMUI`/`CanvasRenderer`/`PPOTrainerLike`
were already fully task-agnostic — extending one meant extending both with
almost no extra design work, just wiring.
