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
| D | **Mini-batching.** The big one. PPO here was doing single-sample SGD — 6144 individual one-column matmuls per `learn()` call, the worst possible shape for a JS engine (thousands of tiny function calls instead of a few wide ones). Reworked the epoch loop to process 64-sample mini-batches instead: gather states into an `(inputSize, 64)` matrix, one batched forward + backward per minibatch. This is also just... standard PPO — Stable-Baselines3's default `batch_size` is 64. We were doing a simplification, not the canonical algorithm. | **~680ms** |

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
`lib/` classes weren't broken for them.

## Where this stands

~2x faster than the original (1400ms → 680ms per `learn()` call), on top of
the worker migration that was always going to be necessary regardless of
train speed. The mini-batching change also happens to be a **correctness/
convergence improvement**, not just a speed one — 64-sample averaged updates
are the standard PPO recipe; 2048 individual single-sample updates per epoch
was noisier than intended. Worth watching KL divergence and clip fraction
after this change, since the number of weight updates per epoch dropped from
2048 to 32 — if training looks stalled, the learning rate may want retuning
upward to compensate for fewer, larger-batch steps.

**Why it's 2x and not 10x, honestly:** everything up through step C was an
*overhead* problem — allocation churn, redundant work, function-call count.
Mini-batching (step D) attacks the same category of problem (fewer, wider
calls instead of many tiny ones) but doesn't change the total floating-point
work: the network still does the same number of multiply-adds either way.
We're now compute-bound on genuine arithmetic in a naive O(n³) JS matmul
loop, not overhead-bound. The instrumentation panel added in §4 stays in the
UI specifically so the next slowdown — whatever it turns out to be — gets
diagnosed with real numbers again instead of another round of guessing.

## If more speed is needed later

The remaining levers, roughly in order of effort:
1. **Shrink the workload** — smaller hidden layers, fewer epochs, smaller
   `HORIZON`. Free, but trades away sample efficiency/convergence quality.
2. **WASM with SIMD** for the matmul core — same algorithm, a runtime that
   can actually vectorize the inner loop.
3. **GPU-backed ops** (WebGL/WebGPU, or a real tensor library) — the biggest
   possible win, since this workload (many small batched matmuls) is exactly
   what GPUs are good at. Also the largest engineering lift of the three.
