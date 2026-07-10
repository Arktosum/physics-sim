// src/training/workers/workerHarness.ts
//
// Shared plumbing for every training Worker in this project. What's actually
// duplicated between algorithms is never the training logic (Q-learning,
// DQN, REINFORCE, and PPO all learn in genuinely different ways) — it's the
// scaffolding around it: throttled metrics/frame postMessage, worker-stall
// tracking, save/load message handling, and the live-eval loop shape. This
// file is that scaffolding, parameterized per algorithm by the options below.
//
// Contract for `tick()`: do ONE burst of training work and return (or
// resolve, if async) — do NOT self-reschedule with setTimeout. This harness
// owns the "keep going forever" loop, so every worker gets it for free and
// there's no risk of double-scheduling.

export interface TrainingWorkerOptions {
    /** Runs one burst of training work (e.g. "however many steps fit in a time budget, then maybe train"). */
    tick(): void | Promise<void>;

    /** Builds the metrics payload sent to the main thread. stepsPerSecond and maxWorkerFrameGapMs are added automatically. */
    buildMetrics(): Record<string, unknown>;

    /** Builds the physics frame payload sent to the main thread, for whichever environment previewMode currently selects. */
    buildFrame(previewMode: 'training' | 'live'): unknown;

    /** One greedy step of the live-eval task (separate environment, driven by the current policy's deterministic action, never touches training data). */
    runEvalStep(): void;

    /** Cumulative step counter, used to compute stepsPerSecond. */
    getTotalSteps(): number;

    /** Current episode number, used to name the file when 'save' is requested. */
    getEpisode(): number;

    /** Agent (de)serialization, wired straight to Save/Load buttons on the main thread. */
    toJSON(): string;
    loadJSON(json: string): void;

    metricsIntervalMs?: number; // default 100ms (~10Hz) — plenty for numbers and charts
    frameIntervalMs?: number;   // default 33ms (~30fps) — plenty for a physics preview
    evalIntervalMs?: number;    // default 16ms (~60Hz) — real-time playback pace for the live demo
}

export function runTrainingWorker(opts: TrainingWorkerOptions): void {
    const METRICS_INTERVAL_MS = opts.metricsIntervalMs ?? 100;
    const FRAME_INTERVAL_MS = opts.frameIntervalMs ?? 33;
    const EVAL_INTERVAL_MS = opts.evalIntervalMs ?? 16;

    let renderEnabled = true;
    // defaults to the noise-free policy view: if training is fast enough to
    // watch at all, the blurry exploration rollout is a worse first look
    // than what the agent actually believes is best
    let previewMode: 'training' | 'live' = 'live';

    // Perf diagnostics — maxFrameGapMs is measured on the frame timer itself
    // (whether or not a frame is actually sent while rendering is toggled
    // off): if the worker's event loop is genuinely blocked, every one of
    // its own timers delays by the same amount, so this doubles as "how
    // stalled was this thread" regardless of the render toggle. Reset every
    // time metrics are posted, so the UI shows "worst stall in the last
    // ~100ms window," not one stale sample.
    let lastFrameTickAt = performance.now();
    let maxFrameGapMs = 0;
    let lastMetricsAt = performance.now();
    let lastMetricsSteps = opts.getTotalSteps();

    self.onmessage = (e: MessageEvent) => {
        const msg = e.data;
        if (!msg || typeof msg !== 'object') return;

        switch (msg.type) {
            case 'setRenderEnabled':
                // Uncheck "Render Physics" for max speed: stop bothering to
                // serialize frame data at all, not just stop drawing it.
                renderEnabled = !!msg.enabled;
                break;

            case 'setPreviewMode':
                previewMode = msg.mode === 'live' ? 'live' : 'training';
                break;

            case 'save':
                postMessage({ type: 'brainData', payload: { json: opts.toJSON(), episode: opts.getEpisode() } });
                break;

            case 'load':
                opts.loadJSON(msg.json);
                postMessage({ type: 'loaded' });
                break;
        }
    };

    setInterval(() => {
        const now = performance.now();
        const elapsedMs = now - lastMetricsAt;
        const totalSteps = opts.getTotalSteps();
        const stepsDelta = totalSteps - lastMetricsSteps;
        const stepsPerSecond = elapsedMs > 0 ? (stepsDelta / elapsedMs) * 1000 : 0;
        lastMetricsAt = now;
        lastMetricsSteps = totalSteps;

        postMessage({
            type: 'metrics',
            payload: { ...opts.buildMetrics(), stepsPerSecond, maxWorkerFrameGapMs: maxFrameGapMs },
        });
        maxFrameGapMs = 0; // Start a fresh stall-detection window for the next report.
    }, METRICS_INTERVAL_MS);

    setInterval(() => {
        const now = performance.now();
        const gap = now - lastFrameTickAt;
        if (gap > maxFrameGapMs) maxFrameGapMs = gap;
        lastFrameTickAt = now;

        if (!renderEnabled) return;
        postMessage({ type: 'frame', payload: opts.buildFrame(previewMode) });
    }, FRAME_INTERVAL_MS);

    setInterval(() => {
        opts.runEvalStep();
    }, EVAL_INTERVAL_MS);

    // Ignite. This never stops — the harness owns the reschedule loop so
    // individual tick() implementations don't each need their own.
    (async function loop() {
        await opts.tick();
        setTimeout(loop, 0);
    })();
}
