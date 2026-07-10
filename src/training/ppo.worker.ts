// src/training/ppo.worker.ts
//
// The entire training loop now lives here, on its own thread. Nothing in
// this file — or in PPOAgent / PPOTrainer / DoublePendulumTask, which it
// imports unmodified — ever touches the DOM. That's exactly what makes it
// safe to let trainer.tick() run flat-out: there is no render loop on this
// thread for it to starve.
//
// Communication with the main thread is one-way except for a single control
// message (render on/off): this thread periodically POSTS small snapshots
// (metrics, and optionally a physics frame) out; it doesn't wait for
// anything back before continuing to train.

import { PPOAgent } from '../engine/PPOAgent';
import { PPOTrainer } from './PPOTrainer';
import { DoublePendulumTask } from '../sim/DoublePendulumTask';

// ==========================================
// Setup — identical shape to what PPOPendulum_main.ts used to do directly.
// ==========================================
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const TRACK_HEIGHT = CANVAS_HEIGHT - 150;
const FIXED_DT = 0.016;

const task = new DoublePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT);
const initialState = task.reset();
const agent = new PPOAgent(initialState.length);

// Training is no longer racing a 60fps render loop for the MAIN thread's
// attention, so it might seem like this should be as large as possible. It
// can't be: this Worker is still single-threaded internally, and its own
// setInterval callbacks below (posting frame/metrics) share that same
// thread with trainer.tick(). A big budget here just moves the bottleneck
// — instead of the main thread starving training, training starves its own
// outbound messages, and the frame stream arrives in stale, ~250ms-late
// chunks instead of a smooth ~33ms cadence. Keeping this small costs a
// little raw throughput (more yields = more scheduling overhead) but keeps
// the worker responsive to its own timers, which is what actually mattered.
const TRAIN_TIME_BUDGET_MS = 20;
const trainer = new PPOTrainer(agent, task, TRAIN_TIME_BUDGET_MS);

// ==========================================
// Outbound snapshots — throttled independently of training speed.
// Training might run thousands of steps/sec; nothing downstream needs
// updates anywhere near that often.
// ==========================================
const METRICS_INTERVAL_MS = 100;  // ~10Hz — plenty for numbers and charts
const FRAME_INTERVAL_MS = 33;     // ~30fps — plenty for a physics preview

let renderEnabled = true;

function buildFrame() {
    const env = task.env;
    return {
        points: env.points.map(p => ({
            position: { x: p.position.x, y: p.position.y },
            mass: p.mass,
            isPinned: p.isPinned,
        })),
        constraints: env.constraints
            .map((c: any) => {
                if ('p1' in c && 'p2' in c) {
                    return {
                        p1: { position: { x: c.p1.position.x, y: c.p1.position.y } },
                        p2: { position: { x: c.p2.position.x, y: c.p2.position.y } },
                    };
                }
                if ('lockedY' in c) {
                    return { lockedY: c.lockedY };
                }
                return null;
            })
            .filter((c: unknown) => c !== null),
    };
}

function buildMetrics() {
    return {
        episode: trainer.episode,
        score: trainer.score,
        stepsThisEpisode: trainer.stepsThisEpisode,
        maxSurvivalTime: trainer.maxSurvivalTime,
        currentCriticLoss: trainer.currentCriticLoss,
        currentAdvantage: trainer.currentAdvantage,
        currentClipFraction: trainer.currentClipFraction,
        currentKlDivergence: trainer.currentKlDivergence,
        actionHistory: trainer.actionHistory,
        scoreHistory: trainer.scoreHistory,
        survivalTimeHistory: trainer.survivalTimeHistory,
        totalSteps: trainer.totalSteps,
        lastTrainMs: trainer.lastTrainMs,
        avgTrainMs: trainer.avgTrainMs,
    };
}

// ==========================================
// Perf diagnostics — lets us actually SEE where time goes instead of guessing.
//
// maxFrameGapMs is measured on the FRAME_INTERVAL_MS timer itself (whether or
// not a frame is actually sent while rendering is toggled off): if the worker's
// event loop is genuinely blocked by train(), every one of its own timers gets
// delayed by the same amount, so this doubles as "how stalled was this thread"
// regardless of the render toggle. It's reset to 0 every time metrics are
// posted, so what you see in the UI is "the worst stall in the last ~100ms
// window", not a single stale sample.
// ==========================================
let lastFrameTickAt = 0;
let maxFrameGapMs = 0;
let lastMetricsAt = 0;
let lastMetricsSteps = 0;

// ==========================================
// Inbound control messages
// ==========================================
self.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (!msg || typeof msg !== 'object') return;

    switch (msg.type) {
        case 'setRenderEnabled':
            // When the person unchecks "Render Physics" for max speed, tell
            // THIS thread to stop bothering to serialize frame data too —
            // not just skip drawing it on the main thread. No point paying
            // the postMessage cost for something nobody's looking at.
            renderEnabled = !!msg.enabled;
            break;

        case 'save':
            postMessage({ type: 'brainData', payload: { json: agent.toJSON(), episode: trainer.episode } });
            break;

        case 'load':
            agent.loadJSON(msg.json);
            postMessage({ type: 'loaded' });
            break;
    }
};

lastMetricsAt = performance.now();

setInterval(() => {
    const now = performance.now();
    const elapsedMs = now - lastMetricsAt;
    const stepsDelta = trainer.totalSteps - lastMetricsSteps;
    const stepsPerSecond = elapsedMs > 0 ? (stepsDelta / elapsedMs) * 1000 : 0;
    lastMetricsAt = now;
    lastMetricsSteps = trainer.totalSteps;

    postMessage({
        type: 'metrics',
        payload: { ...buildMetrics(), stepsPerSecond, maxWorkerFrameGapMs: maxFrameGapMs },
    });
    maxFrameGapMs = 0; // Start a fresh stall-detection window for the next report.
}, METRICS_INTERVAL_MS);

lastFrameTickAt = performance.now();

setInterval(() => {
    const now = performance.now();
    const gap = now - lastFrameTickAt;
    if (gap > maxFrameGapMs) maxFrameGapMs = gap;
    lastFrameTickAt = now;

    if (!renderEnabled) return;
    postMessage({ type: 'frame', payload: buildFrame() });
}, FRAME_INTERVAL_MS);

// ==========================================
// Ignite. This never stops.
// ==========================================
trainer.tick();