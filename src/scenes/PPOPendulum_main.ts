import { CanvasRenderer } from '../renderer/CanvasRenderer';
import type { RenderableEnvironment } from '../renderer/CanvasRenderer';
import { PPODOMUI } from '../ui/PPODOMUI';
import type { PPOTrainerLike } from '../training/PPOTrainer';

// ==========================================
// 1. DOM Setup — unchanged
// ==========================================
const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
if (!canvas) throw new Error("Could not find sim-canvas in DOM");

const renderToggle = document.getElementById('ui-render-toggle') as HTMLInputElement;
if (!renderToggle) console.warn("Could not find ui-render-toggle in DOM, rendering will always be active.");

const renderer = new CanvasRenderer(canvas);

// ==========================================
// 2. Trainer Mirror — a plain data holder that PPODOMUI reads from exactly
// like it used to read from a live PPOTrainer. The only difference is WHO
// updates these fields: instead of a local tick() loop mutating them
// directly, incoming 'metrics' messages from the Worker do.
// ==========================================
const trainerMirror: PPOTrainerLike = {
    episode: 1,
    score: 0,
    stepsThisEpisode: 0,
    maxSurvivalTime: 0,
    currentCriticLoss: 0,
    currentAdvantage: 0,
    currentClipFraction: 0,
    currentKlDivergence: 0,
    actionHistory: [],
    scoreHistory: [],
    survivalTimeHistory: [],
    totalSteps: 0,
    lastTrainMs: 0,
    avgTrainMs: 0,
    stepsPerSecond: 0,
    maxWorkerFrameGapMs: 0,
    mainThreadFrameGapMs: 0,
};

const ui = new PPODOMUI(trainerMirror);

// ==========================================
// 3. The Worker — this is where ALL training now happens. Vite's
// `new URL(..., import.meta.url)` pattern below is what tells it to bundle
// ppo.worker.ts as its own separate module for the worker thread.
// ==========================================
const worker = new Worker(new URL('../training/ppo.worker.ts', import.meta.url), { type: 'module' });

let latestFrame: RenderableEnvironment | null = null;

worker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (!msg || typeof msg !== 'object') return;

    switch (msg.type) {
        case 'metrics': {
            // Copy fields in place rather than replacing the object, so the
            // reference PPODOMUI was constructed with stays valid.
            Object.assign(trainerMirror, msg.payload);
            break;
        }
        case 'frame': {
            latestFrame = msg.payload as RenderableEnvironment;
            break;
        }
        // 'brainData' / 'loaded' are available for a future Save/Load UI on
        // this scene (PPOAgent now supports it) — no handler needed until
        // that UI exists.
    }
};

// ==========================================
// 4. Render toggle — now ALSO tells the Worker to stop serializing frames,
// not just tells the main thread to stop drawing them. Checking the box
// again resumes both.
// ==========================================
if (renderToggle) {
    renderToggle.addEventListener('change', () => {
        worker.postMessage({ type: 'setRenderEnabled', enabled: renderToggle.checked });
    });
}

// ==========================================
// 5. The main thread's only remaining loop: draw whatever the Worker most
// recently sent. No physics, no gradients, no agent calls happen here at all.
//
// It also tracks its OWN stall gap — the time between consecutive rAF calls —
// independent of anything the Worker reports. If training moved fully off
// this thread, this number should track the monitor refresh rate (~16.6ms)
// no matter what the Worker is doing. If it spikes in lockstep with
// maxWorkerFrameGapMs, something is still coupling the two threads. If it
// spikes on its OWN, the bottleneck is main-thread rendering/DOM work, not
// training at all.
// ==========================================
let lastRenderAt = performance.now();
let maxMainThreadFrameGapMs = 0;

setInterval(() => {
    trainerMirror.mainThreadFrameGapMs = maxMainThreadFrameGapMs;
    maxMainThreadFrameGapMs = 0; // Same "worst stall in the last window" reporting as the Worker uses.
}, 1000);

function renderLoop() {
    const now = performance.now();
    const gap = now - lastRenderAt;
    if (gap > maxMainThreadFrameGapMs) maxMainThreadFrameGapMs = gap;
    lastRenderAt = now;

    ui.update();

    if ((!renderToggle || renderToggle.checked) && latestFrame) {
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
        renderer.render(latestFrame);
    }

    requestAnimationFrame(renderLoop);
}

renderLoop();