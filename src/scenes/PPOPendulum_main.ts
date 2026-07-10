import { DoublePendulumTask } from '../sim/DoublePendulumTask';
import { PPOAgent } from '../engine/PPOAgent';
import { PPOTrainer } from '../training/PPOTrainer';
import { CanvasRenderer } from '../renderer/CanvasRenderer';
import { PPODOMUI } from '../ui/PPODOMUI';

// ==========================================
// 1. DOM Setup
// ==========================================
const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
if (!canvas) throw new Error("Could not find sim-canvas in DOM");

const renderToggle = document.getElementById('ui-render-toggle') as HTMLInputElement;
if (!renderToggle) console.warn("Could not find ui-render-toggle in DOM, rendering will always be active.");

// ==========================================
// 2. Physics & Task Setup
// ==========================================
// Your DoublePendulumTask handles creating the Environment internally.
// constructor(canvasWidth, canvasHeight, trackHeight, fixedDt)
const task = new DoublePendulumTask(canvas.width, canvas.height, canvas.height - 150, 0.016);

// We extract the environment FROM the task so the renderer knows what to draw
const env = task.env;

// ==========================================
// 3. Agent & Trainer Setup
// ==========================================
// Get the initial state (now an array of 8 elements because of our sin/cos fix!)
const initialState = task.reset();

// Dynamically size the Neural Network to match the state array length
const agent = new PPOAgent(initialState.length);

// timeBudgetMs = 16 (Spend ~16ms doing deep learning math before yielding to the browser)
const trainer = new PPOTrainer(agent, task, 16);

// ==========================================
// 4. Renderer & UI Setup
// ==========================================
const renderer = new CanvasRenderer(canvas);
const ui = new PPODOMUI(trainer);

// ==========================================
// 5. The Main Loops
// ==========================================

// Start the high-speed background training engine. 
// This runs on its own decoupled setTimeout loop.
trainer.tick();

// Start the visual engine.
// This is locked to the monitor's refresh rate (typically 60fps)
function renderLoop() {
    // 1. Always update the lightweight DOM dashboard
    ui.update();

    // 2. Only run the expensive physics rasterizer if the user hasn't paused it
    if (!renderToggle || renderToggle.checked) {
        // (Assuming CanvasRenderer doesn't auto-clear, it's good practice to clear the frame)
        const ctx = canvas.getContext('2d');
        if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        renderer.render(env);
    }

    // Loop!
    requestAnimationFrame(renderLoop);
}

// Ignite
renderLoop();