import { CanvasRenderer } from './renderer/CanvasRenderer';
import { DQNAgent } from './engine/DQNAgent';

import {
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    TRACK_HEIGHT,
    THRUST_LEVELS,
    ENERGY_PENALTY_WEIGHT,
    FIXED_DT,
    TRAIN_TIME_BUDGET_MS,
    AGENT_CONFIG,
} from './config';
import { Trainer } from './training/Trainer';
import { DiagnosticsPanel } from './ui/DiagnosticsPanel';
import { CartPoleTask } from './sim/CartPoleTask';

// ==========================================
// Wiring only. Every actual behavior lives in sim/CartPoleTask,
// training/Trainer, ui/DiagnosticsPanel, or engine/DQNAgent.
// ==========================================

const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;
const ctx = canvas.getContext('2d')!;
const renderer = new CanvasRenderer(canvas);

const task = new CartPoleTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT, ENERGY_PENALTY_WEIGHT);

const agent = new DQNAgent(AGENT_CONFIG.inputSize, THRUST_LEVELS.length);
agent.epsilonDecay = AGENT_CONFIG.epsilonDecay;
agent.learningRate = AGENT_CONFIG.learningRate;

const trainer = new Trainer(agent, task, THRUST_LEVELS, TRAIN_TIME_BUDGET_MS);
const diagnostics = new DiagnosticsPanel(ctx, CANVAS_WIDTH, TRACK_HEIGHT, THRUST_LEVELS);

// ==========================================
// RENDER LOOP — steady 60fps, reads whatever state currently exists.
// Completely decoupled from how many training steps have happened.
// ==========================================
function renderLoop() {
    const latestQValues = agent.getQValues(trainer.currentState);

    renderer.render(task.env);
    diagnostics.draw({
        episode: trainer.episode,
        score: trainer.score,
        currentMovingAvg: trainer.currentMovingAvg,
        currentLoss: trainer.currentLoss,
        currentQ: trainer.currentQ,
        stepsPerSecond: trainer.stepsPerSecond,
        lossHistory: trainer.lossHistory,
        qValueHistory: trainer.qValueHistory,
        latestQValues,
        currentActionIndex: trainer.currentActionIndex,
        thrustFraction: THRUST_LEVELS[trainer.currentActionIndex],
        cartX: task.cart.position.x,
    });

    requestAnimationFrame(renderLoop);
}

trainer.tick();
renderLoop();