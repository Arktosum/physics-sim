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
import { DoublePendulumTask } from './sim/DoublePendulumTask';

// ==========================================
// Wiring only. Every actual behavior lives in sim/CartPoleTask,
// training/Trainer, ui/DiagnosticsPanel, or engine/DQNAgent.
// ==========================================

const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;
const ctx = canvas.getContext('2d')!;
const renderer = new CanvasRenderer(canvas);

const task = new DoublePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT);
const agent = new DQNAgent(AGENT_CONFIG.inputSize, THRUST_LEVELS.length);
agent.epsilonDecay = AGENT_CONFIG.epsilonDecay;
agent.learningRate = AGENT_CONFIG.learningRate;

const trainer = new Trainer(agent, task, THRUST_LEVELS, TRAIN_TIME_BUDGET_MS);
const diagnostics = new DiagnosticsPanel(ctx, CANVAS_WIDTH, TRACK_HEIGHT, THRUST_LEVELS);

// ==========================================
// DYNAMIC UI CONTROLS (Speed, Save, Load)
// ==========================================
const uiContainer = document.createElement('div');
uiContainer.style.position = 'absolute';
uiContainer.style.top = '10px';
uiContainer.style.right = '20px';
uiContainer.style.display = 'flex';
uiContainer.style.flexDirection = 'column';
uiContainer.style.gap = '10px';
document.body.appendChild(uiContainer);

// 1. SPEED CONTROL
const speedLabel = document.createElement('label');
speedLabel.style.color = 'white';
speedLabel.style.fontFamily = 'monospace';
speedLabel.innerText = 'Train Budget (ms): ';
const speedInput = document.createElement('input');
speedInput.type = 'number';
speedInput.value = TRAIN_TIME_BUDGET_MS.toString();
speedInput.style.width = '60px';
speedInput.style.background = '#333';
speedInput.style.color = 'white';
speedInput.style.border = '1px solid #555';
speedInput.addEventListener('input', () => {
    const val = parseInt(speedInput.value);
    if (!isNaN(val) && val >= 0) trainer.timeBudgetMs = val;
});
speedLabel.appendChild(speedInput);
uiContainer.appendChild(speedLabel);

// 2. SAVE BUTTON
const saveBtn = document.createElement('button');
saveBtn.innerText = '💾 Save Brain (.json)';
saveBtn.style.cursor = 'pointer';
saveBtn.onclick = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(agent.toJSON());
    const anchor = document.createElement('a');
    anchor.setAttribute("href", dataStr);
    // Names the file with the current episode so you know how smart it is
    anchor.setAttribute("download", `dqn-brain-ep${trainer.episode}.json`);
    anchor.click();
};
uiContainer.appendChild(saveBtn);

// 3. LOAD BUTTON
const loadLabel = document.createElement('label');
loadLabel.innerText = '📂 Load Brain: ';
loadLabel.style.color = 'white';
loadLabel.style.fontFamily = 'monospace';
const loadInput = document.createElement('input');
loadInput.type = 'file';
loadInput.accept = '.json';
loadInput.style.color = 'white';
loadInput.addEventListener('change', (e: any) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
        if (typeof event.target?.result === 'string') {
            agent.loadJSON(event.target.result);

            // 1. WIPE THE POISONED MEMORY!
            agent.memory.clear();

            // 2. Reset the physical world
            trainer.currentState = task.reset();

            // 3. Reset the tracking metrics so they don't skew the charts
            trainer.episode = 1;
            trainer.score = 0;
            trainer.stepsThisEpisode = 0;
            trainer.currentLoss = 0;
            trainer.currentQ = 0;

            // 4. Wipe the UI charts clean
            trainer.scoreHistory.length = 0;
            trainer.lossHistory.length = 0;
            trainer.qValueHistory.length = 0;
            trainer.movingAverageHistory.length = 0;
            trainer.survivalTimeHistory.length = 0;

            console.log("Brain loaded and memory wiped!");
        }
    };

    // Clear the input value so the browser allows you to select the same file again later
    loadInput.value = '';
    reader.readAsText(file);
});
loadLabel.appendChild(loadInput);
uiContainer.appendChild(loadLabel);
// ==========================================
// 4. INJECT CHAOS BUTTON
const chaosBtn = document.createElement('button');
chaosBtn.innerText = '🎲 Reset Chaos (Epsilon = 1.0)';
chaosBtn.style.cursor = 'pointer';
chaosBtn.onclick = () => {
    agent.epsilon = 1.0;

    // Optional: Wipe the memory buffer so it doesn't train on old rut data
    agent.memory.clear();

    console.log("Chaos injected! Agent is exploring again.");
};
uiContainer.appendChild(chaosBtn);

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
        avgSurvivalTime: trainer.currentAvgSurvivalTime,
        maxSurvivalTime: trainer.maxSurvivalTime,
    });

    requestAnimationFrame(renderLoop);
}

trainer.tick();
renderLoop();