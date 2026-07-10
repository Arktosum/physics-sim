import { CanvasRenderer } from '../renderer/CanvasRenderer';
import { ReinforceAgent } from '../engine/ReinforceAgent';
import { ReinforceTrainer } from '../training/ReinforceTrainer';
import { DoublePendulumTask } from '../sim/DoublePendulumTask';
import {
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    TRACK_HEIGHT,
    FIXED_DT,
    TRAIN_TIME_BUDGET_MS,
    AGENT_CONFIG,
} from '../config';

// ==========================================
// Wiring only, mirroring main.ts's structure — same Task, same renderer,
// swapped agent + trainer. DQNAgent's main.ts is completely untouched.
// ==========================================

const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;
const ctx = canvas.getContext('2d')!;
const renderer = new CanvasRenderer(canvas);

const task = new DoublePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT);
const agent = new ReinforceAgent(AGENT_CONFIG.inputSize);

const trainer = new ReinforceTrainer(agent, task, TRAIN_TIME_BUDGET_MS);

// ==========================================
// DYNAMIC UI CONTROLS (Speed, Save, Load) — same pattern as main.ts
// ==========================================
const uiContainer = document.createElement('div');
uiContainer.style.position = 'absolute';
uiContainer.style.top = '10px';
uiContainer.style.right = '20px';
uiContainer.style.display = 'flex';
uiContainer.style.flexDirection = 'column';
uiContainer.style.gap = '10px';
document.body.appendChild(uiContainer);

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

const saveBtn = document.createElement('button');
saveBtn.innerText = '💾 Save Brain (.json)';
saveBtn.style.cursor = 'pointer';
saveBtn.onclick = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(agent.toJSON());
    const anchor = document.createElement('a');
    anchor.setAttribute("href", dataStr);
    anchor.setAttribute("download", `reinforce-brain-ep${trainer.episode}.json`);
    anchor.click();
};
uiContainer.appendChild(saveBtn);

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

            trainer.currentState = task.reset();
            trainer.episode = 1;
            trainer.score = 0;
            trainer.stepsThisEpisode = 0;
            trainer.currentActorLoss = 0;
            trainer.currentCriticLoss = 0;

            trainer.scoreHistory.length = 0;
            trainer.actorLossHistory.length = 0;
            trainer.criticLossHistory.length = 0;
            trainer.movingAverageHistory.length = 0;
            trainer.survivalTimeHistory.length = 0;

            console.log("Brain loaded!");
        }
    };
    loadInput.value = '';
    reader.readAsText(file);
});
loadLabel.appendChild(loadInput);
uiContainer.appendChild(loadLabel);

// ==========================================
// DIAGNOSTICS — drawn inline here rather than reusing ui/DiagnosticsPanel.ts,
// since that panel is built around discrete Q-values-per-action, which has
// no equivalent for a continuous Gaussian policy (there's no "bar per action"
// to draw anymore — mean/std IS the whole action distribution).
// ==========================================
function drawDiagnostics() {
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(10, 10, 260, 230);
    ctx.fillStyle = 'white';
    ctx.font = '14px monospace';
    ctx.fillText(`Episode:   ${trainer.episode}`, 20, 35);
    ctx.fillText(`Score:     ${trainer.score.toFixed(1)}`, 20, 55);
    ctx.fillText(`MaxScore:  ${trainer.maxScore.toFixed(1)}`, 20, 75);

    ctx.fillStyle = '#4ade80';
    ctx.fillText(`MovAvg:    ${trainer.currentMovingAvg.toFixed(1)}`, 20, 95);
    ctx.fillText(`MaxMovAvg: ${trainer.maxMovingAvg.toFixed(1)}`, 20, 115);

    ctx.fillStyle = '#facc15';
    ctx.fillText(`Mean:      ${trainer.currentMean.toFixed(3)}`, 20, 140);
    ctx.fillText(`Std:       ${trainer.currentStd.toFixed(3)}`, 20, 160);

    ctx.fillStyle = '#ef4444';
    ctx.fillText(`ActorLoss: ${trainer.currentActorLoss.toFixed(4)}`, 20, 185);
    ctx.fillStyle = '#8b5cf6';
    ctx.fillText(`CriticLoss:${trainer.currentCriticLoss.toFixed(4)}`, 20, 205);
    ctx.fillStyle = 'white';
    ctx.fillText(`Advantage: ${trainer.currentAdvantage.toFixed(3)}`, 20, 225);

    drawChart(trainer.actorLossHistory, canvas.width - 220, 20, 200, 60, 'Actor Loss', '#ef4444');
    drawChart(trainer.criticLossHistory, canvas.width - 220, 90, 200, 60, 'Critic Loss', '#8b5cf6');
    drawChart(trainer.movingAverageHistory, canvas.width - 220, 160, 200, 60, 'Moving Avg', '#4ade80');
}

function drawChart(data: number[], x: number, y: number, w: number, h: number, label: string, color: string) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = '#4b5563';
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = 'white';
    ctx.font = '10px monospace';
    ctx.fillText(label, x + 5, y + 12);

    if (data.length < 2) return;

    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const range = (max - min) || 1;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    const stepX = w / (data.length - 1);
    for (let i = 0; i < data.length; i++) {
        const px = x + i * stepX;
        const py = y + h - ((data[i] - min) / range) * h;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
}

function renderLoop() {
    renderer.render(task.env);
    drawDiagnostics();
    requestAnimationFrame(renderLoop);
}

trainer.tick();
renderLoop();