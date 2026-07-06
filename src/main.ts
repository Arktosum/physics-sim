import { Environment } from './engine/Environment';
import { CanvasRenderer } from './renderer/CanvasRenderer';
import { PointMass } from './state/PointMass';
import { AxisConstraint, BoundaryConstraint, DistanceConstraint } from './engine/Constraint';
import { Actuator } from './engine/Actuator';
import { DQNAgent } from './engine/DQNAgent';

const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = 800;
canvas.height = 600;
const ctx = canvas.getContext('2d')!;
const renderer = new CanvasRenderer(canvas);

const env = new Environment(9.81);
const trackHeight = 400;

const cart = new PointMass(400, trackHeight, 10, false);
const pole = new PointMass(400, 300, 2, false);

const link = new DistanceConstraint(cart, pole, 100);
const track = new AxisConstraint(cart, trackHeight);
const screenBounds = new BoundaryConstraint(cart, canvas.width, canvas.height, 0);

env.addPoint(cart);
env.addPoint(pole);
env.addConstraint(link);
env.addConstraint(track);
env.addConstraint(screenBounds);

const motor = new Actuator(cart, 1500);

const THRUST_LEVELS = [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0];
const ENERGY_PENALTY_WEIGHT = 0.02;

const agent = new DQNAgent(4, THRUST_LEVELS.length);
agent.epsilonDecay = 0.995;
agent.learningRate = 0.001;

let currentActionIndex = 0;

let episode = 1;
let score = 0;
let maxScore = 0;
const scoreHistory: number[] = [];
const lossHistory: number[] = [];
const qValueHistory: number[] = [];
const movingAverageHistory: number[] = [];

let currentLoss = 0;
let currentQ = 0;
let currentMovingAvg = 0;
let maxMovingAvg = 0;
let latestQValues: number[] = new Array(THRUST_LEVELS.length).fill(0);

// NEW: throughput tracking, so you can actually see the speedup.
let stepsThisSecond = 0;
let stepsPerSecond = 0;
let lastThroughputCheck = performance.now();

function resetWorld() {
    cart.position.x = 400;
    cart.position.y = trackHeight;
    cart.oldPosition.x = 400;
    cart.oldPosition.y = trackHeight;
    const wobble = (Math.random() - 0.5) * 0.2;
    pole.position.x = 400 + Math.sin(wobble) * 100;
    pole.position.y = trackHeight - Math.cos(wobble) * 100;
    pole.oldPosition.x = pole.position.x;
    pole.oldPosition.y = pole.position.y;
    motor.activeDirection = 0;
}

function clamp(val: number, min: number, max: number): number {
    return isNaN(val) ? 0 : Math.max(min, Math.min(max, val));
}

function senseAndNormalize(dt: number): number[] {
    const dx = pole.position.x - cart.position.x;
    const dy = cart.position.y - pole.position.y;
    const a = Math.atan2(dx, dy);
    const oldDx = pole.oldPosition.x - cart.oldPosition.x;
    const oldDy = cart.oldPosition.y - pole.oldPosition.y;
    const oldA = Math.atan2(oldDx, oldDy);
    let da = a - oldA;
    if (da > Math.PI) da -= 2 * Math.PI;
    if (da < -Math.PI) da += 2 * Math.PI;
    const v = da / dt;
    const cartV = (cart.position.x - cart.oldPosition.x) / dt;
    return [
        clamp((cart.position.x - 400) / 400, -1, 1),
        clamp(cartV / 500.0, -1, 1),
        clamp(a / 1.0, -1, 1),
        clamp(v / 10.0, -1, 1)
    ];
}

const FIXED_DT = 0.016;
resetWorld();
let currentState = senseAndNormalize(FIXED_DT);

// ==========================================
// ONE ENVIRONMENT STEP — no rendering in here at all.
// This is the unit of work the training loop hammers on as fast as it can.
// ==========================================
function doOneStep() {
    currentActionIndex = agent.getAction(currentState);
    const thrustFraction = THRUST_LEVELS[currentActionIndex];
    motor.activeDirection = thrustFraction;
    motor.apply();
    env.update(FIXED_DT);

    const nextState = senseAndNormalize(FIXED_DT);
    const rawA = nextState[2];
    const isDead = Math.abs(rawA) > 0.8 || cart.position.x < 50 || cart.position.x > 750;

    const reward = isDead
        ? -10
        : 1 - ENERGY_PENALTY_WEIGHT * Math.abs(thrustFraction);

    agent.remember(currentState, currentActionIndex, reward, nextState, isDead);
    const metrics = agent.replay();
    if (metrics && !isNaN(metrics.loss)) {
        currentLoss = currentLoss * 0.99 + metrics.loss * 0.01;
        currentQ = currentQ * 0.99 + metrics.qValue * 0.01;
        if (Math.random() < 0.05) {
            lossHistory.push(currentLoss);
            qValueHistory.push(currentQ);
            if (lossHistory.length > 200) lossHistory.shift();
            if (qValueHistory.length > 200) qValueHistory.shift();
        }
    }

    if (isDead) {
        if (score > maxScore) maxScore = score;
        scoreHistory.push(score);
        if (scoreHistory.length > 100) scoreHistory.shift();
        currentMovingAvg = scoreHistory.reduce((a, b) => a + b, 0) / scoreHistory.length;
        if (currentMovingAvg > maxMovingAvg) maxMovingAvg = currentMovingAvg;
        movingAverageHistory.push(currentMovingAvg);
        if (movingAverageHistory.length > 200) movingAverageHistory.shift();
        agent.decayEpsilon();
        resetWorld();
        episode++;
        score = 0;
        currentState = senseAndNormalize(FIXED_DT);
    } else {
        score += reward;
        currentState = nextState;
    }

    stepsThisSecond++;
}

// ==========================================
// TRAINING LOOP — runs as fast as the CPU allows, independent of rendering.
// Budgets ~8ms of work per tick, then yields via setTimeout(fn, 0) so the
// browser tab stays responsive (can still paint, handle input, etc.)
// ==========================================
const TRAIN_TIME_BUDGET_MS = 8;

function trainLoop() {
    const start = performance.now();
    while (performance.now() - start < TRAIN_TIME_BUDGET_MS) {
        doOneStep();
    }

    const now = performance.now();
    if (now - lastThroughputCheck >= 1000) {
        stepsPerSecond = stepsThisSecond;
        stepsThisSecond = 0;
        lastThroughputCheck = now;
    }

    setTimeout(trainLoop, 0);
}

// ==========================================
// RENDER LOOP — steady 60fps, reads whatever state currently exists.
// Completely decoupled from how many training steps have happened.
// ==========================================
function renderLoop() {
    latestQValues = agent.getQValues(currentState);
    renderer.render(env);
    drawThrustGauge();
    drawDiagnostics();
    requestAnimationFrame(renderLoop);
}

function drawThrustGauge() {
    const centerX = cart.position.x;
    const y = trackHeight + 30;
    const maxBarWidth = 80;
    const thrustFraction = THRUST_LEVELS[currentActionIndex];

    ctx.strokeStyle = '#4b5563';
    ctx.strokeRect(centerX - maxBarWidth, y - 6, maxBarWidth * 2, 12);

    const barWidth = Math.abs(thrustFraction) * maxBarWidth;
    ctx.fillStyle = thrustFraction < 0 ? '#ef4444' : thrustFraction > 0 ? '#22c55e' : '#6b7280';
    if (thrustFraction < 0) {
        ctx.fillRect(centerX - barWidth, y - 6, barWidth, 12);
    } else if (thrustFraction > 0) {
        ctx.fillRect(centerX, y - 6, barWidth, 12);
    }

    ctx.fillStyle = 'white';
    ctx.font = '11px monospace';
    const label = thrustFraction === 0 ? 'COAST' : `${(thrustFraction * 100).toFixed(0)}%`;
    ctx.fillText(label, centerX - 18, y + 25);
}

function drawDiagnostics() {
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(10, 10, 250, 230);
    ctx.fillStyle = 'white';
    ctx.font = '14px monospace';
    ctx.fillText(`Episode:   ${episode}`, 20, 35);
    ctx.fillText(`Score:     ${score.toFixed(1)}`, 20, 55);
    ctx.fillText(`MovAvg:    ${currentMovingAvg.toFixed(1)}`, 20, 75);
    ctx.fillText(`Loss:      ${currentLoss.toFixed(4)}`, 20, 95);
    ctx.fillText(`Q-Val:     ${currentQ.toFixed(2)}`, 20, 115);
    ctx.fillStyle = '#facc15';
    ctx.fillText(`Steps/sec: ${stepsPerSecond}`, 20, 135);
    ctx.fillStyle = 'white';
    drawChart(lossHistory, 20, 150, 220, 40, 'MSE Loss', '#ef4444');
    drawChart(qValueHistory, 20, 195, 220, 40, 'Avg Max Q', '#8b5cf6');

    drawActionQValues();
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

function drawActionQValues() {
    const panelX = canvas.width - 260;
    const panelY = 10;
    const panelW = 250;
    const panelH = 140;

    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(panelX, panelY, panelW, panelH);
    ctx.fillStyle = 'white';
    ctx.font = '11px monospace';
    ctx.fillText('Q-values per thrust level', panelX + 8, panelY + 16);

    const maxQ = Math.max(...latestQValues, 1);
    const minQ = Math.min(...latestQValues, 0);
    const range = (maxQ - minQ) || 1;

    const barAreaW = panelW - 20;
    const barW = barAreaW / THRUST_LEVELS.length - 4;
    const baseY = panelY + panelH - 20;
    const maxBarH = panelH - 45;

    for (let i = 0; i < THRUST_LEVELS.length; i++) {
        const x = panelX + 10 + i * (barW + 4);
        const h = ((latestQValues[i] - minQ) / range) * maxBarH;
        const isChosen = i === currentActionIndex;

        ctx.fillStyle = isChosen ? '#facc15' : '#3b82f6';
        ctx.fillRect(x, baseY - h, barW, h);

        ctx.fillStyle = '#9ca3af';
        ctx.font = '8px monospace';
        const levelLabel = THRUST_LEVELS[i] === 0 ? '0' : (THRUST_LEVELS[i] * 100).toFixed(0);
        ctx.fillText(levelLabel, x, baseY + 12);
    }
}

trainLoop();
renderLoop();