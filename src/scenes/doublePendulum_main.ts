import { Environment } from './engine/Environment';
import { CanvasRenderer } from './renderer/CanvasRenderer';
import { PointMass } from './state/PointMass';
import { AxisConstraint, BoundaryConstraint, DistanceConstraint } from './engine/Constraint';
import { Actuator } from './engine/Actuator';
import { QLearningAgent } from './engine/QLearningAgent';

// 1. Initialize Canvas and Renderer
const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = 800;
canvas.height = 600;
const ctx = canvas.getContext('2d')!;
const renderer = new CanvasRenderer(canvas);

// Input for STEPS_PER_FRAME
const inputContainer = document.createElement('div');
inputContainer.style.position = 'absolute';
inputContainer.style.top = '170px';
inputContainer.style.left = '20px';
inputContainer.style.zIndex = '10';
inputContainer.innerHTML = `
    <label style="color: white; font-family: monospace;">Steps/Frame: </label>
    <input type="number" id="stepsInput" value="50" style="width: 50px; background: #333; color: white; border: 1px solid #555;">
`;
document.body.appendChild(inputContainer);
const stepsInput = document.getElementById('stepsInput') as HTMLInputElement;

// 2. Initialize Physics Environment
const env = new Environment(9.81);

// 3. Populate our world: Double Pendulum Cart-Pole
const trackHeight = 400;
const cart = new PointMass(400, trackHeight, 10, false);
const pole1 = new PointMass(400, 300, 2, false);
const pole2 = new PointMass(400, 200, 2, false);

// Constraints
const link1 = new DistanceConstraint(cart, pole1, 100);
const link2 = new DistanceConstraint(pole1, pole2, 100);
const track = new AxisConstraint(cart, trackHeight);
const screenBounds = new BoundaryConstraint(cart, canvas.width, canvas.height, 0);

env.addPoint(cart);
env.addPoint(pole1);
env.addPoint(pole2);
env.addConstraint(link1);
env.addConstraint(link2);
env.addConstraint(track);
env.addConstraint(screenBounds);

const motor = new Actuator(cart, 1500);

// Brain: Higher complexity for double pendulum
const agent = new QLearningAgent(2);

let episode = 1;
let score = 0;
let maxScore = 0;
const scoreHistory: number[] = [];
const movingAverageHistory: number[] = [];
let currentMovingAvg = 0;
let currentStateStr = "";



function binVariable(value: number, min: number, max: number, bins: number): number {
    let clamped = Math.max(min, Math.min(max, value));
    let normalized = (clamped - min) / (max - min);
    let bin = Math.floor(normalized * bins);
    return Math.min(bin, bins - 1);
}

// Discretization for double pendulum
function discretize(cartX: number, a1: number, a2: number, v1: number, v2: number): string {
    const binX = binVariable(cartX, 50, 750, 4);
    const binA1 = binVariable(a1, -1, 1, 16);   // was 10
    const binA2 = binVariable(a2, -1, 1, 16);   // was 10
    const binV1 = binVariable(v1, -6, 6, 8);    // was 6
    const binV2 = binVariable(v2, -8, 8, 8);    // was 6
    return `${binX}-${binA1}-${binA2}-${binV1}-${binV2}`;
}

function resetWorld() {
    cart.position.x = 400;
    cart.position.y = trackHeight;
    cart.oldPosition.x = 400;
    cart.oldPosition.y = trackHeight;

    const wobble1 = (Math.random() - 0.5) * 0.3; // radians
    const wobble2 = (Math.random() - 0.5) * 0.3;

    pole1.position.x = 400 + Math.sin(wobble1) * 100;
    pole1.position.y = trackHeight - Math.cos(wobble1) * 100;
    pole1.oldPosition.x = pole1.position.x;
    pole1.oldPosition.y = pole1.position.y;

    pole2.position.x = pole1.position.x + Math.sin(wobble1 + wobble2) * 100;
    pole2.position.y = pole1.position.y - Math.cos(wobble1 + wobble2) * 100;
    pole2.oldPosition.x = pole2.position.x;
    pole2.oldPosition.y = pole2.position.y;

    motor.activeDirection = 0;
}

function senseUniverse(dt: number) {
    const dx1 = pole1.position.x - cart.position.x;
    const dy1 = cart.position.y - pole1.position.y;
    const dx2 = pole2.position.x - pole1.position.x;
    const dy2 = pole1.position.y - pole2.position.y;
    const a1 = Math.atan2(dx1, dy1);
    const a2 = Math.atan2(dx2, dy2);

    const oldDx1 = pole1.oldPosition.x - cart.oldPosition.x;
    const oldDy1 = cart.oldPosition.y - pole1.oldPosition.y;
    const oldDx2 = pole2.oldPosition.x - pole1.oldPosition.x;
    const oldDy2 = pole1.oldPosition.y - pole2.oldPosition.y;
    const oldA1 = Math.atan2(oldDx1, oldDy1);
    const oldA2 = Math.atan2(oldDx2, oldDy2);

    return {
        cartX: cart.position.x,
        a1,
        a2,
        v1: (a1 - oldA1) / dt,
        v2: (a2 - oldA2) / dt
    };
}

const FIXED_DT = 0.016;
resetWorld();
let s = senseUniverse(FIXED_DT);
currentStateStr = discretize(s.cartX, s.a1, s.a2, s.v1, s.v2);

function step() {
    const stepsPerFrame = parseInt(stepsInput.value) || 1;

    for (let i = 0; i < stepsPerFrame; i++) {
        const action = agent.getAction(currentStateStr);
        motor.activeDirection = action === 0 ? -1 : 1;
        motor.apply();
        env.update(FIXED_DT);

        const next = senseUniverse(FIXED_DT);
        const isDead = Math.abs(next.a1) > 0.8 || Math.abs(next.a2) > 0.8 || next.cartX < 50 || next.cartX > 750;
        const reward = isDead ? -100 : (0.5 * Math.cos(next.a1) + 0.5 * Math.cos(next.a2));

        const nextStr = discretize(next.cartX, next.a1, next.a2, next.v1, next.v2);
        if (episode % 500 === 0 && i === 0) {
            console.log(`raw: a1=${next.a1.toFixed(4)} a2=${next.a2.toFixed(4)} v1=${next.v1.toFixed(4)} v2=${next.v2.toFixed(4)} cartX=${next.cartX.toFixed(1)}`);
        }

        agent.learn(currentStateStr, action, reward, nextStr, isDead);

        function mirrorState(state: string): string {
            const [x, a1, a2, v1, v2] = state.split('-').map(Number);
            const flip = (bin: number, numBins: number) => numBins - 1 - bin;
            return `${flip(x, 4)}-${flip(a1, 16)}-${flip(a2, 16)}-${flip(v1, 8)}-${flip(v2, 8)}`;
        }

        agent.learn(mirrorState(currentStateStr), 1 - action, reward, mirrorState(nextStr), isDead);

        if (isDead) {
            if (score > maxScore) maxScore = score;
            scoreHistory.push(score);
            if (scoreHistory.length > 50) scoreHistory.shift();
            currentMovingAvg = scoreHistory.reduce((a, b) => a + b, 0) / scoreHistory.length;
            movingAverageHistory.push(currentMovingAvg);
            if (movingAverageHistory.length > 100) movingAverageHistory.shift();
            const a = agent as any;
            a.epsilon = Math.max(0.1, a.epsilon * 0.9999);
            if (episode % 500 === 0) {
                console.log(agent.getCoverageStats(14400)); // 14400 = 4*10*10*6*6 from your discretize bins
            }
            resetWorld();
            episode++;
            score = 0;
            currentStateStr = discretize(400, 0, 0, 0, 0);

        } else {
            score += reward;
            currentStateStr = nextStr;
        }
    }

    renderer.render(env);

    // Draw Stats Dashboard
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(10, 10, 250, 150);
    ctx.fillStyle = 'white';
    ctx.font = '16px monospace';
    ctx.fillText(`Episode: ${episode}`, 20, 35);
    ctx.fillText(`Score: ${score}`, 20, 60);
    ctx.fillText(`Max Score: ${maxScore}`, 20, 85);
    ctx.fillText(`Moving Avg: ${Math.round(currentMovingAvg)}`, 20, 110);
    ctx.fillText(`Chaos: ${(agent as any).epsilon.toFixed(2)}`, 20, 135);

    // Draw graph
    const graphWidth = 200;
    const graphHeight = 100;
    const graphX = canvas.width - graphWidth - 20;
    const graphY = 20;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(graphX, graphY, graphWidth, graphHeight);
    ctx.strokeStyle = '#4b5563';
    ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);

    if (movingAverageHistory.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = '#60a5fa';
        const maxInHistory = Math.max(...movingAverageHistory, 10);
        const stepX = graphWidth / Math.max(movingAverageHistory.length - 1, 1);
        for (let i = 0; i < movingAverageHistory.length; i++) {
            const x = graphX + i * stepX;
            const y = graphY + graphHeight - (movingAverageHistory[i] / maxInHistory) * graphHeight;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    requestAnimationFrame(step);
}

requestAnimationFrame(step);