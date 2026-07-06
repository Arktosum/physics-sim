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

// Create and style the input for STEPS_PER_FRAME
const inputContainer = document.createElement('div');
inputContainer.style.position = 'absolute';
inputContainer.style.top = '170px';
inputContainer.style.left = '20px';
inputContainer.style.zIndex = '10';
inputContainer.innerHTML = `
    <label style="color: white; font-family: monospace;">Steps/Frame: </label>
    <input type="number" id="stepsInput" value="15" style="width: 50px; background: #333; color: white; border: 1px solid #555;">
`;
document.body.appendChild(inputContainer);
const stepsInput = document.getElementById('stepsInput') as HTMLInputElement;

// 2. Initialize Physics Environment
const env = new Environment(9.81);

// 3. Populate our world: The Passive Cart-Pole
const trackHeight = 400;

// The Cart & Pole
const cart = new PointMass(400, trackHeight, 10, false);
const poleTop = new PointMass(400, 200, 2, false);

// The Constraints
const bone = new DistanceConstraint(cart, poleTop, 200);
const track = new AxisConstraint(cart, trackHeight);
const screenBounds = new BoundaryConstraint(cart, canvas.width, canvas.height, 0);

env.addPoint(cart);
env.addPoint(poleTop);
env.addConstraint(bone);
env.addConstraint(track);
env.addConstraint(screenBounds);

// The Motor
const motor = new Actuator(cart, 1000);

// 4. Initialize The Brain (2 actions: Left and Right)
const agent = new QLearningAgent(2);

// State Tracking Variables
let episode = 1;
let score = 0;
let maxScore = 0;
const scoreHistory: number[] = [];
const movingAverageHistory: number[] = [];
let currentMovingAvg = 0;
let currentStateStr = "";

// ==========================================
// THE BRIDGE: Discretization Helpers
// ==========================================
function binVariable(value: number, min: number, max: number, bins: number): number {
    let clamped = Math.max(min, Math.min(max, value));
    let normalized = (clamped - min) / (max - min);
    let bin = Math.floor(normalized * bins);
    return Math.min(bin, bins - 1);
}

function discretize(cartX: number, cartVelX: number, poleAngle: number, poleVelAngle: number): string {
    const binX = binVariable(cartX, 50, 750, 4);
    const binVelX = binVariable(cartVelX, -300, 300, 6);
    const binAngle = binVariable(poleAngle, -0.418, 0.418, 12);
    const binVelAngle = binVariable(poleVelAngle, -4, 4, 12);
    return `${binX}-${binVelX}-${binAngle}-${binVelAngle}`;
}

// Helper to reset the physical world when the agent dies
function resetWorld() {
    cart.position.x = 400;
    cart.position.y = trackHeight;
    cart.oldPosition.x = 400;
    cart.oldPosition.y = trackHeight;

    const wobble = (Math.random() - 0.5) * 0.1;
    poleTop.position.x = 400 + Math.sin(wobble) * 200;
    poleTop.position.y = trackHeight - Math.cos(wobble) * 200;
    poleTop.oldPosition.x = poleTop.position.x;
    poleTop.oldPosition.y = poleTop.position.y;

    motor.activeDirection = 0;
}

// THE FIX: Purely stateless physics sensing using your Verlet engine's built-in memory!
function senseUniverse(dt: number) {
    const cartX = cart.position.x;
    const cartVelX = (cart.position.x - cart.oldPosition.x) / dt;

    const dx = poleTop.position.x - cart.position.x;
    const dy = cart.position.y - poleTop.position.y;
    const poleAngle = Math.atan2(dx, dy);

    const oldDx = poleTop.oldPosition.x - cart.oldPosition.x;
    const oldDy = cart.oldPosition.y - poleTop.oldPosition.y;
    const oldPoleAngle = Math.atan2(oldDx, oldDy);

    const poleVelAngle = (poleAngle - oldPoleAngle) / dt;

    return { cartX, cartVelX, poleAngle, poleVelAngle };
}

// Initial Setup
const FIXED_DT = 0.016;
resetWorld();
let initialRawState = senseUniverse(FIXED_DT);
currentStateStr = discretize(initialRawState.cartX, initialRawState.cartVelX, initialRawState.poleAngle, initialRawState.poleVelAngle);

// 5. The Main Loop (The Heartbeat)
function step() {
    // Dynamic access to the user-inputted value
    const stepsPerFrame = parseInt(stepsInput.value) || 1;

    for (let i = 0; i < stepsPerFrame; i++) {

        // 1. DECIDE
        const action = agent.getAction(currentStateStr);

        // 2. ACT
        motor.activeDirection = action === 0 ? -1 : 1;
        motor.apply();
        env.update(FIXED_DT);

        // 3. JUDGE & SENSE NEW REALITY
        const nextState = senseUniverse(FIXED_DT);
        const isDead = Math.abs(nextState.poleAngle) > 0.418 || nextState.cartX < 50 || nextState.cartX > 750;

        // EDGE FEAR PENALTY: Subtract reward if it gets too close to the edges!
        let penalty = 0;
        if (nextState.cartX < 150 || nextState.cartX > 650) penalty = 0.1;

        const reward = isDead ? -100 : (1 - penalty);

        // 4. LEARN 
        const nextStateStr = discretize(nextState.cartX, nextState.cartVelX, nextState.poleAngle, nextState.poleVelAngle);

        agent.learn(currentStateStr, action, reward, nextStateStr, isDead);

        if (isDead) {
            if (score > maxScore) maxScore = score;

            scoreHistory.push(score);
            if (scoreHistory.length > 50) scoreHistory.shift();

            currentMovingAvg = scoreHistory.reduce((a, b) => a + b, 0) / scoreHistory.length;
            movingAverageHistory.push(currentMovingAvg);
            if (movingAverageHistory.length > 100) movingAverageHistory.shift();

            // Decay epsilon
            const a = agent as any;
            if (a.epsilon > 0.01) {
                a.epsilon *= 0.9995;
            }

            resetWorld();
            episode++;
            score = 0;

            const freshState = senseUniverse(FIXED_DT);
            currentStateStr = discretize(freshState.cartX, freshState.cartVelX, freshState.poleAngle, freshState.poleVelAngle);

        } else {
            score += reward;
            currentStateStr = nextStateStr;
        }
    }

    // --- DRAWING ---
    renderer.render(env);

    // Overlay AI Stats on the canvas
    ctx.fillStyle = 'white';
    ctx.font = '20px monospace';
    ctx.fillText(`Episode: ${episode}`, 20, 40);
    ctx.fillText(`Apples:  ${score}`, 20, 70);

    // Draw Moving Avg and Max Score
    ctx.fillStyle = '#60a5fa';
    ctx.fillText(`Mov Avg: ${Math.round(currentMovingAvg)}`, 20, 100);
    ctx.fillStyle = '#4ade80';
    ctx.fillText(`Max App: ${maxScore}`, 20, 130);

    // Draw epsilon
    const currentEpsilon = (agent as any).epsilon ?? 1.0;
    ctx.fillStyle = '#facc15';
    ctx.fillText(`Chaos (Epsilon): ${(currentEpsilon * 100).toFixed(1)}%`, 20, 160);

    // --- DRAW PLOT ---
    const graphWidth = 200;
    const graphHeight = 100;
    const graphX = canvas.width - graphWidth - 20;
    const graphY = 20;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(graphX, graphY, graphWidth, graphHeight);
    ctx.strokeStyle = '#4b5563';
    ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);
    ctx.fillStyle = 'white';
    ctx.font = '12px monospace';
    ctx.fillText('Moving Avg History', graphX, graphY - 5);

    if (movingAverageHistory.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2;

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