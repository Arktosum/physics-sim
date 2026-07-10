// src/engine/PPOAgent.ts

import { DenseLayer } from "../lib/DenseLayer";
import { NeuralNetwork } from "../lib/NeuralNetwork";
import { ReLULayer } from "../lib/ReLULayer";
import { Matrix } from "../lib/Matrix";
import { clamp, sampleStandardNormal } from "../lib/mathUtils";
import type { PPORolloutBuffer } from "../training/PPORolloutBuffer";

const LOG_STD_MIN = -3.0;
const LOG_STD_MAX = 0.5;

// PPO's epoch loop used to run epochs * length single-sample updates back to
// back (e.g. 3 * 2048 = 6144), each one a handful of ~64-wide matrix ops.
// That's the worst shape for a JS engine: thousands of tiny function calls
// instead of a few wide ones. MINIBATCH_SIZE=64 is also the standard PPO
// default (e.g. Stable-Baselines3) — we were doing single-sample SGD before,
// which is a simplification, not the canonical algorithm. Batching brings
// this in line with normal PPO AND cuts weight-update calls by ~64x
// (2048/64 * 3 = 96 batched updates instead of 6144 single-sample ones).
const MINIBATCH_SIZE = 64;

// Yielding once per minibatch (rather than every N samples) keeps the
// Worker's own timers (frame/metrics postMessage) responsive between
// batched updates without adding much overhead — there are far fewer
// iterations to yield inside now.
const yieldToEventLoop = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0));

export class PPOAgent {
    public actor: NeuralNetwork;
    public critic: NeuralNetwork;

    // PPO Specific Hyperparameters
    public actorLearningRate: number = 0.0005;  // Standard stable PPO rate
    public criticLearningRate: number = 0.001;
    public gamma: number = 0.99;                // Discount factor
    public lam: number = 0.95;                  // Lambda (λ) for Generalized Advantage Estimation
    public clipRatio: number = 0.2;             // THE SPEED LIMIT: 20% max change per update

    private readonly inputSize: number;

    // Batch-shaped scratch, owned by the agent (not the networks) since
    // they're built from scattered per-sample state arrays gathered across
    // a minibatch — something only learn() has the context to assemble.
    // Sized lazily on first use, reused for every minibatch after that as
    // long as the width doesn't change (it only changes on the last, possibly
    // ragged, minibatch of a learn() call).
    private stateBatchScratch: Matrix | null = null;
    private criticGradBatchScratch: Matrix | null = null;
    private actorGradBatchScratch: Matrix | null = null;

    constructor(inputSize: number) {
        this.inputSize = inputSize;
        // Same architecture as before!
        this.actor = new NeuralNetwork([
            new DenseLayer(inputSize, 64), new ReLULayer(),
            new DenseLayer(64, 64), new ReLULayer(),
            new DenseLayer(64, 2) // [meanRaw, logStdRaw]
        ]);

        this.critic = new NeuralNetwork([
            new DenseLayer(inputSize, 64), new ReLULayer(),
            new DenseLayer(64, 64), new ReLULayer(),
            new DenseLayer(64, 1) // [V(s)]
        ]);
    }

    /**
     * Calculates the Log Probability of a specific action given a Gaussian distribution.
     * This is the mathematical core of PPO's "Ratio" calculation.
     */
    private calculateLogProb(action: number, mean: number, std: number): number {
        const variance = std * std;
        const diff = action - mean;
        // Formula: -0.5 * [ ((x - μ)^2 / σ^2) + ln(2π * σ^2) ]
        return -0.5 * ((diff * diff) / variance + Math.log(2 * Math.PI * variance));
    }

    private decodeActorOutput(state: number[]): { meanRaw: number; mean: number; logStdRaw: number; std: number } {
        const [meanRaw, logStdRawUnclamped] = this.actor.predict(state);
        const logStdRaw = clamp(logStdRawUnclamped, LOG_STD_MIN, LOG_STD_MAX);
        const mean = Math.tanh(meanRaw);
        const std = Math.exp(logStdRaw);
        return { meanRaw, mean, logStdRaw, std };
    }

    /**
     * The PPO Act function. It doesn't just return the action; it queries BOTH networks
     * simultaneously so the RolloutBuffer can record exactly what the brain was thinking.
     */
    public act(state: number[]): {
        rawAction: number;
        clampedAction: number;
        logProb: number;
        value: number
    } {
        // 1. Ask the Actor what to do
        const { mean, std } = this.decodeActorOutput(state);
        const z = sampleStandardNormal();
        const rawAction = mean + std * z;
        const clampedAction = clamp(rawAction, -1, 1);

        // 2. Calculate the exact mathematical probability of that action
        const logProb = this.calculateLogProb(rawAction, mean, std);

        // 3. Ask the Critic for its baseline expectation right now
        const value = this.critic.predict(state)[0];

        return { rawAction, clampedAction, logProb, value };
    }

    public getValue(state: number[]): number {
        return this.critic.predict(state)[0];
    }

    /**
     * The PPO Learn Function.
     * Takes a completely filled RolloutBuffer, trains the Actor and Critic, and returns the health metrics.
     */
    public async learn(buffer: PPORolloutBuffer, bootstrapValue: number = 0): Promise<{
        actorLoss: number;
        criticLoss: number;
        avgAdvantage: number;
        clipFraction: number;
        klDivergence: number
    }> {
        const episode = buffer.get();
        const length = episode.length;

        // ============================================================================
        // STEP 1: Generalized Advantage Estimation (GAE)
        // REINFORCE just looked at the final score. GAE looks at the Critic's prediction 
        // step-by-step and smoothly blends short-term rewards with long-term predictions.
        // ============================================================================
        const advantages = new Array(length).fill(0);
        const returns = new Array(length).fill(0);

        let lastAdvantage = 0;
        // The value of the state AFTER the final stored step. This buffer is filled by a
        // fixed HORIZON, not by waiting for the pendulum to actually fall — so most of the
        // time this rollout is truncated mid-episode, not terminated. In that case the true
        // future value is NOT 0, it's whatever the Critic thinks the next state is worth.
        // (If the last transition genuinely WAS terminal, `nextNonTerminal` inside the loop
        // below already multiplies this by 0 for us on its very first use, so passing a real
        // bootstrap here is always safe — no separate "was it actually done" check needed.)
        let lastValue = bootstrapValue;

        for (let t = length - 1; t >= 0; t--) {
            const transition = episode[t];

            // If the pendulum fell on this frame, the next state has no future value.
            const nextNonTerminal = transition.done ? 0 : 1;

            // The Temporal Difference Error (Did this exact step go better than the Critic expected?)
            const delta = transition.reward + this.gamma * lastValue * nextNonTerminal - transition.value;

            // GAE blends this step's error with the accumulated errors of the future
            lastAdvantage = delta + this.gamma * this.lam * nextNonTerminal * lastAdvantage;
            advantages[t] = lastAdvantage;

            // The true return is what the Critic originally guessed, PLUS the calculated advantage
            returns[t] = advantages[t] + transition.value;
            lastValue = transition.value;
        }

        // --- Normalize the Advantages (Crucial for Neural Network stability) ---
        const advMean = advantages.reduce((a, b) => a + b, 0) / length;
        const advVariance = advantages.reduce((sum, a) => sum + Math.pow(a - advMean, 2), 0) / length;
        const advStd = Math.sqrt(advVariance) + 1e-8; // 1e-8 prevents division by zero

        for (let t = 0; t < length; t++) {
            advantages[t] = (advantages[t] - advMean) / advStd;
        }

        // ============================================================================
        // STEP 2: The Clipped Surrogate Objective (The PPO Epoch)
        // ============================================================================
        let totalActorLoss = 0;
        let totalCriticLoss = 0;
        let clipCount = 0;
        let klSum = 0;
        let epochs = 3; // PPO usually runs this loop 4 to 10 times over the same batch of memory (Epochs).
        const GRADIENT_CLIP = 5.0;
        const clipGrad = (g: number) => (isNaN(g) ? 0 : clamp(g, -GRADIENT_CLIP, GRADIENT_CLIP));

        // PPO usually runs this loop 4 to 10 times over the same batch of memory (Epochs).

        let minibatchesSinceYield = 0;

        for (let e = 0; e < epochs; e++) {
            for (let mbStart = 0; mbStart < length; mbStart += MINIBATCH_SIZE) {
                const batchWidth = Math.min(MINIBATCH_SIZE, length - mbStart);

                // --- Gather this minibatch's states into (inputSize, batchWidth) ---
                if (!this.stateBatchScratch || this.stateBatchScratch.cols !== batchWidth) {
                    this.stateBatchScratch = new Matrix(this.inputSize, batchWidth);
                }
                const stateBatch = this.stateBatchScratch;
                for (let col = 0; col < batchWidth; col++) {
                    const s = episode[mbStart + col].state;
                    for (let row = 0; row < this.inputSize; row++) {
                        stateBatch.data[row * batchWidth + col] = s[row];
                    }
                }

                // ================= CRITIC: one batched forward + backward =================
                const criticPred = this.critic.predictBatch(stateBatch); // (1, batchWidth)

                if (!this.criticGradBatchScratch || this.criticGradBatchScratch.cols !== batchWidth) {
                    this.criticGradBatchScratch = new Matrix(1, batchWidth);
                }
                const criticGradBatch = this.criticGradBatchScratch;

                for (let col = 0; col < batchWidth; col++) {
                    const error = criticPred.data[col] - returns[mbStart + col];
                    // Same Huber-ish gradient clip as the old single-sample train().
                    criticGradBatch.data[col] = Math.max(-1, Math.min(1, error));
                    totalCriticLoss += error * error;
                }

                this.critic.backwardBatchWithGradient(criticGradBatch, this.criticLearningRate);

                // ================= ACTOR: one batched forward, per-sample PPO math, one batched backward =================
                // The clipped-surrogate objective and its gradient are inherently
                // per-sample scalar math (ratio, clipping, log-prob derivatives) —
                // there's no useful way to express that as a matrix op. What we DO
                // batch is the expensive part: the forward pass that produces
                // mean/std for every sample in the minibatch, and the backward pass
                // that applies all their gradients in one update instead of 64.
                const actorOut = this.actor.predictBatch(stateBatch); // (2, batchWidth): row0=meanRaw, row1=logStdRawUnclamped

                if (!this.actorGradBatchScratch || this.actorGradBatchScratch.cols !== batchWidth) {
                    this.actorGradBatchScratch = new Matrix(2, batchWidth);
                }
                const actorGradBatch = this.actorGradBatchScratch;

                for (let col = 0; col < batchWidth; col++) {
                    const idx = mbStart + col;
                    const { action, logProb: oldLogProb } = episode[idx];
                    const advantage = advantages[idx];

                    const meanRaw = actorOut.data[0 * batchWidth + col];
                    const logStdRawUnclamped = actorOut.data[1 * batchWidth + col];
                    const logStdRaw = clamp(logStdRawUnclamped, LOG_STD_MIN, LOG_STD_MAX);
                    const liveMean = Math.tanh(meanRaw);
                    const liveStd = Math.exp(logStdRaw);

                    // 2. What does the NEW Actor think of this state NOW?
                    const newLogProb = this.calculateLogProb(action, liveMean, liveStd);

                    // KL Divergence Metric (How much has the brain changed its mind since the action was taken?)
                    klSum += (oldLogProb - newLogProb);

                    // 3. THE RATIO (New Probability / Old Probability)
                    // Because we are using Logarithmic math: exp(A - B) is the exact same as A / B
                    const ratio = Math.exp(newLogProb - oldLogProb);

                    // 4. THE SPEED LIMIT (Clipping)
                    const unclippedObjective = ratio * advantage;
                    const clippedObjective = clamp(ratio, 1 - this.clipRatio, 1 + this.clipRatio) * advantage;

                    // PPO wants to MAXIMIZE the minimum of these two objectives.
                    const objective = Math.min(unclippedObjective, clippedObjective);

                    // 5. CALCULATING THE MATHEMATICAL GRADIENT
                    // This is where we literally sever the brain's ability to learn if it goes too fast.
                    let dObj_dRatio = 0;

                    if (unclippedObjective === objective) {
                        // We are inside the 20% Trust Region! The gradient is allowed to flow.
                        dObj_dRatio = advantage;
                    } else {
                        // WE HIT THE SPEED LIMIT.
                        // The gradient flatlines to ZERO. The neural network cannot update its weights.
                        dObj_dRatio = 0;
                        clipCount++;
                    }

                    // --- Manual Chain Rule Backpropagation ---
                    // Because we want to MAXIMIZE the objective, we MINIMIZE the negative objective (Loss)
                    const dLoss_dRatio = -dObj_dRatio;

                    // Derivative of Ratio w.r.t newLogProb is just the Ratio itself (calculus of e^x)
                    const dLoss_dNewLogProb = dLoss_dRatio * ratio;

                    // Derivatives of the Bell Curve w.r.t our Neural Network outputs
                    const dLogProb_dMean = (action - liveMean) / (liveStd * liveStd);
                    const dLogProb_dLogStdRaw = ((action - liveMean) ** 2) / (liveStd * liveStd) - 1;

                    const dLoss_dMean = dLoss_dNewLogProb * dLogProb_dMean;
                    const dLoss_dLogStdRaw = dLoss_dNewLogProb * dLogProb_dLogStdRaw;

                    // Chain through the Tanh squashing function on the Mean
                    const dMean_dMeanRaw = 1 - liveMean * liveMean;
                    const dLoss_dMeanRaw = clipGrad(dLoss_dMean * dMean_dMeanRaw);
                    const dLoss_dLogStdRawClipped = clipGrad(dLoss_dLogStdRaw);

                    actorGradBatch.data[0 * batchWidth + col] = dLoss_dMeanRaw;
                    actorGradBatch.data[1 * batchWidth + col] = dLoss_dLogStdRawClipped;

                    totalActorLoss += Math.abs(dLoss_dMeanRaw) + Math.abs(dLoss_dLogStdRawClipped);
                }

                // 6. APPLY THE UPDATE TO THE ACTOR — one batched backward for the
                // whole minibatch (backwardBatchWithGradient averages the gradient
                // across batchWidth internally, i.e. standard mini-batch SGD).
                this.actor.backwardBatchWithGradient(actorGradBatch, this.actorLearningRate);

                minibatchesSinceYield++;
                if (minibatchesSinceYield >= 4) {
                    minibatchesSinceYield = 0;
                    await yieldToEventLoop();
                }
            }
        }

        return {
            actorLoss: totalActorLoss / length,
            criticLoss: totalCriticLoss / length,
            avgAdvantage: advantages.reduce((a, b) => a + Math.abs(b), 0) / length,
            clipFraction: clipCount / length, // The % of times the speed limit was hit
            klDivergence: klSum / length
        };
    }

    public toJSON(): string {
        const dump = (net: NeuralNetwork) => net.layers.map(layer => {
            const l = layer as any;
            if (l.weights && l.biases) {
                return { weights: Array.from(l.weights.data), biases: Array.from(l.biases.data) };
            }
            return null;
        });
        return JSON.stringify({ actor: dump(this.actor), critic: dump(this.critic) });
    }

    public loadJSON(jsonString: string): void {
        const parsed = JSON.parse(jsonString);
        const load = (net: NeuralNetwork, data: any[]) => {
            for (let i = 0; i < net.layers.length; i++) {
                const l = net.layers[i] as any;
                const d = data[i];
                if (l.weights && l.biases && d) {
                    for (let j = 0; j < d.weights.length; j++) l.weights.data[j] = d.weights[j];
                    for (let j = 0; j < d.biases.length; j++) l.biases.data[j] = d.biases[j];
                }
            }
        };
        load(this.actor, parsed.actor);
        load(this.critic, parsed.critic);
    }
}