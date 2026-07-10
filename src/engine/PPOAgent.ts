// src/engine/PPOAgent.ts

import { DenseLayer } from "../lib/DenseLayer";
import { NeuralNetwork } from "../lib/NeuralNetwork";
import { ReLULayer } from "../lib/ReLULayer";
import { clamp, sampleStandardNormal } from "../lib/mathUtils";
import type { PPORolloutBuffer } from "../training/PPORolloutBuffer";

const LOG_STD_MIN = -3.0;
const LOG_STD_MAX = 0.5;

export class PPOAgent {
    public actor: NeuralNetwork;
    public critic: NeuralNetwork;

    // PPO Specific Hyperparameters
    public actorLearningRate: number = 0.0003;  // Standard stable PPO rate
    public criticLearningRate: number = 0.001;
    public gamma: number = 0.99;                // Discount factor
    public lam: number = 0.95;                  // Lambda (λ) for Generalized Advantage Estimation
    public clipRatio: number = 0.2;             // THE SPEED LIMIT: 20% max change per update

    constructor(inputSize: number) {
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
    public learn(buffer: PPORolloutBuffer): {
        actorLoss: number;
        criticLoss: number;
        avgAdvantage: number;
        clipFraction: number;
        klDivergence: number
    } {
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
        let lastValue = 0; // The value of the state AFTER the final step (0 if the pendulum fell)

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
        let epochs = 6; // PPO usually runs this loop 4 to 10 times over the same batch of memory (Epochs).
        const GRADIENT_CLIP = 5.0;
        const clipGrad = (g: number) => (isNaN(g) ? 0 : clamp(g, -GRADIENT_CLIP, GRADIENT_CLIP));

        // PPO usually runs this loop 4 to 10 times over the same batch of memory (Epochs).

        for (let e = 0; e < epochs; e++) {
            for (let t = 0; t < length; t++) {
                const { state, action, logProb: oldLogProb } = episode[t];
                const advantage = advantages[t];
                const G_t = returns[t];

                // 1. Train the Critic (Coach) - Standard Mean Squared Error against the GAE Return
                const criticLoss = this.critic.train(state, [G_t], this.criticLearningRate);
                totalCriticLoss += criticLoss;

                // 2. Ask the NEW Actor what it thinks of this state NOW
                const { mean: liveMean, std: liveStd } = this.decodeActorOutput(state);
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

                // 6. APPLY THE UPDATE TO THE ACTOR
                this.actor.trainWithGradient(state, [dLoss_dMeanRaw, dLoss_dLogStdRawClipped], this.actorLearningRate);

                totalActorLoss += Math.abs(dLoss_dMeanRaw) + Math.abs(dLoss_dLogStdRawClipped);
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
}