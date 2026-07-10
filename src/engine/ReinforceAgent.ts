import { DenseLayer } from "../lib/DenseLayer";
import { NeuralNetwork } from "../lib/NeuralNetwork";
import { ReLULayer } from "../lib/ReLULayer";
import { clamp, sampleStandardNormal } from "../lib/mathUtils";

/** One timestep recorded during an episode, kept in the EpisodeBuffer. */
export interface Transition {
    state: number[];
    /** The raw sampled action (mean + std*z), BEFORE clamping to the valid control range. */
    rawAction: number;
    /** The mean/std the Actor actually output for this state, needed to recompute logProb at learn() time. */
    mean: number;
    std: number;
    reward: number;
}

const LOG_STD_MIN = -3.0;   // std floor  ~= 0.05  -> stops the policy from ever going fully deterministic (which would kill exploration and blow up 1/std terms)
const LOG_STD_MAX = 0.5;    // std ceiling ~= 1.65  -> stops the policy from exploding into pure noise

/**
 * A single-action Gaussian-policy REINFORCE agent.
 *
 * Structurally this is DQNAgent's sibling, not its replacement: same "brain
 * holds weights, agent holds hyperparameters" shape, same NeuralNetwork/
 * DenseLayer building blocks. The differences are exactly the ones the notes
 * called out:
 *   - Actor outputs (mean, logStd) instead of one Q-value per discrete action.
 *   - Critic outputs a state-only baseline V(s), not a state+action Q(s,a).
 *   - There's no replay buffer / epsilon-greedy here — see EpisodeBuffer and
 *     ReinforceTrainer for how exploration and learning timing work instead.
 */
export class ReinforceAgent {
    public actor: NeuralNetwork;   // state -> [meanRaw, logStdRaw]
    public critic: NeuralNetwork;  // state -> [V(s)]

    public actorLearningRate: number = 0.00005;
    public criticLearningRate: number = 0.002; // baseline is a simpler regression target, can usually move a bit faster
    public gamma: number = 0.99;

    constructor(inputSize: number) {
        const makeActor = () => [
            new DenseLayer(inputSize, 64),
            new ReLULayer(),
            new DenseLayer(64, 64),
            new ReLULayer(),
            new DenseLayer(64, 2), // [meanRaw, logStdRaw]
        ];
        const makeCritic = () => [
            new DenseLayer(inputSize, 64),
            new ReLULayer(),
            new DenseLayer(64, 64),
            new ReLULayer(),
            new DenseLayer(64, 1), // [V(s)]
        ];

        this.actor = new NeuralNetwork(makeActor());
        this.critic = new NeuralNetwork(makeCritic());
    }

    /**
     * Squashes the Actor's raw mean output into [-1, 1] with tanh (matching
     * the [-1, 1] control range your Actuator/THRUST_LEVELS already used),
     * and turns logStd into a positive, clamped std.
     */
    private decodeActorOutput(state: number[]): { meanRaw: number; mean: number; logStdRaw: number; std: number } {
        const [meanRaw, logStdRawUnclamped] = this.actor.predict(state);
        const logStdRaw = clamp(logStdRawUnclamped, LOG_STD_MIN, LOG_STD_MAX);
        const mean = Math.tanh(meanRaw);
        const std = Math.exp(logStdRaw);
        return { meanRaw, mean, logStdRaw, std };
    }

    /**
     * Samples an action for the given state. Returns everything the
     * EpisodeBuffer needs to later recompute logProb during learn().
     */
    public act(state: number[]): { rawAction: number; clampedAction: number; mean: number; std: number } {
        const { mean, std } = this.decodeActorOutput(state);
        const z = sampleStandardNormal();
        const rawAction = mean + std * z;

        // The environment only ever sees a control value in [-1, 1] — clamping
        // here is the same simplification DDPG-style noise injection makes:
        // the Gaussian can propose something slightly outside the valid range,
        // we clip what's actually sent to the Actuator, but train logProb
        // against the raw (unclamped) sample so the gradient math stays exact.
        const clampedAction = clamp(rawAction, -1, 1);

        return { rawAction, clampedAction, mean, std };
    }

    /** V(s) — exposed mainly for diagnostics/UI, same role getQValues() played for DQNAgent. */
    public getValue(state: number[]): number {
        return this.critic.predict(state)[0];
    }

    /**
     * The REINFORCE update. Called once per finished episode with the full
     * trajectory. Unlike DQNAgent.replay(), there's no batch sampling —
     * every step of the episode gets used, in order, exactly once.
     */
    public learn(episode: Transition[]): {
        actorLoss: number;
        criticLoss: number;
        avgAbsoluteAdvantage: number;
        gradientClipRate: number;
        avgEntropy: number
    } {
        // --- STEP 1: Compute discounted returns G_t, walking backward through the episode ---
        const returns: number[] = new Array(episode.length);
        let runningReturn = 0;
        for (let t = episode.length - 1; t >= 0; t--) {
            runningReturn = episode[t].reward + this.gamma * runningReturn;
            returns[t] = runningReturn;
        }

        let totalCriticLoss = 0;
        let totalActorLoss = 0;
        let totalAbsoluteAdvantage = 0;
        let totalEntropy = 0;
        let clipCount = 0;

        // --- STEP 1.5: Compute raw advantages first, THEN normalize across the
        // whole episode (subtract mean, divide by std). This is the standard
        // REINFORCE stabilizer I skipped in the first pass: without it, a
        // single spiky episode (e.g. one that hits your -20 terminal penalty
        // vs. one that survives with +1/step rewards) hands the Actor
        // wildly different-scale gradients episode to episode, which is
        // exactly the kind of thing that blows weights up into NaN.
        const rawAdvantages: number[] = new Array(episode.length);
        for (let t = 0; t < episode.length; t++) {
            const baseline = this.critic.predict(episode[t].state)[0];
            rawAdvantages[t] = returns[t] - baseline;
        }
        const advMean = rawAdvantages.reduce((a, b) => a + b, 0) / rawAdvantages.length;
        const advVariance = rawAdvantages.reduce((sum, a) => sum + (a - advMean) ** 2, 0) / rawAdvantages.length;
        const advStd = Math.sqrt(advVariance) + 1e-6; // epsilon guards against divide-by-zero on a dead-flat episode

        // Safety net matching the [-1, 1] clip DQN's Critic already gets for free
        // via NeuralNetwork.train() — the Actor's custom gradient has no such
        // built-in protection, since trainWithGradient() is designed to accept
        // ANY gradient. We clip here instead, at the call site.
        const GRADIENT_CLIP = 5.0;
        const clipGrad = (g: number) => {
            if (isNaN(g)) return 0;
            if (Math.abs(g) >= GRADIENT_CLIP) clipCount++;
            return clamp(g, -GRADIENT_CLIP, GRADIENT_CLIP);
        };

        // --- STEP 2: Walk forward, updating Critic then Actor at each step ---
        for (let t = 0; t < episode.length; t++) {
            const { state, rawAction } = episode[t];
            const G_t = returns[t];

            // Critic regression
            const criticLoss = this.critic.train(state, [G_t], this.criticLearningRate);
            totalCriticLoss += criticLoss;

            // TRACK ABSOLUTE ADVANTAGE
            totalAbsoluteAdvantage += Math.abs(rawAdvantages[t]);

            const normalizedAdvantage = (rawAdvantages[t] - advMean) / advStd;
            const { mean: liveMean, std: liveStd } = this.decodeActorOutput(state);

            // TRACK ENTROPY for this step
            totalEntropy += 0.5 * Math.log(2 * Math.PI * Math.E * liveStd * liveStd);

            // d(logProb)/d(mean) = (a - mean) / std^2
            const dLogProb_dMean = (rawAction - liveMean) / (liveStd * liveStd);
            // d(logProb)/d(logStdRaw) = (a - mean)^2/std^2 - 1   [std = exp(logStdRaw), chain rule folded in]
            const dLogProb_dLogStdRaw = ((rawAction - liveMean) ** 2) / (liveStd * liveStd) - 1;

            // Loss = -logProb(a|s) * Advantage  ->  dLoss/d(x) = -Advantage * dLogProb/d(x)
            const dLoss_dMean = -normalizedAdvantage * dLogProb_dMean;
            const dLoss_dLogStdRaw = -normalizedAdvantage * dLogProb_dLogStdRaw;

            // Chain through the tanh squashing applied to the raw mean output.
            const dMean_dMeanRaw = 1 - liveMean * liveMean; // d/dx[tanh(x)] = 1 - tanh(x)^2
            const dLoss_dMeanRaw = clipGrad(dLoss_dMean * dMean_dMeanRaw);
            const dLoss_dLogStdRawClipped = clipGrad(dLoss_dLogStdRaw);

            this.actor.trainWithGradient(state, [dLoss_dMeanRaw, dLoss_dLogStdRawClipped], this.actorLearningRate);

            totalActorLoss += Math.abs(dLoss_dMeanRaw) + Math.abs(dLoss_dLogStdRawClipped);
        }

        return {
            actorLoss: totalActorLoss / episode.length,
            criticLoss: totalCriticLoss / episode.length,
            avgAbsoluteAdvantage: totalAbsoluteAdvantage / episode.length,
            gradientClipRate: clipCount / (episode.length * 2), // * 2 because we clip mean and std gradients
            avgEntropy: totalEntropy / episode.length
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