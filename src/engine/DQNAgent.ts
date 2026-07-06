import { DenseLayer } from "../lib/DenseLayer";
import { NeuralNetwork } from "../lib/NeuralNetwork";
import { ReLULayer } from "../lib/ReLULayer";
import { ReplayBuffer } from "../lib/ReplayBuffer";


export class DQNAgent {
    public brain: NeuralNetwork;
    public targetBrain: NeuralNetwork; // THE STABILIZER
    public memory: ReplayBuffer;

    // The Personality (Hyperparameters)
    public epsilon: number = 1.0;
    public epsilonDecay: number = 0.9995;
    public epsilonMin: number = 0.01;
    public gamma: number = 0.99;
    public learningRate: number = 0.0005;
    public batchSize: number = 32;

    private numActions: number;
    private trainSteps: number = 0;
    public targetUpdateFreq: number = 200;  // Sync the Target Brain every 200 steps

    constructor(inputSize: number, numActions: number) {
        this.numActions = numActions;
        this.memory = new ReplayBuffer(50000);

        // Upgraded Architecture: 128 Neurons per layer
        const createArchitecture = () => [
            new DenseLayer(inputSize, 64),
            new ReLULayer(),
            new DenseLayer(64, 64),
            new ReLULayer(),
            new DenseLayer(64, numActions)
        ];

        this.brain = new NeuralNetwork(createArchitecture());
        this.targetBrain = new NeuralNetwork(createArchitecture());

        // Initialize Target Brain
        this.syncTargetNetwork();
    }

    /**
     * Deep copies the weights from the Online Brain to the Target Brain
     */
    public syncTargetNetwork(): void {
        for (let i = 0; i < this.brain.layers.length; i++) {
            const onlineLayer = this.brain.layers[i] as any;
            const targetLayer = this.targetBrain.layers[i] as any;

            if (onlineLayer.weights && targetLayer.weights) {
                for (let j = 0; j < onlineLayer.weights.data.length; j++) {
                    targetLayer.weights.data[j] = onlineLayer.weights.data[j];
                }
                for (let j = 0; j < onlineLayer.biases.data.length; j++) {
                    targetLayer.biases.data[j] = onlineLayer.biases.data[j];
                }
            }
        }
    }

    public getAction(state: number[]): number {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        }

        const qValues = this.brain.predict(state);

        let bestAction = 0;
        let maxQ = qValues[0];
        for (let i = 1; i < this.numActions; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    /**
     * NEW: Expose raw Q-values for every action, not just the argmax.
     * Purely for diagnostics/UI — lets you draw a bar per thrust level
     * and see the agent's actual preference spread, not just its final pick.
     */
    public getQValues(state: number[]): number[] {
        return this.brain.predict(state);
    }

    public remember(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
        this.memory.add(state, action, reward, nextState, done);
    }

    public replay(): { loss: number, qValue: number } | null {
        if (!this.memory.isReady(this.batchSize)) {
            return null;
        }

        const batch = this.memory.sample(this.batchSize);
        let totalLoss = 0;
        let totalTargetQ = 0;

        for (const memory of batch) {
            const targetQ = this.brain.predict(memory.state);

            if (memory.done) {
                targetQ[memory.action] = memory.reward;
                totalTargetQ += memory.reward;
            } else {
                const nextQ = this.targetBrain.predict(memory.nextState);
                const maxFutureQ = Math.max(...nextQ);

                targetQ[memory.action] = memory.reward + (this.gamma * maxFutureQ);
                totalTargetQ += targetQ[memory.action];
            }

            totalLoss += this.brain.train(memory.state, targetQ, this.learningRate);
        }

        this.trainSteps++;
        if (this.trainSteps % this.targetUpdateFreq === 0) {
            this.syncTargetNetwork();
        }

        return {
            loss: totalLoss / this.batchSize,
            qValue: totalTargetQ / this.batchSize
        };
    }

    public decayEpsilon(): void {
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }


    /**
     * Extracts the raw array data from the network layers to save as a JSON string
     */
    public toJSON(): string {
        const layersData = this.brain.layers.map(layer => {
            const l = layer as any;
            if (l.weights && l.biases) {
                // Float32Arrays must be converted to standard arrays for JSON.stringify
                return {
                    weights: Array.from(l.weights.data),
                    biases: Array.from(l.biases.data)
                };
            }
            return null; // ReLULayers have no weights, so they return null
        });

        return JSON.stringify({
            layers: layersData,
            epsilon: this.epsilon
        });
    }

    /**
     * Injects saved data back into the network and synchronizes the target brain
     */
    public loadJSON(jsonString: string): void {
        const parsed = JSON.parse(jsonString);

        for (let i = 0; i < this.brain.layers.length; i++) {
            const l = this.brain.layers[i] as any;
            const data = parsed.layers[i];

            if (l.weights && l.biases && data) {
                for (let j = 0; j < data.weights.length; j++) l.weights.data[j] = data.weights[j];
                for (let j = 0; j < data.biases.length; j++) l.biases.data[j] = data.biases[j];
            }
        }

        if (parsed.epsilon !== undefined) this.epsilon = parsed.epsilon;

        this.syncTargetNetwork(); // Very important to ensure both brains align!
    }
}