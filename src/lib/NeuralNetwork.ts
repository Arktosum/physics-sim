import { Matrix } from './Matrix';
import type { Layer } from './Layer';

export class NeuralNetwork {
    public layers: Layer[];

    constructor(layers: Layer[]) {
        this.layers = layers;
    }

    /**
     * FORWARD PASS
     * Takes an array of raw numbers (e.g. physics state), passes it through every layer,
     * and returns the final prediction as an array of numbers (e.g. Q-Values).
     */
    public predict(inputArray: number[]): number[] {
        let currentData = new Matrix(inputArray.length, 1);
        for (let i = 0; i < inputArray.length; i++) {
            currentData.data[i] = inputArray[i];
        }

        for (const layer of this.layers) {
            currentData = layer.forward(currentData);
        }

        return Array.from(currentData.data);
    }

    /**
     * BACKWARD PASS (The Learning Step)
     * Returns the Mean Squared Error (Loss) for diagnostic tracking.
     */
    public train(inputArray: number[], targetArray: number[], learningRate: number): number {
        // --- STEP 1: Forward Pass ---
        let currentData = new Matrix(inputArray.length, 1);
        for (let i = 0; i < inputArray.length; i++) {
            currentData.data[i] = inputArray[i];
        }

        for (const layer of this.layers) {
            currentData = layer.forward(currentData);
        }

        // --- STEP 2: Calculate the Initial Error Gradient & Loss ---
        let gradient = new Matrix(targetArray.length, 1);
        let totalError = 0;

        for (let i = 0; i < targetArray.length; i++) {
            const error = currentData.data[i] - targetArray[i];

            // THE FIX: Gradient Clipping (Huber Loss approximation)
            // If the agent dies (-10 penalty), the error is massive. We clip the 
            // gradient to [-1, 1] so it takes a controlled step instead of exploding.
            gradient.data[i] = Math.max(-1, Math.min(1, error));

            totalError += error * error; // We keep true MSE for the UI chart
        }

        const mseLoss = totalError / targetArray.length;

        // --- STEP 3: Backward Pass ---
        for (let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backward(gradient, learningRate);
        }

        return mseLoss;
    }

    /**
     * BACKWARD PASS WITH A CUSTOM GRADIENT (for Actor networks)
     *
     * DQN's Critic/baseline networks learn by regression toward a target array,
     * so `train()`'s built-in "prediction - target" MSE gradient is exactly right
     * for them. An Actor in DDPG/REINFORCE/PPO does NOT learn by regression —
     * DDPG needs dQ/da chained back through the Actor, REINFORCE/PPO need
     * -logProb(a|s) * Advantage. Both are just "some gradient w.r.t. the
     * network's output", so this method does the forward pass, skips the MSE
     * step entirely, and backprops whatever gradient the caller computed.
     */
    public trainWithGradient(inputArray: number[], outputGradient: number[], learningRate: number): void {
        // --- STEP 1: Forward Pass (must run first so each layer caches its input) ---
        let currentData = new Matrix(inputArray.length, 1);
        for (let i = 0; i < inputArray.length; i++) {
            currentData.data[i] = inputArray[i];
        }

        for (const layer of this.layers) {
            currentData = layer.forward(currentData);
        }

        // --- STEP 2: Load the caller's gradient directly, no MSE/clipping involved ---
        let gradient = new Matrix(outputGradient.length, 1);
        for (let i = 0; i < outputGradient.length; i++) {
            gradient.data[i] = outputGradient[i];
        }

        // --- STEP 3: Backward Pass, identical mechanics to train() ---
        for (let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backward(gradient, learningRate);
        }
    }
}