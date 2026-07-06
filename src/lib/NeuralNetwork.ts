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
}