import { Matrix } from './Matrix';
import type { Layer } from './Layer';

export class NeuralNetwork {
    public layers: Layer[];

    // Reused across predict()/train()/trainWithGradient() calls instead of
    // allocating a fresh input/gradient Matrix every time. Sizes are fixed by
    // this network's architecture, so once these are sized on first use they
    // never need to change — same reasoning as the per-layer scratch buffers.
    private inputScratch: Matrix | null = null;
    private targetScratch: Matrix | null = null;

    constructor(layers: Layer[]) {
        this.layers = layers;
    }

    private getInputScratch(size: number): Matrix {
        if (!this.inputScratch || this.inputScratch.rows !== size) {
            this.inputScratch = new Matrix(size, 1);
        }
        return this.inputScratch;
    }

    private getTargetScratch(size: number): Matrix {
        if (!this.targetScratch || this.targetScratch.rows !== size) {
            this.targetScratch = new Matrix(size, 1);
        }
        return this.targetScratch;
    }

    /**
     * FORWARD PASS
     * Takes an array of raw numbers (e.g. physics state), passes it through every layer,
     * and returns the final prediction as an array of numbers (e.g. Q-Values).
     */
    public predict(inputArray: number[]): number[] {
        let currentData: Matrix = this.getInputScratch(inputArray.length).setFromArray(inputArray);

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
        let currentData: Matrix = this.getInputScratch(inputArray.length).setFromArray(inputArray);

        for (const layer of this.layers) {
            currentData = layer.forward(currentData);
        }

        // --- STEP 2: Calculate the Initial Error Gradient & Loss ---
        const gradientBuffer = this.getTargetScratch(targetArray.length);
        let gradient: Matrix = gradientBuffer;
        let totalError = 0;

        for (let i = 0; i < targetArray.length; i++) {
            const error = currentData.data[i] - targetArray[i];

            // THE FIX: Gradient Clipping (Huber Loss approximation)
            // If the agent dies (-10 penalty), the error is massive. We clip the
            // gradient to [-1, 1] so it takes a controlled step instead of exploding.
            gradientBuffer.data[i] = Math.max(-1, Math.min(1, error));

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
        let currentData: Matrix = this.getInputScratch(inputArray.length).setFromArray(inputArray);

        for (const layer of this.layers) {
            currentData = layer.forward(currentData);
        }

        this.backwardWithGradient(outputGradient, learningRate);
    }

    /**
     * Same backward pass as trainWithGradient(), but skips the forward pass
     * entirely. ONLY safe to call when forward() (via predict()/train()/
     * trainWithGradient()) already ran for this exact input as the very last
     * thing done to this network — each layer's backward() reads its own
     * cached `input`/output from that forward call, and this method doesn't
     * re-populate them.
     *
     * PPOAgent.learn() calls decodeActorOutput() (a predict()) to compute
     * mean/std for a sample, then immediately backprops a gradient derived
     * from that same mean/std for that same sample — trainWithGradient()'s
     * own forward pass in that path was recomputing something we'd already
     * just computed a few lines earlier. This is that skip.
     */
    public backwardWithGradient(outputGradient: number[], learningRate: number): void {
        let gradient: Matrix = this.getTargetScratch(outputGradient.length).setFromArray(outputGradient);

        for (let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backward(gradient, learningRate);
        }
    }

    /**
     * Batched forward pass: inputBatch is (inputSize, batchWidth) — one
     * column per sample. Layers that implement forwardBatch() (DenseLayer)
     * use it; parameter-free layers (ReLULayer) run their normal forward()
     * unchanged since it's already shape-agnostic.
     */
    public predictBatch(inputBatch: Matrix): Matrix {
        let currentData: Matrix = inputBatch;
        for (const layer of this.layers) {
            currentData = layer.forwardBatch ? layer.forwardBatch(currentData) : layer.forward(currentData);
        }
        return currentData;
    }

    /**
     * Batched backward pass paired with predictBatch(): outputGradient is
     * (outputSize, batchWidth). Same fallback rule as predictBatch() for
     * layers without a batched implementation.
     */
    public backwardBatchWithGradient(outputGradient: Matrix, learningRate: number): void {
        let gradient: Matrix = outputGradient;
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            gradient = layer.backwardBatch ? layer.backwardBatch(gradient, learningRate) : layer.backward(gradient, learningRate);
        }
    }
}