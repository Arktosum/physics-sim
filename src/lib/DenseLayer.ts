import { Matrix } from './Matrix';
import type { Layer } from './Layer';

export class DenseLayer implements Layer {
    private weights: Matrix;
    private biases: Matrix;
    private input: Matrix | null = null;

    // Scratch buffers, sized once here and reused on every forward()/backward()
    // call. A single learn() call runs epochs * horizonLength samples through
    // every layer (thousands of times), so allocating fresh Matrices per call
    // was flooding the GC with tiny short-lived Float32Arrays — this trades
    // that away for a handful of buffers that live for the layer's lifetime.
    private readonly outputScratch: Matrix;
    private readonly weightsGradientScratch: Matrix;
    private readonly inputGradientScratch: Matrix;
    private readonly inputTransposeScratch: Matrix;
    private readonly weightsTransposeScratch: Matrix;

    // Batch-shaped scratch — sizes depend on batchWidth, so unlike the
    // buffers above these are sized lazily on first batched call (batchWidth
    // isn't known until then) and only reallocated if that width changes.
    // weightsGradientScratch/weightsTransposeScratch above are reused as-is:
    // a weight gradient is (outputSize, inputSize) regardless of batch width
    // (matmul contracts/sums over the batch dimension), and W^T is always
    // (inputSize, outputSize) — neither depends on batchWidth.
    private batchInput: Matrix | null = null;
    private outputBatchScratch: Matrix | null = null;
    private inputTransposeBatchScratch: Matrix | null = null;
    private inputGradientBatchScratch: Matrix | null = null;
    private biasGradientScratch: Matrix | null = null;

    constructor(inputSize: number, outputSize: number) {
        // He Initialization: Random normal values scaled by sqrt(2 / inputSize).
        // This stops our gradients from vanishing into 0 or exploding into Infinity.
        this.weights = Matrix.randn(outputSize, inputSize).mult(Math.sqrt(2 / inputSize));

        // Biases safely start at 0
        this.biases = Matrix.zeros(outputSize, 1);

        this.outputScratch = new Matrix(outputSize, 1);
        this.weightsGradientScratch = new Matrix(outputSize, inputSize);
        this.inputGradientScratch = new Matrix(inputSize, 1);
        this.inputTransposeScratch = new Matrix(1, inputSize);
        this.weightsTransposeScratch = new Matrix(inputSize, outputSize);
    }

    /**
     * FORWARD PASS
     * Math: Output = (Weights • Input) + Biases
     */
    forward(input: Matrix): Matrix {
        // Save the input so we can use it during Backpropagation later
        this.input = input;

        Matrix.dotInto(this.weights, input, this.outputScratch);
        return this.outputScratch.add(this.biases);
    }

    /**
     * BACKWARD PASS (The Calculus)
     * Takes the error gradient from the layer ahead of it, updates its own weights/biases,
     * and passes the remaining error gradient backward to the layer behind it.
     */
    backward(outputGradient: Matrix, learningRate: number): Matrix {
        if (!this.input) {
            throw new Error("Must call forward() before backward() can calculate gradients.");
        }

        // 1. Calculate Weight Gradients: dW = outputGradient • input^T
        this.input.transposeInto(this.inputTransposeScratch);
        Matrix.dotInto(outputGradient, this.inputTransposeScratch, this.weightsGradientScratch);

        // 2. Calculate Input Gradients to pass backward: dX = W^T • outputGradient
        // (We must calculate this BEFORE we mutate the weights in the next step!)
        this.weights.transposeInto(this.weightsTransposeScratch);
        Matrix.dotInto(this.weightsTransposeScratch, outputGradient, this.inputGradientScratch);

        // 3. Update Weights: W = W - (dW * learningRate)
        this.weightsGradientScratch.mult(learningRate);
        this.weights.sub(this.weightsGradientScratch);

        // 4. Update Biases: B = B - (dB * learningRate)
        // The gradient of the bias is exactly equal to the outputGradient.
        // We do this in a raw loop for maximum CPU cache performance.
        for (let i = 0; i < this.biases.data.length; i++) {
            this.biases.data[i] -= outputGradient.data[i] * learningRate;
        }

        // Return the input gradient so the layer behind us can do its own backprop
        return this.inputGradientScratch;
    }

    /**
     * Batched forward: input is (inputSize, batchWidth) — one column per
     * sample. Same math as forward(), except the bias add has to broadcast
     * across every column instead of just adding two same-shaped matrices.
     */
    forwardBatch(input: Matrix): Matrix {
        this.batchInput = input;
        const batchWidth = input.cols;

        if (!this.outputBatchScratch || this.outputBatchScratch.cols !== batchWidth) {
            this.outputBatchScratch = new Matrix(this.weights.rows, batchWidth);
        }

        Matrix.dotInto(this.weights, input, this.outputBatchScratch);
        return this.outputBatchScratch.addBroadcastColumn(this.biases);
    }

    /**
     * Batched backward: outputGradient is (outputSize, batchWidth). The
     * weight-gradient matmul (outputGradient • inputᵀ) sums over the batch
     * dimension automatically — that's what makes this a mini-batch update
     * rather than batchWidth sequential single-sample updates — so we divide
     * by batchWidth before applying it, turning the sum into a mean (standard
     * mini-batch SGD; matches what learningRate meant in the single-sample path).
     */
    backwardBatch(outputGradient: Matrix, learningRate: number): Matrix {
        if (!this.batchInput) {
            throw new Error("Must call forwardBatch() before backwardBatch() can calculate gradients.");
        }

        const batchWidth = outputGradient.cols;
        const inputSize = this.weights.cols;
        const outputSize = this.weights.rows;

        if (!this.inputTransposeBatchScratch || this.inputTransposeBatchScratch.rows !== batchWidth) {
            this.inputTransposeBatchScratch = new Matrix(batchWidth, inputSize);
        }
        this.batchInput.transposeInto(this.inputTransposeBatchScratch);

        // dW = outputGradient • inputᵀ — shape (outputSize, inputSize), same as
        // the single-sample case, so the existing scratch buffer is reusable.
        Matrix.dotInto(outputGradient, this.inputTransposeBatchScratch, this.weightsGradientScratch);

        if (!this.inputGradientBatchScratch || this.inputGradientBatchScratch.cols !== batchWidth) {
            this.inputGradientBatchScratch = new Matrix(inputSize, batchWidth);
        }
        this.weights.transposeInto(this.weightsTransposeScratch);
        Matrix.dotInto(this.weightsTransposeScratch, outputGradient, this.inputGradientBatchScratch);

        this.weightsGradientScratch.mult(learningRate / batchWidth);
        this.weights.sub(this.weightsGradientScratch);

        if (!this.biasGradientScratch) this.biasGradientScratch = new Matrix(outputSize, 1);
        outputGradient.sumRowsInto(this.biasGradientScratch);
        for (let i = 0; i < outputSize; i++) {
            this.biases.data[i] -= (this.biasGradientScratch.data[i] / batchWidth) * learningRate;
        }

        return this.inputGradientBatchScratch;
    }
}