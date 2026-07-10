import { Matrix } from './Matrix';

/**
 * The fundamental contract for all Neural Network layers.
 * Whether it's a Dense layer holding weights and biases, or an Activation layer 
 * just applying a math function, it must implement these two methods.
 */
export interface Layer {
    /**
     * Takes an input Matrix, processes it, and returns the output Matrix.
     * Crucially, the layer must internally save whatever state it needs (like the input)
     * so it can calculate the gradients properly during the backward pass.
     */
    forward(input: Matrix): Matrix;

    /**
     * Takes the gradient of the loss with respect to this layer's output.
     * Uses Calculus to:
     * 1. Update its own internal parameters (Weights/Biases) using the learningRate.
     * 2. Return the gradient of the loss with respect to this layer's input (to pass back to the previous layer).
     */
    backward(outputGradient: Matrix, learningRate: number): Matrix;

    /**
     * Optional batched variants: process a (features, batchWidth) matrix — one
     * column per sample — as a handful of wide matmuls instead of looping
     * forward()/backward() once per sample. Only worth implementing for
     * layers with learnable parameters whose gradient needs to be SUMMED
     * across the batch (e.g. DenseLayer); parameter-free layers like
     * ReLULayer are already shape-agnostic and can be fed a batch straight
     * through forward()/backward() with no changes. NeuralNetwork falls back
     * to the single-sample methods for any layer that doesn't implement these.
     */
    forwardBatch?(input: Matrix): Matrix;
    backwardBatch?(outputGradient: Matrix, learningRate: number): Matrix;
}