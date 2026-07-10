export class Matrix {
    public rows: number;
    public cols: number;
    public data: Float32Array;

    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        // Float32Array initializes all values to 0 automatically.
        // It's a continuous block of memory, making it incredibly fast for CPU cache.
        this.data = new Float32Array(rows * cols);
    }

    // ==========================================
    // INITIALIZATION & FACTORY METHODS
    // ==========================================

    /**
     * Creates a Matrix filled with 0s.
     */
    static zeros(rows: number, cols: number): Matrix {
        return new Matrix(rows, cols);
    }

    /**
     * Creates a Matrix filled with random numbers drawn from a 
     * Standard Normal Distribution (mean 0, variance 1) using the Box-Muller transform.
     */
    static randn(rows: number, cols: number): Matrix {
        const result = new Matrix(rows, cols);
        for (let i = 0; i < result.data.length; i++) {
            let u = 0, v = 0;
            while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
            while (v === 0) v = Math.random();
            const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            result.data[i] = num;
        }
        return result;
    }

    /**
     * Converts a standard 2D JavaScript array into our highly-optimized Matrix.
     */
    static fromArray(arr: number[][]): Matrix {
        const rows = arr.length;
        const cols = arr[0].length;
        const result = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result.data[i * cols + j] = arr[i][j];
            }
        }
        return result;
    }

    /**
     * Copies values from a plain array into this Matrix's existing buffer,
     * in place. Used to feed a reusable scratch Matrix from a fresh
     * `number[]` each call without allocating a new Matrix every time.
     */
    public setFromArray(arr: number[] | Float32Array): this {
        for (let i = 0; i < arr.length; i++) this.data[i] = arr[i];
        return this;
    }

    /**
     * Converts the Matrix back to a 2D array (useful for debugging or interfacing).
     */
    public toArray(): number[][] {
        const arr: number[][] = [];
        for (let i = 0; i < this.rows; i++) {
            const row: number[] = [];
            for (let j = 0; j < this.cols; j++) {
                row.push(this.data[i * this.cols + j]);
            }
            arr.push(row);
        }
        return arr;
    }

    // ==========================================
    // MATRIX OPERATIONS (Returns a new Matrix)
    // ==========================================

    /**
     * Standard Matrix Multiplication (Dot Product).
     * O(n^3) complexity, optimized for 1D array indexing.
     */
    static dot(a: Matrix, b: Matrix): Matrix {
        if (a.cols !== b.rows) {
            throw new Error(`Matrix mismatch: Columns of A (${a.cols}) must match Rows of B (${b.rows})`);
        }
        return Matrix.dotInto(a, b, new Matrix(a.rows, b.cols));
    }

    /**
     * Same as dot(), but writes into a caller-supplied `out` Matrix instead of
     * allocating a new one. Training calls this thousands of times per learn()
     * pass, so reusing a pre-sized scratch buffer here avoids a matching flood
     * of Float32Array allocations (and the GC pauses that come with them).
     */
    static dotInto(a: Matrix, b: Matrix, out: Matrix): Matrix {
        if (a.cols !== b.rows) {
            throw new Error(`Matrix mismatch: Columns of A (${a.cols}) must match Rows of B (${b.rows})`);
        }
        if (out.rows !== a.rows || out.cols !== b.cols) {
            throw new Error(`dotInto: output matrix is ${out.rows}x${out.cols}, expected ${a.rows}x${b.cols}`);
        }
        // Hoisted out of the inner loops: `i * a.cols` and `i * b.cols` were
        // being recomputed on every (i, j) pair, and `k * b.cols` was being
        // recomputed on every single (i, j, k) triple — a multiply this hot
        // loop runs millions of times per learn() call is worth avoiding.
        const aCols = a.cols, bCols = b.cols;
        for (let i = 0; i < a.rows; i++) {
            const aRowOffset = i * aCols;
            const outRowOffset = i * bCols;
            for (let j = 0; j < bCols; j++) {
                let sum = 0;
                let bIdx = j;
                for (let k = 0; k < aCols; k++) {
                    sum += a.data[aRowOffset + k] * b.data[bIdx];
                    bIdx += bCols;
                }
                out.data[outRowOffset + j] = sum;
            }
        }
        return out;
    }

    /**
     * Flips rows and columns. Critical for calculating gradients during Backpropagation.
     */
    public transpose(): Matrix {
        return this.transposeInto(new Matrix(this.cols, this.rows));
    }

    /**
     * Same as transpose(), but writes into a caller-supplied `out` Matrix
     * instead of allocating a new one. See dotInto() for why this matters.
     */
    public transposeInto(out: Matrix): Matrix {
        if (out.rows !== this.cols || out.cols !== this.rows) {
            throw new Error(`transposeInto: output matrix is ${out.rows}x${out.cols}, expected ${this.cols}x${this.rows}`);
        }
        const rows = this.rows, cols = this.cols;
        for (let i = 0; i < rows; i++) {
            const rowOffset = i * cols;
            let outIdx = i;
            for (let j = 0; j < cols; j++) {
                // Read from [i, j], write to [j, i]
                out.data[outIdx] = this.data[rowOffset + j];
                outIdx += rows;
            }
        }
        return out;
    }

    // ==========================================
    // ELEMENT-WISE OPERATIONS (In-Place for speed)
    // ==========================================

    public add(n: Matrix | number): this {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) throw new Error("Matrix dimensions must match for addition");
            for (let i = 0; i < this.data.length; i++) this.data[i] += n.data[i];
        } else {
            for (let i = 0; i < this.data.length; i++) this.data[i] += n;
        }
        return this; // Return 'this' to allow method chaining (e.g., m.add(1).mult(2))
    }

    public sub(n: Matrix | number): this {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) throw new Error("Matrix dimensions must match for subtraction");
            for (let i = 0; i < this.data.length; i++) this.data[i] -= n.data[i];
        } else {
            for (let i = 0; i < this.data.length; i++) this.data[i] -= n;
        }
        return this;
    }

    /**
     * Adds a (rows, 1) column vector to every column of this matrix, in place.
     * This is bias-broadcast for batched layers: a Dense layer's bias is one
     * value per output neuron, added identically to every sample in the batch.
     */
    public addBroadcastColumn(bias: Matrix): this {
        if (bias.rows !== this.rows || bias.cols !== 1) {
            throw new Error(`addBroadcastColumn: bias is ${bias.rows}x${bias.cols}, expected ${this.rows}x1`);
        }
        for (let i = 0; i < this.rows; i++) {
            const b = bias.data[i];
            const rowOffset = i * this.cols;
            for (let j = 0; j < this.cols; j++) {
                this.data[rowOffset + j] += b;
            }
        }
        return this;
    }

    /**
     * Sums each row across all columns into a caller-supplied (rows, 1) `out`
     * Matrix. Used to turn a batched output-gradient (one column per sample)
     * into the single bias gradient shared by every sample in that batch.
     */
    public sumRowsInto(out: Matrix): Matrix {
        if (out.rows !== this.rows || out.cols !== 1) {
            throw new Error(`sumRowsInto: output matrix is ${out.rows}x${out.cols}, expected ${this.rows}x1`);
        }
        for (let i = 0; i < this.rows; i++) {
            let sum = 0;
            const rowOffset = i * this.cols;
            for (let j = 0; j < this.cols; j++) sum += this.data[rowOffset + j];
            out.data[i] = sum;
        }
        return out;
    }

    public mult(n: Matrix | number): this {
        if (n instanceof Matrix) {
            // Hadamard Product (Element-wise multiplication)
            if (this.rows !== n.rows || this.cols !== n.cols) throw new Error("Matrix dimensions must match for Hadamard product");
            for (let i = 0; i < this.data.length; i++) this.data[i] *= n.data[i];
        } else {
            // Scalar multiplication
            for (let i = 0; i < this.data.length; i++) this.data[i] *= n;
        }
        return this;
    }

    // ==========================================
    // FUNCTIONAL OPERATIONS
    // ==========================================

    /**
     * Applies a function to every element in the matrix in-place.
     * Perfect for Activation Functions (ReLU, Sigmoid).
     */
    public map(fn: (val: number, index: number) => number): this {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = fn(this.data[i], i);
        }
        return this;
    }
}