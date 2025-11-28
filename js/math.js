/**
 * MHSA Visualizer - Math Utilities Module
 * Matrix operations and mathematical functions for attention computation
 */

export const MathUtils = {
    /**
     * Matrix multiplication: A @ B
     * @param {number[][]} A - First matrix
     * @param {number[][]} B - Second matrix
     * @returns {number[][]} Result matrix
     */
    matmul(A, B) {
        const rowsA = A.length;
        const colsA = A[0].length;
        const colsB = B[0].length;
        const C = [];

        for (let i = 0; i < rowsA; i++) {
            C[i] = [];
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    },

    /**
     * Transpose a matrix
     * @param {number[][]} A - Input matrix
     * @returns {number[][]} Transposed matrix
     */
    transpose(A) {
        const rows = A.length;
        const cols = A[0].length;
        const T = [];
        for (let j = 0; j < cols; j++) {
            T[j] = [];
            for (let i = 0; i < rows; i++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    },

    /**
     * Apply softmax row-wise with optional temperature
     * @param {number[][]} scores - Score matrix
     * @param {number} temperature - Temperature parameter (default 1.0)
     * @returns {number[][]} Softmax probabilities
     */
    softmax(scores, temperature = 1.0) {
        const result = [];
        for (let i = 0; i < scores.length; i++) {
            const row = scores[i];
            const maxScore = Math.max(...row);
            const expScores = row.map(s => Math.exp((s - maxScore) / temperature));
            const sumExp = expScores.reduce((a, b) => a + b, 0);
            result[i] = expScores.map(e => e / sumExp);
        }
        return result;
    },

    /**
     * Generate a random matrix with Xavier initialization
     * @param {number} rows - Number of rows
     * @param {number} cols - Number of columns
     * @returns {number[][]} Random matrix
     */
    randomMatrix(rows, cols) {
        const matrix = [];
        const scale = Math.sqrt(2.0 / (rows + cols)); // Xavier initialization
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = (Math.random() - 0.5) * 2 * scale;
            }
        }
        return matrix;
    },

    /**
     * Create a zero matrix
     * @param {number} rows - Number of rows
     * @param {number} cols - Number of columns
     * @returns {number[][]} Zero matrix
     */
    zeros(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = new Array(cols).fill(0);
        }
        return matrix;
    },

    /**
     * Element-wise addition of two matrices
     * @param {number[][]} A - First matrix
     * @param {number[][]} B - Second matrix
     * @returns {number[][]} Sum matrix
     */
    add(A, B) {
        const result = [];
        for (let i = 0; i < A.length; i++) {
            result[i] = [];
            for (let j = 0; j < A[0].length; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        return result;
    },

    /**
     * Scale a matrix by a scalar
     * @param {number[][]} A - Input matrix
     * @param {number} scalar - Scalar value
     * @returns {number[][]} Scaled matrix
     */
    scale(A, scalar) {
        return A.map(row => row.map(val => val * scalar));
    },

    /**
     * Compute dot product of two vectors
     * @param {number[]} a - First vector
     * @param {number[]} b - Second vector
     * @returns {number} Dot product
     */
    dot(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    },

    /**
     * Compute L2 norm of a vector
     * @param {number[]} v - Input vector
     * @returns {number} L2 norm
     */
    norm(v) {
        return Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    },

    /**
     * Normalize a vector to unit length
     * @param {number[]} v - Input vector
     * @returns {number[]} Normalized vector
     */
    normalize(v) {
        const n = this.norm(v);
        return n > 0 ? v.map(val => val / n) : v;
    },

    /**
     * Compute cosine similarity between two vectors
     * @param {number[]} a - First vector
     * @param {number[]} b - Second vector
     * @returns {number} Cosine similarity
     */
    cosineSimilarity(a, b) {
        const dotProduct = this.dot(a, b);
        const normA = this.norm(a);
        const normB = this.norm(b);
        return normA > 0 && normB > 0 ? dotProduct / (normA * normB) : 0;
    },

    /**
     * Average multiple attention matrices
     * @param {number[][][]} attentions - Array of attention matrices
     * @returns {number[][]} Averaged attention matrix
     */
    averageAttentions(attentions) {
        const numHeads = attentions.length;
        const seqLen = attentions[0].length;
        const avg = [];

        for (let i = 0; i < seqLen; i++) {
            avg[i] = [];
            for (let j = 0; j < seqLen; j++) {
                let sum = 0;
                for (let h = 0; h < numHeads; h++) {
                    sum += attentions[h][i][j];
                }
                avg[i][j] = sum / numHeads;
            }
        }
        return avg;
    },

    /**
     * Concatenate arrays horizontally (along columns)
     * @param {...number[][]} matrices - Matrices to concatenate
     * @returns {number[][]} Concatenated matrix
     */
    concatenateHorizontal(...matrices) {
        const numRows = matrices[0].length;
        const result = [];
        
        for (let i = 0; i < numRows; i++) {
            result[i] = [];
            for (const matrix of matrices) {
                result[i].push(...matrix[i]);
            }
        }
        return result;
    },

    /**
     * Get a slice of a matrix (row range)
     * @param {number[][]} matrix - Input matrix
     * @param {number} startRow - Start row index
     * @param {number} endRow - End row index
     * @returns {number[][]} Sliced matrix
     */
    sliceRows(matrix, startRow, endRow) {
        return matrix.slice(startRow, endRow);
    },

    /**
     * Get a slice of a matrix (column range)
     * @param {number[][]} matrix - Input matrix
     * @param {number} startCol - Start column index
     * @param {number} endCol - End column index
     * @returns {number[][]} Sliced matrix
     */
    sliceCols(matrix, startCol, endCol) {
        return matrix.map(row => row.slice(startCol, endCol));
    }
};

export default MathUtils;

