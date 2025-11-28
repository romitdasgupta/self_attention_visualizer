/**
 * MHSA Visualizer - Multi-Head Self-Attention Module
 * Core MHSA implementation with support for both random and real model weights
 */

import { MathUtils } from './math.js';

/**
 * Multi-Head Self-Attention class
 * Implements the attention mechanism from "Attention is All You Need"
 */
export class MultiHeadSelfAttention {
    /**
     * Create a new MHSA instance
     * @param {number} embedDim - Embedding dimension (d_model)
     * @param {number} numHeads - Number of attention heads
     * @param {number} temperature - Temperature for softmax (default 1.0)
     */
    constructor(embedDim, numHeads, temperature = 1.0) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = Math.floor(embedDim / numHeads);
        this.temperature = temperature;
        this.heads = [];
        this.Wo = null;
        this.isRealModel = false;
        this.modelName = null;
    }

    /**
     * Initialize with random weights (Xavier initialization)
     */
    initializeRandomWeights() {
        this.heads = [];
        for (let h = 0; h < this.numHeads; h++) {
            this.heads.push({
                Wq: MathUtils.randomMatrix(this.embedDim, this.headDim),
                Wk: MathUtils.randomMatrix(this.embedDim, this.headDim),
                Wv: MathUtils.randomMatrix(this.embedDim, this.headDim)
            });
        }
        // Output projection
        this.Wo = MathUtils.randomMatrix(this.numHeads * this.headDim, this.embedDim);
        this.isRealModel = false;
        this.modelName = 'Random Initialization';
    }

    /**
     * Load weights from a real model
     * @param {Object} weights - Weight matrices from a transformer model
     * @param {string} modelName - Name of the source model
     */
    loadRealWeights(weights, modelName) {
        if (!weights || !weights.heads || !weights.Wo) {
            throw new Error('Invalid weight structure');
        }

        this.numHeads = weights.heads.length;
        this.headDim = weights.heads[0].Wq[0].length;
        this.embedDim = weights.heads[0].Wq.length;
        
        this.heads = weights.heads.map(head => ({
            Wq: head.Wq,
            Wk: head.Wk,
            Wv: head.Wv
        }));
        
        this.Wo = weights.Wo;
        this.isRealModel = true;
        this.modelName = modelName;
    }

    /**
     * Compute scaled dot-product attention
     * @param {number[][]} Q - Query matrix
     * @param {number[][]} K - Key matrix
     * @param {number[][]} V - Value matrix
     * @returns {Object} Attention output and intermediate values
     */
    scaledDotProductAttention(Q, K, V) {
        // Compute attention scores: Q @ K^T
        const KT = MathUtils.transpose(K);
        const scores = MathUtils.matmul(Q, KT);

        // Scale by sqrt(d_k)
        const scale = Math.sqrt(this.headDim);
        const scaledScores = scores.map(row =>
            row.map(val => val / scale)
        );

        // Apply softmax with temperature
        const attentionWeights = MathUtils.softmax(scaledScores, this.temperature);

        // Apply attention to values: attention @ V
        const output = MathUtils.matmul(attentionWeights, V);

        return {
            output,
            attentionWeights,
            scores: scaledScores
        };
    }

    /**
     * Forward pass through multi-head self-attention
     * @param {number[][]} embeddings - Input embeddings [seqLen x embedDim]
     * @returns {Object} Output and all intermediate computations
     */
    forward(embeddings) {
        const seqLen = embeddings.length;
        const headOutputs = [];
        const headAttentions = [];

        // Process each head
        for (let h = 0; h < this.numHeads; h++) {
            const head = this.heads[h];

            // Project to Q, K, V
            const Q = MathUtils.matmul(embeddings, head.Wq);
            const K = MathUtils.matmul(embeddings, head.Wk);
            const V = MathUtils.matmul(embeddings, head.Wv);

            // Compute attention
            const { output, attentionWeights, scores } =
                this.scaledDotProductAttention(Q, K, V);

            headOutputs.push({
                Q, K, V, output, attentionWeights, scores
            });
            headAttentions.push(attentionWeights);
        }

        // Concatenate heads
        const concatenated = [];
        for (let i = 0; i < seqLen; i++) {
            concatenated[i] = [];
            for (let h = 0; h < this.numHeads; h++) {
                concatenated[i].push(...headOutputs[h].output[i]);
            }
        }

        // Final linear projection
        const finalOutput = MathUtils.matmul(concatenated, this.Wo);

        return {
            output: finalOutput,
            headOutputs,
            headAttentions,
            concatenated
        };
    }

    /**
     * Get model information
     * @returns {Object} Model configuration details
     */
    getInfo() {
        return {
            embedDim: this.embedDim,
            numHeads: this.numHeads,
            headDim: this.headDim,
            temperature: this.temperature,
            isRealModel: this.isRealModel,
            modelName: this.modelName,
            totalParams: this.calculateParams()
        };
    }

    /**
     * Calculate total number of parameters
     * @returns {number} Parameter count
     */
    calculateParams() {
        // Each head: 3 * (embedDim * headDim) for Wq, Wk, Wv
        const perHead = 3 * this.embedDim * this.headDim;
        const headsTotal = this.numHeads * perHead;
        // Output projection: (numHeads * headDim) * embedDim
        const outputProj = this.numHeads * this.headDim * this.embedDim;
        return headsTotal + outputProj;
    }
}

/**
 * Factory function to create MHSA with random initialization
 * @param {number} embedDim - Embedding dimension
 * @param {number} numHeads - Number of heads
 * @param {number} temperature - Temperature parameter
 * @returns {MultiHeadSelfAttention} Initialized MHSA instance
 */
export function createRandomMHSA(embedDim, numHeads, temperature = 1.0) {
    const mhsa = new MultiHeadSelfAttention(embedDim, numHeads, temperature);
    mhsa.initializeRandomWeights();
    return mhsa;
}

export default MultiHeadSelfAttention;

