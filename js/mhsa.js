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
        this.bo = null;  // Output bias
        this.isRealModel = false;
        this.modelName = null;
        this.useBias = false;  // Whether to use biases in projections
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
                Wv: MathUtils.randomMatrix(this.embedDim, this.headDim),
                Wo: MathUtils.randomMatrix(this.headDim, this.embedDim),  // Per-head output
                bq: null,
                bk: null,
                bv: null
            });
        }
        // Combined output projection (for backwards compatibility)
        this.Wo = MathUtils.randomMatrix(this.numHeads * this.headDim, this.embedDim);
        this.bo = null;
        this.isRealModel = false;
        this.modelName = 'Random Initialization';
        this.useBias = false;
    }

    /**
     * Load weights from a real model (old format with combined Wo)
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
            Wv: head.Wv,
            Wo: head.Wo || null,
            bq: head.bq || null,
            bk: head.bk || null,
            bv: head.bv || null
        }));
        
        this.Wo = weights.Wo;
        this.bo = weights.bo || null;
        this.isRealModel = true;
        this.modelName = modelName;
        this.useBias = !!(weights.heads[0].bq);
    }

    /**
     * Load pre-extracted weights from JSON file format
     * @param {Object} weights - Pre-extracted weights with per-head structure
     * @param {string} modelName - Name of the source model
     */
    loadPreExtractedWeights(weights, modelName) {
        if (!weights || !weights.heads) {
            throw new Error('Invalid pre-extracted weight structure');
        }

        const config = weights.config || {};
        this.numHeads = config.num_heads || weights.heads.length;
        this.headDim = config.head_dim || weights.heads[0].Wq[0].length;
        this.embedDim = config.embed_dim || weights.heads[0].Wq.length;
        
        // Load per-head weights
        this.heads = weights.heads.map(head => ({
            Wq: head.Wq,
            Wk: head.Wk,
            Wv: head.Wv,
            Wo: head.Wo,  // Per-head output projection
            bq: head.bq || null,
            bk: head.bk || null,
            bv: head.bv || null
        }));
        
        // For combined projection (backwards compatibility), we'll compute it during forward
        this.Wo = null;  // Will use per-head Wo instead
        this.bo = weights.bo || null;
        this.isRealModel = weights.modelConfig?.isRealModel !== false;
        this.modelName = modelName || weights.model_name || 'Pre-extracted Model';
        this.useBias = !!(weights.heads[0].bq);
        
        console.log(`Loaded ${this.modelName}: ${this.numHeads} heads, embed_dim=${this.embedDim}, head_dim=${this.headDim}, useBias=${this.useBias}`);
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
     * Apply linear projection with optional bias
     * @param {number[][]} X - Input matrix [seqLen x inputDim]
     * @param {number[][]} W - Weight matrix [inputDim x outputDim]
     * @param {number[]|null} b - Optional bias vector [outputDim]
     * @returns {number[][]} Output matrix [seqLen x outputDim]
     */
    linearProjection(X, W, b = null) {
        const result = MathUtils.matmul(X, W);
        
        if (b && b.length > 0) {
            // Add bias to each row
            for (let i = 0; i < result.length; i++) {
                for (let j = 0; j < result[i].length; j++) {
                    result[i][j] += b[j];
                }
            }
        }
        
        return result;
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
        const headProjectedOutputs = [];  // Per-head outputs after Wo projection

        // Process each head
        for (let h = 0; h < this.numHeads; h++) {
            const head = this.heads[h];

            // Project to Q, K, V (with optional biases)
            const Q = this.linearProjection(embeddings, head.Wq, head.bq);
            const K = this.linearProjection(embeddings, head.Wk, head.bk);
            const V = this.linearProjection(embeddings, head.Wv, head.bv);

            // Compute attention
            const { output, attentionWeights, scores } =
                this.scaledDotProductAttention(Q, K, V);

            headOutputs.push({
                Q, K, V, output, attentionWeights, scores,
                Wq: head.Wq,
                Wk: head.Wk,
                Wv: head.Wv,
                Wo: head.Wo
            });
            headAttentions.push(attentionWeights);

            // Apply per-head output projection if available
            if (head.Wo) {
                const projected = MathUtils.matmul(output, head.Wo);
                headProjectedOutputs.push(projected);
            }
        }

        let finalOutput;
        let concatenated;

        if (headProjectedOutputs.length === this.numHeads) {
            // Sum the per-head projected outputs (like in real transformers)
            finalOutput = headProjectedOutputs[0].map(row => [...row]);
            for (let h = 1; h < this.numHeads; h++) {
                for (let i = 0; i < seqLen; i++) {
                    for (let j = 0; j < this.embedDim; j++) {
                        finalOutput[i][j] += headProjectedOutputs[h][i][j];
                    }
                }
            }
            
            // Add output bias if present
            if (this.bo && this.bo.length > 0) {
                for (let i = 0; i < seqLen; i++) {
                    for (let j = 0; j < this.embedDim; j++) {
                        finalOutput[i][j] += this.bo[j];
                    }
                }
            }

            // Still compute concatenated for visualization purposes
            concatenated = [];
            for (let i = 0; i < seqLen; i++) {
                concatenated[i] = [];
                for (let h = 0; h < this.numHeads; h++) {
                    concatenated[i].push(...headOutputs[h].output[i]);
                }
            }
        } else {
            // Fall back to concatenation + combined Wo (original behavior)
            concatenated = [];
            for (let i = 0; i < seqLen; i++) {
                concatenated[i] = [];
                for (let h = 0; h < this.numHeads; h++) {
                    concatenated[i].push(...headOutputs[h].output[i]);
                }
            }

            // Final linear projection
            finalOutput = MathUtils.matmul(concatenated, this.Wo);
        }

        return {
            output: finalOutput,
            headOutputs,
            headAttentions,
            concatenated,
            headProjectedOutputs: headProjectedOutputs.length > 0 ? headProjectedOutputs : null
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

/**
 * Factory function to create MHSA with pre-extracted weights
 * @param {Object} weights - Pre-extracted weights object
 * @param {number} temperature - Temperature parameter
 * @returns {MultiHeadSelfAttention} Initialized MHSA instance
 */
export function createMHSAFromWeights(weights, temperature = 1.0) {
    const config = weights.config || {};
    const embedDim = config.embed_dim || weights.heads[0].Wq.length;
    const numHeads = config.num_heads || weights.heads.length;
    
    const mhsa = new MultiHeadSelfAttention(embedDim, numHeads, temperature);
    mhsa.loadPreExtractedWeights(weights, weights.model_name || 'Pre-extracted Model');
    return mhsa;
}

export default MultiHeadSelfAttention;

