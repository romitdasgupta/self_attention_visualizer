/**
 * MHSA Visualizer - Model Weights Module
 * Handles loading and caching of pre-extracted transformer model weights
 */

import { Config } from './config.js';

// Cache for loaded model weights
const weightsCache = new Map();

// Loading state
let currentLoadingModel = null;

/**
 * Load model weights from a JSON file
 * @param {string} modelKey - Key from Config.models
 * @param {Function} progressCallback - Optional callback for loading progress
 * @returns {Promise<Object>} Loaded weights
 */
export async function loadModelWeights(modelKey, progressCallback = null) {
    // Check cache first
    if (weightsCache.has(modelKey)) {
        console.log(`Using cached weights for ${modelKey}`);
        return weightsCache.get(modelKey);
    }

    const modelConfig = Config.models[modelKey];
    if (!modelConfig) {
        throw new Error(`Unknown model: ${modelKey}`);
    }

    if (!modelConfig.weightsFile) {
        throw new Error(`Model ${modelKey} has no weights file configured`);
    }

    // Prevent duplicate loading
    if (currentLoadingModel === modelKey) {
        // Wait for existing load
        await new Promise(resolve => setTimeout(resolve, 100));
        if (weightsCache.has(modelKey)) {
            return weightsCache.get(modelKey);
        }
    }

    currentLoadingModel = modelKey;

    try {
        if (progressCallback) {
            progressCallback({ status: 'downloading', progress: 0, file: modelConfig.weightsFile });
        }

        console.log(`Loading weights from ${modelConfig.weightsFile}...`);
        
        const response = await fetch(modelConfig.weightsFile);
        
        if (!response.ok) {
            throw new Error(`Failed to load ${modelConfig.weightsFile}: ${response.status}`);
        }

        // Track download progress if possible
        const reader = response.body?.getReader();
        const contentLength = response.headers.get('Content-Length');
        
        let weights;
        
        if (reader && contentLength) {
            // Stream with progress
            const total = parseInt(contentLength, 10);
            let loaded = 0;
            const chunks = [];
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                chunks.push(value);
                loaded += value.length;
                
                if (progressCallback) {
                    progressCallback({ 
                        status: 'downloading', 
                        progress: (loaded / total) * 100,
                        file: modelConfig.weightsFile 
                    });
                }
            }
            
            // Combine chunks and parse
            const allChunks = new Uint8Array(loaded);
            let position = 0;
            for (const chunk of chunks) {
                allChunks.set(chunk, position);
                position += chunk.length;
            }
            
            const jsonText = new TextDecoder().decode(allChunks);
            weights = JSON.parse(jsonText);
        } else {
            // Simple fetch without progress
            weights = await response.json();
        }

        if (progressCallback) {
            progressCallback({ status: 'complete', progress: 100 });
        }

        // Validate structure
        if (!weights.heads || !weights.config) {
            throw new Error('Invalid weights file structure');
        }

        // Add model config reference
        weights.modelKey = modelKey;
        weights.modelConfig = modelConfig;

        // Cache the weights
        weightsCache.set(modelKey, weights);
        
        console.log(`Loaded ${modelKey}: ${weights.heads.length} heads, embed_dim=${weights.config.embed_dim}`);

        return weights;

    } catch (error) {
        console.error(`Failed to load weights for ${modelKey}:`, error);
        throw error;
    } finally {
        currentLoadingModel = null;
    }
}

/**
 * Get weights for a specific head from loaded model
 * @param {Object} weights - Loaded weights object
 * @param {number} headIndex - Head index (0-based)
 * @returns {Object} Head weights {Wq, Wk, Wv, Wo, bq, bk, bv}
 */
export function getHeadWeights(weights, headIndex) {
    if (!weights.heads || headIndex >= weights.heads.length) {
        throw new Error(`Invalid head index ${headIndex} for model with ${weights.heads?.length || 0} heads`);
    }
    return weights.heads[headIndex];
}

/**
 * Convert loaded weights to MHSA-compatible format
 * @param {Object} weights - Loaded weights object
 * @returns {Object} Weights in MHSA format
 */
export function toMHSAWeights(weights) {
    const config = weights.config;
    
    return {
        embedDim: config.embed_dim,
        numHeads: config.num_heads,
        headDim: config.head_dim,
        heads: weights.heads.map(head => ({
            Wq: head.Wq,
            Wk: head.Wk,
            Wv: head.Wv,
            bq: head.bq || null,
            bk: head.bk || null,
            bv: head.bv || null,
            Wo: head.Wo
        })),
        bo: weights.bo || null,
        isRealModel: weights.modelConfig?.isRealModel || false,
        modelName: weights.model_name || weights.modelConfig?.name || 'Unknown'
    };
}

/**
 * Clear the weights cache
 */
export function clearCache() {
    weightsCache.clear();
    console.log('Weights cache cleared');
}

/**
 * Check if a model's weights are cached
 * @param {string} modelKey - Model key
 * @returns {boolean} Whether weights are cached
 */
export function isCached(modelKey) {
    return weightsCache.has(modelKey);
}

/**
 * Get list of available models
 * @returns {Array} Array of {key, name, description, embedDim, numHeads}
 */
export function getAvailableModels() {
    return Object.entries(Config.models).map(([key, config]) => ({
        key,
        name: config.name,
        description: config.description,
        embedDim: config.embedDim,
        numHeads: config.numHeads,
        headDim: config.headDim,
        isRealModel: config.isRealModel
    }));
}

export default {
    loadModelWeights,
    getHeadWeights,
    toMHSAWeights,
    clearCache,
    isCached,
    getAvailableModels
};

