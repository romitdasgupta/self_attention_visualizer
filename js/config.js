/**
 * MHSA Visualizer - Configuration Module
 * Central configuration and constants
 */

export const Config = {
    // Default model settings
    defaults: {
        embedDim: 64,
        numHeads: 4,
        temperature: 1.0,
        inputText: "The cat sat on the mat"
    },

    // Available embedding sources
    embeddingSources: {
        RANDOM: 'random',
        DETERMINISTIC: 'deterministic',
        DISTILBERT: 'distilbert'
    },

    // Pre-extracted model configurations
    // These use real weights extracted from trained models
    models: {
        'distilbert-4head': {
            name: 'DistilBERT (4 heads)',
            description: 'First 4 attention heads from DistilBERT layer 0',
            weightsFile: 'models/distilbert-4head.json',
            hfId: 'distilbert-base-uncased',
            embedDim: 768,
            numHeads: 4,
            headDim: 64,
            isRealModel: true
        },
        'distilbert-2head': {
            name: 'DistilBERT (2 heads)',
            description: 'First 2 attention heads from DistilBERT layer 0',
            weightsFile: 'models/distilbert-2head.json',
            hfId: 'distilbert-base-uncased',
            embedDim: 768,
            numHeads: 2,
            headDim: 64,
            isRealModel: true
        },
        'distilbert-1head': {
            name: 'DistilBERT (1 head)',
            description: 'First attention head from DistilBERT layer 0',
            weightsFile: 'models/distilbert-1head.json',
            hfId: 'distilbert-base-uncased',
            embedDim: 768,
            numHeads: 1,
            headDim: 64,
            isRealModel: true
        },
        'demo-tiny': {
            name: 'Demo (Tiny)',
            description: 'Small demo model with Xavier-initialized weights',
            weightsFile: 'models/demo-tiny-weights.json',
            embedDim: 64,
            numHeads: 4,
            headDim: 16,
            isRealModel: false
        }
    },

    // Visualization colors
    colors: {
        primary: '#667eea',
        secondary: '#764ba2',
        query: '#FF6B6B',
        key: '#4ECDC4',
        value: '#95E1D3',
        attention: '#ffc107',
        output: '#84fab0',
        embedding: '#a8edea',
        
        // Head colors for multi-head visualization
        heads: ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F7DC6F', '#DDA0DD', '#87CEEB', '#FFA07A', '#98D8C8']
    },

    // Animation settings
    animation: {
        duration: 300,
        easing: 'ease-in-out'
    },

    // Step names for sequence visualization
    stepNames: [
        'Input Tokens',
        'Token Embeddings',
        'Linear Projections',
        'Query-Key Dot Product',
        'Scaling',
        'Softmax',
        'Attention × Values',
        'Concatenate Heads',
        'Final Output'
    ],

    // Canvas settings
    canvas: {
        heatmapCellSize: 50,
        heatmapMargin: 80,
        maxCellSize: 50,
        minCellSize: 20
    }
};

/**
 * Get a model configuration by key
 * @param {string} modelKey - Model identifier
 * @returns {Object} Model configuration
 */
export function getModelConfig(modelKey) {
    return Config.models[modelKey] || null;
}

/**
 * Get all available model keys
 * @returns {string[]} Array of model keys
 */
export function getAvailableModels() {
    return Object.keys(Config.models);
}

/**
 * Check if transformers.js is supported
 * @returns {boolean} Whether WebGPU/WASM is supported
 */
export function isTransformersSupported() {
    // Check for basic requirements
    return typeof WebAssembly !== 'undefined';
}

export default Config;

