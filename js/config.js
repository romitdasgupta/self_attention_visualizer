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
    // Note: Xenova ONNX models don't export hidden states/attention weights,
    // so we only offer methods that fully support all visualization features
    embeddingSources: {
        RANDOM: 'random',
        DETERMINISTIC: 'deterministic'
    },

    // Model configurations removed - Xenova ONNX exports don't support
    // hidden state extraction required for proper embeddings visualization
    models: {},

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

