/**
 * MHSA Visualizer - Embeddings Module
 * Handles token embeddings from various sources including real transformer models
 */

import { Config } from './config.js';

// Global state for loaded models
let loadedPipeline = null;
let loadedTokenizer = null;
let loadedModel = null;
let currentModelKey = null;

/**
 * Get model config from Config.models
 */
function getModelConfig(key) {
    return Config.models[key] || null;
}

/**
 * Generate deterministic embeddings based on token hash
 * (Simple fallback when no real model is available)
 * @param {string[]} tokens - Array of tokens
 * @param {number} embedDim - Embedding dimension
 * @returns {number[][]} Embedding matrix
 */
export function generateDeterministicEmbeddings(tokens, embedDim) {
    const embeddings = [];
    for (let i = 0; i < tokens.length; i++) {
        const embedding = [];
        // Generate deterministic embeddings based on token hash
        const hash = tokens[i].split('').reduce((acc, char) =>
            acc + char.charCodeAt(0), 0);
        for (let j = 0; j < embedDim; j++) {
            const angle = (hash * 0.1 + j * 0.5) % (2 * Math.PI);
            embedding[j] = Math.sin(angle) * 0.5;
        }
        embeddings.push(embedding);
    }
    return embeddings;
}

/**
 * Generate random embeddings
 * @param {string[]} tokens - Array of tokens
 * @param {number} embedDim - Embedding dimension
 * @returns {number[][]} Embedding matrix
 */
export function generateRandomEmbeddings(tokens, embedDim) {
    const embeddings = [];
    for (let i = 0; i < tokens.length; i++) {
        const embedding = [];
        for (let j = 0; j < embedDim; j++) {
            embedding[j] = (Math.random() - 0.5) * 2;
        }
        embeddings.push(embedding);
    }
    return embeddings;
}

/**
 * Check if transformers.js is available
 * @returns {boolean} Whether transformers.js can be used
 */
export function isTransformersAvailable() {
    return typeof window !== 'undefined' && typeof WebAssembly !== 'undefined';
}

/**
 * Load the transformers.js library dynamically
 * @returns {Promise<Object>} Transformers module
 */
async function loadTransformersLibrary() {
    if (window.transformers) {
        return window.transformers;
    }
    
    // Try to import from CDN
    try {
        const module = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
        window.transformers = module;
        return module;
    } catch (error) {
        console.error('Failed to load transformers.js:', error);
        throw new Error('Could not load transformers.js library. Please check your internet connection.');
    }
}

/**
 * Load a transformer model for embeddings
 * @param {string} modelKey - Key of the model to load
 * @param {Function} progressCallback - Callback for loading progress
 * @returns {Promise<Object>} Loaded model info
 */
export async function loadModel(modelKey, progressCallback = null) {
    const modelConfig = getModelConfig(modelKey);
    if (!modelConfig) {
        throw new Error(`Unknown model: ${modelKey}`);
    }

    // Return cached model if already loaded
    if (currentModelKey === modelKey && loadedModel && loadedTokenizer) {
        return {
            model: loadedModel,
            tokenizer: loadedTokenizer,
            config: modelConfig,
            cached: true
        };
    }

    const { pipeline, AutoTokenizer, AutoModel, env } = await loadTransformersLibrary();
    
    // Configure environment
    env.allowLocalModels = false;
    env.useBrowserCache = true;

    const progressHandler = progressCallback || ((progress) => {
        if (progress.status === 'downloading') {
            console.log(`Downloading ${progress.file}: ${(progress.progress || 0).toFixed(1)}%`);
        }
    });

    try {
        // Load tokenizer
        loadedTokenizer = await AutoTokenizer.from_pretrained(modelConfig.hfId, {
            progress_callback: progressHandler
        });

        // Load model
        loadedModel = await AutoModel.from_pretrained(modelConfig.hfId, {
            progress_callback: progressHandler
        });

        currentModelKey = modelKey;

        return {
            model: loadedModel,
            tokenizer: loadedTokenizer,
            config: modelConfig,
            cached: false
        };
    } catch (error) {
        console.error('Failed to load model:', error);
        throw new Error(`Failed to load model ${modelConfig.name}: ${error.message}`);
    }
}

/**
 * Extract embeddings from a loaded transformer model
 * @param {string} text - Input text
 * @param {Object} model - Loaded model
 * @param {Object} tokenizer - Loaded tokenizer
 * @returns {Promise<Object>} Embeddings and token info
 */
export async function extractEmbeddings(text, model, tokenizer) {
    // Tokenize input
    const encoded = await tokenizer(text, {
        return_tensors: 'pt',
        padding: true,
        truncation: true
    });

    // Get model outputs - request hidden states for decoder-only models
    const outputs = await model(encoded, { output_hidden_states: true });
    
    // Debug: Log all available output keys
    console.log('Model output keys:', Object.keys(outputs));
    for (const key of Object.keys(outputs)) {
        const val = outputs[key];
        if (val && val.dims) {
            console.log(`  ${key}: dims=${JSON.stringify(val.dims)}`);
        } else if (Array.isArray(val)) {
            console.log(`  ${key}: array of length ${val.length}`);
        } else {
            console.log(`  ${key}: ${typeof val}`);
        }
    }
    
    // Extract the hidden state - different models have different output structures
    let hiddenState = null;
    
    if (outputs.last_hidden_state) {
        // Encoder models (BERT, DistilBERT, etc.)
        hiddenState = outputs.last_hidden_state;
        console.log('Using last_hidden_state');
    } else if (outputs.hidden_states && outputs.hidden_states.length > 0) {
        // Decoder-only models (GPT-2, etc.) - get last layer hidden states
        hiddenState = outputs.hidden_states[outputs.hidden_states.length - 1];
        console.log('Using hidden_states[last]');
    } else {
        // Try to find any tensor that looks like hidden states (3D tensor: batch, seq, hidden)
        for (const key of Object.keys(outputs)) {
            const tensor = outputs[key];
            if (tensor && tensor.dims && tensor.dims.length === 3) {
                // Check if this looks like hidden states (not logits which have vocab_size as last dim)
                const lastDim = tensor.dims[2];
                // Hidden states typically have dimension 768, 384, 1024, etc., not vocab size (50257+ for GPT-2)
                if (lastDim < 10000) {
                    hiddenState = tensor;
                    console.log(`Using fallback tensor '${key}' with dims ${JSON.stringify(tensor.dims)}`);
                    break;
                }
            }
        }
    }
    
    if (!hiddenState) {
        // Log available outputs for debugging
        console.error('Could not find hidden states. Available outputs:', Object.keys(outputs));
        
        // Get token info for the fallback
        const tokenIds = Array.from(encoded.input_ids.data);
        const tokenStrings = tokenIds.map(id => tokenizer.decode([id]));
        
        // Return info that will trigger fallback in caller
        throw new Error(`HIDDEN_STATES_NOT_AVAILABLE:${tokenStrings.join('|')}`);
    }
    
    // Convert to regular array
    const embeddings = [];
    const data = hiddenState.data;
    const [batchSize, seqLen, hiddenSize] = hiddenState.dims;
    
    for (let i = 0; i < seqLen; i++) {
        const embedding = [];
        for (let j = 0; j < hiddenSize; j++) {
            embedding.push(data[i * hiddenSize + j]);
        }
        embeddings.push(embedding);
    }

    // Get token strings
    const tokenIds = Array.from(encoded.input_ids.data);
    const tokens = tokenizer.decode(tokenIds, { skip_special_tokens: false })
        .split(' ')
        .filter(t => t.length > 0);
    
    // Get actual tokens from tokenizer
    const tokenStrings = tokenIds.map(id => tokenizer.decode([id]));

    return {
        embeddings,
        tokens: tokenStrings,
        tokenIds,
        hiddenSize
    };
}

/**
 * Extract attention weights from a transformer model
 * @param {string} text - Input text
 * @param {Object} model - Loaded model
 * @param {Object} tokenizer - Loaded tokenizer
 * @param {number} layer - Layer to extract from (default: last layer)
 * @returns {Promise<Object>} Attention weights and metadata
 */
export async function extractAttentionWeights(text, model, tokenizer, layer = -1) {
    // Tokenize input with attention output
    const encoded = await tokenizer(text, {
        return_tensors: 'pt',
        padding: true,
        truncation: true
    });

    // Get model outputs with attention
    const outputs = await model(encoded, { output_attentions: true });
    
    // Check for attention weights
    if (!outputs.attentions || outputs.attentions.length === 0) {
        // Log available outputs for debugging
        console.warn('Attention weights not found. Available outputs:', Object.keys(outputs));
        
        // For models that don't return attentions, create synthetic attention based on position
        const tokenIds = Array.from(encoded.input_ids.data);
        const seqLen = tokenIds.length;
        const numHeads = 12; // Default assumption
        
        // Create uniform attention as fallback
        const attentionWeights = [];
        for (let h = 0; h < numHeads; h++) {
            const headAttention = [];
            for (let i = 0; i < seqLen; i++) {
                const row = [];
                for (let j = 0; j < seqLen; j++) {
                    // Uniform attention
                    row.push(1.0 / seqLen);
                }
                headAttention.push(row);
            }
            attentionWeights.push(headAttention);
        }
        
        const tokenStrings = tokenIds.map(id => tokenizer.decode([id]));
        
        return {
            attentionWeights,
            tokens: tokenStrings,
            numHeads,
            layer: 0,
            numLayers: 1,
            isSynthetic: true
        };
    }

    // Get attention from specified layer
    const layerIdx = layer < 0 ? outputs.attentions.length + layer : layer;
    const attentionTensor = outputs.attentions[layerIdx];
    
    // Convert to regular arrays [batch, num_heads, seq_len, seq_len]
    const attentionData = attentionTensor.data;
    const [batch, numHeads, seqLen, seqLen2] = attentionTensor.dims;
    
    const attentionWeights = [];
    for (let h = 0; h < numHeads; h++) {
        const headAttention = [];
        for (let i = 0; i < seqLen; i++) {
            const row = [];
            for (let j = 0; j < seqLen2; j++) {
                const idx = h * seqLen * seqLen2 + i * seqLen2 + j;
                row.push(attentionData[idx]);
            }
            headAttention.push(row);
        }
        attentionWeights.push(headAttention);
    }

    // Get token strings
    const tokenIds = Array.from(encoded.input_ids.data);
    const tokenStrings = tokenIds.map(id => tokenizer.decode([id]));

    return {
        attentionWeights,
        tokens: tokenStrings,
        numHeads,
        layer: layerIdx,
        numLayers: outputs.attentions.length
    };
}

/**
 * Get embeddings using the best available method
 * @param {string[]} tokens - Array of tokens
 * @param {number} embedDim - Desired embedding dimension
 * @param {string} source - Embedding source ('random', 'deterministic', or model key)
 * @param {Function} progressCallback - Progress callback for model loading
 * @returns {Promise<Object>} Embeddings and source info
 */
export async function getEmbeddings(tokens, embedDim, source = 'deterministic', progressCallback = null) {
    switch (source) {
        case 'random':
            return {
                embeddings: generateRandomEmbeddings(tokens, embedDim),
                tokens,
                source: 'Random',
                isRealModel: false
            };
            
        case 'deterministic':
            return {
                embeddings: generateDeterministicEmbeddings(tokens, embedDim),
                tokens,
                source: 'Deterministic Hash',
                isRealModel: false
            };

        case 'distilbert':
            // Use transformers.js for DistilBERT embeddings
            return await getDistilBertEmbeddings(tokens.join(' '), progressCallback);
            
        default:
            // Try to load a real model
            if (isTransformersAvailable() && getModelConfig(source)) {
                try {
                    const { model, tokenizer, config } = await loadModel(source, progressCallback);
                    const text = tokens.join(' ');
                    const result = await extractEmbeddings(text, model, tokenizer);
                    
                    return {
                        embeddings: result.embeddings,
                        tokens: result.tokens,
                        source: config.name,
                        isRealModel: true,
                        modelConfig: config
                    };
                } catch (error) {
                    console.warn(`Failed to load model ${source}, falling back to deterministic:`, error);
                    return {
                        embeddings: generateDeterministicEmbeddings(tokens, embedDim),
                        tokens,
                        source: 'Deterministic Hash (fallback)',
                        isRealModel: false,
                        error: error.message
                    };
                }
            } else {
                return {
                    embeddings: generateDeterministicEmbeddings(tokens, embedDim),
                    tokens,
                    source: 'Deterministic Hash',
                    isRealModel: false
                };
            }
    }
}

/**
 * Get embeddings from DistilBERT using transformers.js
 * Uses the feature-extraction pipeline which properly exports hidden states
 * @param {string} text - Input text
 * @param {Function} progressCallback - Progress callback for model loading
 * @returns {Promise<Object>} Embeddings and token info
 */
export async function getDistilBertEmbeddings(text, progressCallback = null) {
    if (!isTransformersAvailable()) {
        throw new Error('Transformers.js is not available');
    }

    const { AutoTokenizer, pipeline, env } = await loadTransformersLibrary();
    
    env.allowLocalModels = false;
    env.useBrowserCache = true;

    // Use a model that's known to export hidden states properly
    // Xenova/all-MiniLM-L6-v2 is a sentence-transformer model designed for embeddings
    const modelId = 'Xenova/all-MiniLM-L6-v2';
    
    const progressHandler = progressCallback || ((progress) => {
        if (progress.status === 'downloading') {
            console.log(`Downloading ${progress.file}: ${(progress.progress || 0).toFixed(1)}%`);
        }
    });

    try {
        // Load tokenizer if not already loaded
        if (!loadedTokenizer || currentModelKey !== 'distilbert-embed') {
            progressHandler({ status: 'loading', message: 'Loading tokenizer...' });
            loadedTokenizer = await AutoTokenizer.from_pretrained(modelId, {
                progress_callback: progressHandler
            });
        }

        // Load feature extraction pipeline if not already loaded
        if (!loadedPipeline || currentModelKey !== 'distilbert-embed') {
            progressHandler({ status: 'loading', message: 'Loading embedding model...' });
            loadedPipeline = await pipeline('feature-extraction', modelId, {
                progress_callback: progressHandler
            });
            currentModelKey = 'distilbert-embed';
        }

        // Tokenize to get token strings
        const encoded = await loadedTokenizer(text, {
            return_tensors: 'pt',
            padding: true,
            truncation: true
        });
        const tokenIds = Array.from(encoded.input_ids.data);
        const tokens = tokenIds.map(id => loadedTokenizer.decode([id]));

        // Get embeddings using the feature extraction pipeline
        // pooling: 'none' returns per-token embeddings instead of sentence embedding
        const output = await loadedPipeline(text, { pooling: 'none', normalize: false });
        
        // output is a Tensor with shape [1, seq_len, hidden_size]
        const [batchSize, seqLen, hiddenSize] = output.dims;
        const data = output.data;
        
        const embeddings = [];
        for (let i = 0; i < seqLen; i++) {
            const embedding = [];
            for (let j = 0; j < hiddenSize; j++) {
                embedding.push(data[i * hiddenSize + j]);
            }
            embeddings.push(embedding);
        }

        progressHandler({ status: 'complete', message: 'Ready' });

        return {
            embeddings,
            tokens,
            tokenIds,
            hiddenSize,
            source: 'MiniLM (transformers.js)',
            isRealModel: true,
            modelConfig: {
                name: 'MiniLM-L6-v2',
                embedDim: hiddenSize,  // 384 for MiniLM
                numHeads: 12,
                headDim: 32  // 384 / 12
            }
        };

    } catch (error) {
        console.error('Failed to get embeddings:', error);
        throw error;
    }
}

/**
 * Get real attention patterns from a transformer model
 * @param {string} text - Input text
 * @param {string} modelKey - Model to use
 * @param {Function} progressCallback - Progress callback
 * @returns {Promise<Object>} Real attention patterns
 */
export async function getRealAttention(text, modelKey, progressCallback = null) {
    if (!isTransformersAvailable()) {
        throw new Error('Transformers.js is not available in this environment');
    }

    const modelConfig = getModelConfig(modelKey);
    if (!modelConfig) {
        throw new Error(`Unknown model: ${modelKey}`);
    }

    const { model, tokenizer, config } = await loadModel(modelKey, progressCallback);
    
    let embedResult, attentionResult;
    
    try {
        [embedResult, attentionResult] = await Promise.all([
            extractEmbeddings(text, model, tokenizer),
            extractAttentionWeights(text, model, tokenizer)
        ]);
    } catch (error) {
        // Check if this is a "hidden states not available" error
        if (error.message.startsWith('HIDDEN_STATES_NOT_AVAILABLE:')) {
            const tokenStrings = error.message.split(':')[1].split('|');
            
            // Generate deterministic embeddings based on model dimensions
            const embeddings = generateDeterministicEmbeddings(tokenStrings, config.embedDim);
            
            // Try to get attention weights separately
            try {
                attentionResult = await extractAttentionWeights(text, model, tokenizer);
            } catch (attErr) {
                console.warn('Attention weights also not available:', attErr);
                // Create synthetic attention
                const seqLen = tokenStrings.length;
                const numHeads = config.numHeads || 12;
                const syntheticAttention = [];
                for (let h = 0; h < numHeads; h++) {
                    const headAttn = [];
                    for (let i = 0; i < seqLen; i++) {
                        const row = new Array(seqLen).fill(1.0 / seqLen);
                        headAttn.push(row);
                    }
                    syntheticAttention.push(headAttn);
                }
                attentionResult = {
                    attentionWeights: syntheticAttention,
                    tokens: tokenStrings,
                    numHeads: numHeads,
                    isSynthetic: true
                };
            }
            
            console.warn(`Model ${modelKey} does not support hidden state extraction. Using deterministic embeddings with model tokenization.`);
            
            return {
                embeddings,
                attentionWeights: attentionResult.attentionWeights,
                tokens: tokenStrings,
                numHeads: attentionResult.numHeads,
                modelConfig: config,
                isRealModel: false,
                usedFallback: true,
                fallbackReason: 'Model does not export hidden states in ONNX format'
            };
        }
        
        // Re-throw other errors
        throw error;
    }

    return {
        embeddings: embedResult.embeddings,
        attentionWeights: attentionResult.attentionWeights,
        tokens: embedResult.tokens,
        numHeads: attentionResult.numHeads,
        modelConfig: config,
        isRealModel: true
    };
}

/**
 * Get the currently loaded model info
 * @returns {Object|null} Current model info or null if none loaded
 */
export function getCurrentModelInfo() {
    if (!currentModelKey || !loadedModel) {
        return null;
    }
    
    return {
        modelKey: currentModelKey,
        config: getModelConfig(currentModelKey),
        isLoaded: true
    };
}

/**
 * Unload the current model to free memory
 */
export function unloadModel() {
    loadedModel = null;
    loadedTokenizer = null;
    loadedPipeline = null;
    currentModelKey = null;
}

export default {
    generateDeterministicEmbeddings,
    generateRandomEmbeddings,
    getEmbeddings,
    loadModel,
    extractEmbeddings,
    extractAttentionWeights,
    getRealAttention,
    getCurrentModelInfo,
    unloadModel,
    isTransformersAvailable
};

