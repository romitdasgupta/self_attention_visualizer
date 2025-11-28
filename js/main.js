/**
 * MHSA Visualizer - Main Entry Point
 * Initializes the application and coordinates all modules
 */

import { Config } from './config.js';
import { MathUtils } from './math.js';
import { MultiHeadSelfAttention, createRandomMHSA, createMHSAFromWeights } from './mhsa.js';
import { getEmbeddings, getDistilBertEmbeddings, isTransformersAvailable } from './embeddings.js';
import { loadModelWeights, getAvailableModels } from './model-weights.js';
import {
    drawAttentionHeatmap,
    drawSoftmaxVisualization,
    drawWeightMatrixVisualization,
    drawAggregationVisualization,
    drawComparisonVisualization,
    formatMatrix
} from './visualization.js';
import {
    initializeStepNavigator,
    renderStep,
    nextStep,
    previousStep,
    goToStep,
    resetSequence,
    switchTab,
    setArchView,
    renderArchitectureView,
    renderHeadsParallelView,
    renderArchStats,
    updateFlowDimensions,
    displayTokens,
    displayArchitectureInfo
} from './ui.js';

// Global state
let currentAttention = null;
let sequenceData = null;
let currentEmbeddingSource = 'deterministic';
let currentModelKey = null;  // For real models
let loadedWeights = null;    // Cached model weights
let isLoading = false;

/**
 * Initialize the application
 */
export function init() {
    // Expose functions globally for onclick handlers
    window.runAttention = runAttention;
    window.nextStep = () => nextStep(sequenceData);
    window.previousStep = () => previousStep(sequenceData);
    window.goToStep = (step) => goToStep(step, sequenceData);
    window.resetSequence = () => resetSequence(sequenceData);
    window.switchTab = switchTab;
    window.setArchView = setArchView;
    window.setEmbeddingSource = setEmbeddingSource;
    window.setModelSource = setModelSource;
    
    // Setup the embedding source selector
    setupModelSelector();
    
    // Run initial attention computation
    runAttention();
    renderArchitectureView();
}

/**
 * Setup the embedding source selector UI
 */
function setupModelSelector() {
    const controlsDiv = document.querySelector('.controls');
    if (!controlsDiv) return;
    
    // Check if transformers.js is available for DistilBERT
    const transformersAvailable = isTransformersAvailable();
    
    const modelSelectorHTML = `
        <div class="model-selector">
            <label>Model Source:</label>
            <div class="model-options">
                <div class="model-option" data-source="deterministic" onclick="setEmbeddingSource('deterministic')">
                    <strong>Deterministic</strong>
                    <small>Hash-based (instant)</small>
                </div>
                <div class="model-option" data-source="random" onclick="setEmbeddingSource('random')">
                    <strong>Random</strong>
                    <small>Xavier init (instant)</small>
                </div>
                <div class="model-option real-model" data-source="distilbert-4head" onclick="setModelSource('distilbert-4head')" ${!transformersAvailable ? 'style="opacity: 0.5"' : ''}>
                    <strong>🤖 DistilBERT</strong>
                    <small>Real model (4 heads)</small>
                    ${!transformersAvailable ? '<small style="color: #dc3545;">Requires WebAssembly</small>' : ''}
                </div>
            </div>
            <div id="modelLoadingStatus" class="model-loading-status" style="display: none;">
                <div class="loading-spinner"></div>
                <span id="loadingStatusText">Loading model...</span>
            </div>
        </div>
    `;
    
    // Insert after the control row
    const controlRow = controlsDiv.querySelector('.control-row');
    if (controlRow) {
        controlRow.insertAdjacentHTML('afterend', modelSelectorHTML);
    }
    
    // Set initial active state
    updateModelSelectorUI('deterministic');
}

/**
 * Update the model selector UI to show active state
 * @param {string} source - Active source identifier
 */
function updateModelSelectorUI(source) {
    document.querySelectorAll('.model-option').forEach(opt => {
        const optSource = opt.dataset.source;
        opt.classList.toggle('active', optSource === source);
    });
}

/**
 * Show/hide loading status
 * @param {boolean} show - Whether to show loading
 * @param {string} message - Loading message
 */
function showLoadingStatus(show, message = 'Loading model...') {
    const statusDiv = document.getElementById('modelLoadingStatus');
    const statusText = document.getElementById('loadingStatusText');
    if (statusDiv) {
        statusDiv.style.display = show ? 'flex' : 'none';
    }
    if (statusText) {
        statusText.textContent = message;
    }
}

/**
 * Lock/unlock parameter inputs when using a real model
 * @param {boolean} lock - Whether to lock the inputs
 * @param {Object} config - Model config with embedDim and numHeads
 */
function lockParameterInputs(lock, config = null) {
    const embedDimInput = document.getElementById('embedDim');
    const numHeadsInput = document.getElementById('numHeads');
    
    if (embedDimInput) {
        embedDimInput.disabled = lock;
        if (lock && config) {
            embedDimInput.value = config.embedDim;
        }
    }
    
    if (numHeadsInput) {
        numHeadsInput.disabled = lock;
        if (lock && config) {
            numHeadsInput.value = config.numHeads;
        }
    }
}

/**
 * Set the embedding source (for random/deterministic modes)
 * @param {string} source - Embedding source identifier
 */
export function setEmbeddingSource(source) {
    currentEmbeddingSource = source;
    currentModelKey = null;
    loadedWeights = null;
    
    // Unlock parameter inputs
    lockParameterInputs(false);
    
    // Update UI
    updateModelSelectorUI(source);
    
    // Re-run attention with new source
    runAttention();
}

/**
 * Set the model source (for real models with pre-extracted weights)
 * @param {string} modelKey - Model key from Config.models
 */
export async function setModelSource(modelKey) {
    if (isLoading) return;
    
    const modelConfig = Config.models[modelKey];
    if (!modelConfig) {
        console.error(`Unknown model: ${modelKey}`);
        return;
    }
    
    currentModelKey = modelKey;
    currentEmbeddingSource = 'distilbert';  // Use real embeddings
    
    // Update UI
    updateModelSelectorUI(modelKey);
    
    // Lock parameters to model's dimensions
    lockParameterInputs(true, modelConfig);
    
    // Load weights if not already loaded
    if (!loadedWeights || loadedWeights.modelKey !== modelKey) {
        isLoading = true;
        showLoadingStatus(true, 'Loading model weights...');
        
        try {
            loadedWeights = await loadModelWeights(modelKey, (progress) => {
                if (progress.status === 'downloading') {
                    showLoadingStatus(true, `Downloading: ${progress.progress?.toFixed(0) || 0}%`);
                }
            });
            
            showLoadingStatus(true, 'Loading embeddings model...');
            
            // Now run attention with the loaded weights
            await runAttention();
            
            showLoadingStatus(false);
        } catch (error) {
            console.error('Failed to load model:', error);
            showLoadingStatus(true, `Error: ${error.message}`);
            setTimeout(() => showLoadingStatus(false), 3000);
            
            // Fall back to deterministic
            setEmbeddingSource('deterministic');
        } finally {
            isLoading = false;
        }
    } else {
        // Already loaded, just run attention
        await runAttention();
    }
}

/**
 * Main function to run attention computation
 */
export async function runAttention() {
    const inputText = document.getElementById('inputText')?.value || Config.defaults.inputText;
    let embedDim = parseInt(document.getElementById('embedDim')?.value) || Config.defaults.embedDim;
    let numHeads = parseInt(document.getElementById('numHeads')?.value) || Config.defaults.numHeads;
    const temperature = parseFloat(document.getElementById('temperature')?.value) || Config.defaults.temperature;

    let tokens = inputText.trim().split(/\s+/);
    let embeddings;
    let mhsa;
    let isRealModel = false;

    // Check if we're using a real model
    if (currentModelKey && loadedWeights) {
        // Using real model with pre-extracted weights
        const modelConfig = Config.models[currentModelKey];
        embedDim = modelConfig.embedDim;
        numHeads = modelConfig.numHeads;
        isRealModel = true;

        try {
            // Get tokenization and embeddings from embedding model
            showLoadingStatus(true, 'Getting tokenization...');
            
            const embeddingResult = await getDistilBertEmbeddings(inputText, (progress) => {
                if (progress.message) {
                    showLoadingStatus(true, progress.message);
                }
            });
            
            tokens = embeddingResult.tokens;  // Use tokenized tokens (may include [CLS], [SEP], subwords)
            
            // Check if embedding dimensions match the model weights
            if (embeddingResult.hiddenSize === embedDim) {
                // Dimensions match, use the real embeddings
                embeddings = embeddingResult.embeddings;
            } else {
                // Dimension mismatch - use deterministic embeddings at correct dimension
                // but still benefit from proper tokenization
                embeddings = generateDeterministicEmbeddingsForDim(tokens, embedDim);
            }
            
            showLoadingStatus(false);
            
            // Create MHSA with real weights
            mhsa = createMHSAFromWeights(loadedWeights, temperature);
            
        } catch (error) {
            console.error('Failed to get tokenization:', error);
            showLoadingStatus(true, `Error: ${error.message}. Falling back to deterministic.`);
            setTimeout(() => showLoadingStatus(false), 3000);
            
            // Fall back to deterministic embeddings with the real weights
            embeddings = generateDeterministicEmbeddingsForDim(tokens, embedDim);
            mhsa = createMHSAFromWeights(loadedWeights, temperature);
        }
    } else {
        // Using random/deterministic mode
        const embeddingResult = await getEmbeddings(tokens, embedDim, currentEmbeddingSource);
        embeddings = embeddingResult.embeddings;
        tokens = embeddingResult.tokens || tokens;
        
        // Create MHSA with random weights
        mhsa = createRandomMHSA(embedDim, numHeads, temperature);
    }
    
    // Display tokens (may have been updated by tokenizer)
    displayTokens(tokens);

    // Run the forward pass
    const result = mhsa.forward(embeddings);
    currentAttention = result;

    // Display architecture info with real model indicator
    displayArchitectureInfo(mhsa, tokens.length);
    
    // Add real model badge if applicable
    if (isRealModel) {
        const infoContainer = document.getElementById('architectureInfo');
        if (infoContainer) {
            const badge = document.createElement('div');
            badge.className = 'real-model-badge';
            badge.innerHTML = '🤖 <strong>Real Model:</strong> Using pre-trained DistilBERT attention weights with transformer tokenization';
            infoContainer.insertBefore(badge, infoContainer.firstChild);
        }
    }

    // Draw combined attention heatmap
    const avgAttention = MathUtils.averageAttentions(result.headAttentions);
    const combinedCanvas = document.getElementById('combinedAttentionCanvas');
    if (combinedCanvas) {
        drawAttentionHeatmap(combinedCanvas, avgAttention, tokens);
    }

    // Draw individual head canvases
    const headsContainer = document.getElementById('headsContainer');
    if (headsContainer) {
        headsContainer.innerHTML = '';
        for (let h = 0; h < numHeads; h++) {
            const headDiv = document.createElement('div');
            headDiv.className = 'head-container';
            headDiv.innerHTML = `
                <div class="head-title">Head ${h + 1}${isRealModel ? ' <small>(real weights)</small>' : ''}</div>
                <canvas id="headCanvas${h}"></canvas>
            `;
            headsContainer.appendChild(headDiv);

            setTimeout(() => {
                const canvas = document.getElementById(`headCanvas${h}`);
                if (canvas) {
                    drawAttentionHeatmap(canvas, result.headAttentions[h], tokens);
                }
            }, 10);
        }
    }

    // Generate mathematical breakdown
    generateMathBreakdown(tokens, embeddings, mhsa, result);

    // Generate Q/K/V visualizations
    generateQKVVisualization(tokens, embeddings, mhsa, result);

    // Generate attention flow visualization
    generateAttentionFlowVisualization(tokens, embeddings, mhsa, result);

    // Prepare sequence data
    sequenceData = {
        tokens,
        embeddings,
        mhsa,
        result,
        focusToken: Math.floor(tokens.length / 2),
        isRealModel
    };

    // Initialize step navigator
    initializeStepNavigator();

    // Render initial step
    renderStep(sequenceData);

    // Update architecture view
    renderHeadsParallelView(mhsa.numHeads, mhsa.headDim);
    renderArchStats(mhsa, tokens.length);
    updateFlowDimensions(tokens.length, mhsa.embedDim, mhsa.numHeads, mhsa.headDim);
}

/**
 * Generate deterministic embeddings for a specific dimension
 * (Helper for fallback when real embeddings fail)
 */
function generateDeterministicEmbeddingsForDim(tokens, embedDim) {
    const embeddings = [];
    for (let i = 0; i < tokens.length; i++) {
        const embedding = [];
        const hash = tokens[i].split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
        for (let j = 0; j < embedDim; j++) {
            const angle = (hash * 0.1 + j * 0.5) % (2 * Math.PI);
            embedding[j] = Math.sin(angle) * 0.5;
        }
        embeddings.push(embedding);
    }
    return embeddings;
}

/**
 * Generate mathematical breakdown HTML
 */
function generateMathBreakdown(tokens, embeddings, mhsa, result) {
    const container = document.getElementById('mathBreakdown');
    if (!container) return;

    const head0 = result.headOutputs[0];

    container.innerHTML = `
        <div class="math-step">
            <h4>Step 1: Input Embeddings</h4>
            <p>Each token is represented as a ${mhsa.embedDim}-dimensional vector.</p>
            <div class="formula">X ∈ ℝ^(${tokens.length} × ${mhsa.embedDim})</div>
            ${formatMatrix(embeddings.map(e => e.slice(0, 8)), 'Embeddings (first 8 dimensions)')}
        </div>

        <div class="math-step">
            <h4>Step 2: Linear Projections</h4>
            <p>For each head h, we project the input into Query, Key, and Value spaces:</p>
            <div class="formula">
                Q_h = X · W^Q_h<br>
                K_h = X · W^K_h<br>
                V_h = X · W^V_h<br>
                where W^Q_h, W^K_h, W^V_h ∈ ℝ^(${mhsa.embedDim} × ${mhsa.headDim})
            </div>
            <div class="info-box">
                Each head has its own projection matrices, allowing it to learn different representations.
            </div>
        </div>

        <div class="math-step">
            <h4>Step 3: Scaled Dot-Product Attention</h4>
            <p>For each head, we compute attention scores:</p>
            <div class="formula">
                scores = (Q · K^T) / √d_k<br>
                where d_k = ${mhsa.headDim}
            </div>
            <p>The scaling factor prevents the dot products from becoming too large.</p>
        </div>

        <div class="math-step">
            <h4>Step 4: Softmax Normalization</h4>
            <p>Apply softmax to get attention weights that sum to 1:</p>
            <div class="formula">
                Attention(Q, K, V) = softmax(scores / temperature) · V<br>
                temperature = ${mhsa.temperature}
            </div>
            <p>Higher temperature makes the distribution more uniform; lower makes it more peaked.</p>
        </div>

        <div class="math-step">
            <h4>Example: Head 1 Detailed Computation</h4>
            ${formatMatrix(head0.Q.map(q => q.slice(0, 4)), 'Q (first 4 dims)')}
            ${formatMatrix(head0.K.map(k => k.slice(0, 4)), 'K (first 4 dims)')}
            ${formatMatrix(head0.scores, 'Attention Scores (scaled)')}
            ${formatMatrix(head0.attentionWeights, 'Attention Weights (after softmax)')}
        </div>

        <div class="math-step">
            <h4>Step 5: Concatenate Heads</h4>
            <p>Concatenate outputs from all ${mhsa.numHeads} heads:</p>
            <div class="formula">
                MultiHead = Concat(head_1, head_2, ..., head_${mhsa.numHeads})<br>
                where each head_i ∈ ℝ^(${tokens.length} × ${mhsa.headDim})
            </div>
            <p>Result: ℝ^(${tokens.length} × ${mhsa.numHeads * mhsa.headDim})</p>
        </div>

        <div class="math-step">
            <h4>Step 6: Output Projection</h4>
            <p>Final linear transformation:</p>
            <div class="formula">
                Output = MultiHead · W^O<br>
                where W^O ∈ ℝ^(${mhsa.numHeads * mhsa.headDim} × ${mhsa.embedDim})
            </div>
            ${formatMatrix(result.output.map(o => o.slice(0, 8)), 'Final Output (first 8 dimensions)')}
        </div>

        <div class="info-box">
            <strong>Key Insights:</strong><br>
            • Each attention head can learn to focus on different relationships<br>
            • The attention weights are position-dependent and learned from data<br>
            • Self-attention allows every token to attend to every other token<br>
            • Scaling by √d_k stabilizes gradients during training<br>
            • The final output combines information from all attention heads
        </div>
    `;
}

/**
 * Generate Q/K/V visualization HTML
 */
function generateQKVVisualization(tokens, embeddings, mhsa, result) {
    const container = document.getElementById('qkvVisualization');
    if (!container) return;
    
    const head0 = result.headOutputs[0];

    container.innerHTML = `
        <div class="math-step">
            <h4>Transformation Pipeline for First Token: "${tokens[0]}"</h4>
            <div class="flow-diagram">
                <div class="flow-step">
                    <div class="flow-box token-box">Token<br>"${tokens[0]}"</div>
                    <small>Input word</small>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="flow-box">Embedding<br>[${mhsa.embedDim}d]</div>
                    <small>Dense vector</small>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="flow-box" style="background: linear-gradient(135deg, #FF6B6B, #EE5A6F);">
                        Query<br>[${mhsa.headDim}d]
                    </div>
                    <small>What to look for</small>
                </div>
            </div>
        </div>

        <div class="meaning-box">
            <h4>🎯 How Meaning Emerges</h4>
            <p><strong>Step 1: Projection Matrices Learn Semantic Roles</strong></p>
            <p>The weight matrices W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> are learned during training. They transform the generic embedding into specialized representations:</p>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li><span class="highlight-query">Query (Q)</span> encodes what information this token needs</li>
                <li><span class="highlight-key">Key (K)</span> encodes what information this token can provide</li>
                <li><span class="highlight-value">Value (V)</span> contains the semantic content to be retrieved</li>
            </ul>
        </div>

        <div class="qkv-grid">
            <div class="qkv-card qkv-query">
                <h3>🔍 Query Vectors (Head 1)</h3>
                <p style="font-size: 0.9em; margin-bottom: 10px;">Q = Embedding × W<sup>Q</sup></p>
                <div class="vector-display">
                    ${tokens.map((token, i) => 
                        `<div class="vector-row"><strong>${token}:</strong> [${head0.Q[i].slice(0, 6).map(v => v.toFixed(3)).join(', ')}...]</div>`
                    ).join('')}
                </div>
                <small>Each row shows what that token is "asking for"</small>
            </div>

            <div class="qkv-card qkv-key">
                <h3>🔑 Key Vectors (Head 1)</h3>
                <p style="font-size: 0.9em; margin-bottom: 10px;">K = Embedding × W<sup>K</sup></p>
                <div class="vector-display">
                    ${tokens.map((token, i) => 
                        `<div class="vector-row"><strong>${token}:</strong> [${head0.K[i].slice(0, 6).map(v => v.toFixed(3)).join(', ')}...]</div>`
                    ).join('')}
                </div>
                <small>Each row shows what that token "offers"</small>
            </div>

            <div class="qkv-card qkv-value">
                <h3>💎 Value Vectors (Head 1)</h3>
                <p style="font-size: 0.9em; margin-bottom: 10px;">V = Embedding × W<sup>V</sup></p>
                <div class="vector-display">
                    ${tokens.map((token, i) => 
                        `<div class="vector-row"><strong>${token}:</strong> [${head0.V[i].slice(0, 6).map(v => v.toFixed(3)).join(', ')}...]</div>`
                    ).join('')}
                </div>
                <small>Each row contains the actual information</small>
            </div>
        </div>

        <div class="math-step">
            <h4>📊 Projection Weight Matrices (Head 1)</h4>
            <p>These matrices transform ${mhsa.embedDim}-dimensional embeddings into ${mhsa.headDim}-dimensional Q/K/V vectors:</p>
            <canvas id="weightMatrixCanvas" class="interaction-canvas"></canvas>
        </div>
    `;

    // Draw weight matrix visualization
    setTimeout(() => {
        const canvas = document.getElementById('weightMatrixCanvas');
        if (canvas) {
            drawWeightMatrixVisualization(canvas, mhsa);
        }
    }, 10);
}

/**
 * Generate attention flow visualization HTML
 */
function generateAttentionFlowVisualization(tokens, embeddings, mhsa, result) {
    const container = document.getElementById('attentionFlow');
    if (!container) return;
    
    const head0 = result.headOutputs[0];
    const focusIdx = Math.floor(tokens.length / 2);
    const focusToken = tokens[focusIdx];

    container.innerHTML = `
        <div class="meaning-box">
            <h4>🧠 How Attention Extracts Meaning: Example with "${focusToken}"</h4>
            <p>Let's trace how token "${focusToken}" (position ${focusIdx + 1}) gathers information from other tokens:</p>
        </div>

        <div class="math-step">
            <h4>Step 1: Compute Attention Scores (Q · K<sup>T</sup>)</h4>
            <p>The <span class="highlight-query">query</span> of "${focusToken}" is compared with the <span class="highlight-key">keys</span> of all tokens via dot product:</p>
            <div class="flow-diagram" style="flex-wrap: wrap;">
                ${tokens.map((token, i) => {
                    const score = head0.scores[focusIdx][i];
                    const weight = head0.attentionWeights[focusIdx][i];
                    return `
                        <div class="flow-step" style="flex: 1; min-width: 120px;">
                            <div class="flow-box" style="background: rgba(102, 126, 234, ${weight});">
                                ${token}
                            </div>
                            <small>Score: ${score.toFixed(3)}<br>Weight: ${weight.toFixed(3)}</small>
                        </div>
                    `;
                }).join('')}
            </div>
            <div class="info-box">
                <strong>What's happening:</strong> High dot product (Q·K) means the query and key are aligned in the learned semantic space.
            </div>
        </div>

        <div class="math-step">
            <h4>Step 2: Softmax Normalization</h4>
            <p>Raw scores are converted to probabilities that sum to 1:</p>
            <canvas id="softmaxCanvas" class="interaction-canvas"></canvas>
        </div>

        <div class="math-step">
            <h4>Step 3: Weighted Aggregation of Values</h4>
            <p>The attention weights determine how much of each token's <span class="highlight-value">value</span> vector to include:</p>
            <div class="formula">
                output[${focusToken}] = ${tokens.map((t, i) => 
                    `${head0.attentionWeights[focusIdx][i].toFixed(2)} × V[${t}]`
                ).join(' + ')}
            </div>
            <p style="margin-top: 10px;">This weighted sum creates a context-aware representation of "${focusToken}" based on the entire sequence.</p>
        </div>

        <div class="math-step">
            <h4>Visualizing the Aggregation</h4>
            <canvas id="aggregationCanvas" class="interaction-canvas"></canvas>
        </div>

        <div class="meaning-box">
            <h4>🎓 Key Insights on Semantic Understanding</h4>
            <ol style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                <li><strong>Learned Representations:</strong> During training, the projection matrices learn to encode linguistic patterns.</li>
                <li><strong>Dynamic Contextualization:</strong> Unlike static word embeddings, attention creates context-dependent representations.</li>
                <li><strong>Multiple Heads = Multiple Perspectives:</strong> With ${mhsa.numHeads} heads, the model can simultaneously attend to different aspects.</li>
                <li><strong>Information Routing:</strong> Attention acts as a soft routing mechanism, allowing information to flow from relevant tokens.</li>
            </ol>
        </div>

        <div class="math-step">
            <h4>🔄 Comparison: Before and After Attention</h4>
            <canvas id="comparisonCanvas" class="interaction-canvas"></canvas>
            <p style="margin-top: 10px; text-align: center;">
                <strong>Before:</strong> Static embedding based only on the token itself<br>
                <strong>After:</strong> Contextualized representation incorporating information from the entire sequence
            </p>
        </div>
    `;

    // Draw visualizations
    setTimeout(() => {
        drawSoftmaxVisualization(
            document.getElementById('softmaxCanvas'),
            tokens, focusIdx, head0
        );
        drawAggregationVisualization(
            document.getElementById('aggregationCanvas'),
            tokens, focusIdx, head0
        );
        drawComparisonVisualization(
            document.getElementById('comparisonCanvas'),
            embeddings, result.output, focusIdx, focusToken
        );
    }, 10);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

export default {
    init,
    runAttention,
    setEmbeddingSource,
    setModelSource
};

