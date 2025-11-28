/**
 * MHSA Visualizer - UI Module
 * DOM interactions, tab management, and UI state
 */

import { Config } from './config.js';
import { generateStepContent } from './steps.js';
import { drawStepSoftmaxCanvas } from './visualization.js';

// UI State
let currentStep = 0;
let currentTab = 'overview';
let currentArchView = 'encoder';
const totalSteps = Config.stepNames.length;

/**
 * Initialize the step navigator
 */
export function initializeStepNavigator() {
    const navGrid = document.getElementById('stepNavGrid');
    if (!navGrid) return;
    
    navGrid.innerHTML = Config.stepNames.map((name, idx) => `
        <div class="nav-step ${idx === 0 ? 'active' : ''}" onclick="window.goToStep(${idx})">
            <div class="nav-step-num">Step ${idx + 1}</div>
            <div class="nav-step-name">${name}</div>
        </div>
    `).join('');
}

/**
 * Render the current step
 * @param {Object} sequenceData - Data for the current sequence
 */
export function renderStep(sequenceData) {
    if (!sequenceData) return;

    const progress = ((currentStep + 1) / totalSteps) * 100;
    
    const progressBar = document.getElementById('progressBar');
    const stepIndicator = document.getElementById('stepIndicator');
    
    if (progressBar) progressBar.style.width = progress + '%';
    if (stepIndicator) stepIndicator.textContent = `Step ${currentStep + 1} of ${totalSteps}`;

    // Update navigator
    document.querySelectorAll('.nav-step').forEach((el, idx) => {
        if (idx === currentStep) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    // Render current step content
    const container = document.getElementById('sequenceVisualization');
    if (container) {
        container.innerHTML = generateStepContent(currentStep, sequenceData);
    }

    // Render step-specific canvases after DOM update
    setTimeout(() => {
        if (currentStep === 5) {
            const canvas = document.getElementById('stepSoftmaxCanvas');
            if (canvas) {
                const head0 = sequenceData.result.headOutputs[0];
                drawStepSoftmaxCanvas(canvas, sequenceData.tokens, sequenceData.focusToken, head0);
            }
        }
    }, 10);
}

/**
 * Go to next step
 * @param {Object} sequenceData - Sequence data
 */
export function nextStep(sequenceData) {
    if (currentStep < totalSteps - 1) {
        currentStep++;
        renderStep(sequenceData);
    }
}

/**
 * Go to previous step
 * @param {Object} sequenceData - Sequence data
 */
export function previousStep(sequenceData) {
    if (currentStep > 0) {
        currentStep--;
        renderStep(sequenceData);
    }
}

/**
 * Go to a specific step
 * @param {number} step - Step number
 * @param {Object} sequenceData - Sequence data
 */
export function goToStep(step, sequenceData) {
    currentStep = step;
    renderStep(sequenceData);
}

/**
 * Reset to first step
 * @param {Object} sequenceData - Sequence data
 */
export function resetSequence(sequenceData) {
    currentStep = 0;
    renderStep(sequenceData);
}

/**
 * Get current step
 * @returns {number} Current step index
 */
export function getCurrentStep() {
    return currentStep;
}

/**
 * Switch tabs
 * @param {string} tabName - Name of tab to switch to
 */
export function switchTab(tabName) {
    currentTab = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.textContent.toLowerCase().includes(tabName.toLowerCase()) ||
            tab.getAttribute('onclick')?.includes(tabName)) {
            tab.classList.add('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    const targetContent = document.getElementById(tabName);
    if (targetContent) {
        targetContent.classList.add('active');
    }
}

/**
 * Set architecture view
 * @param {string} view - View type ('encoder', 'decoder', 'full')
 */
export function setArchView(view) {
    currentArchView = view;
    
    // Update toggle buttons
    document.querySelectorAll('.arch-toggle-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.toLowerCase() === view.toLowerCase()) {
            btn.classList.add('active');
        }
    });
    
    // Re-render architecture view
    renderArchitectureView();
}

/**
 * Render architecture visualization
 */
export function renderArchitectureView() {
    const container = document.getElementById('architectureView');
    if (!container) return;
    
    let html = '';
    
    switch (currentArchView) {
        case 'encoder':
            html = generateEncoderView();
            break;
        case 'decoder':
            html = generateDecoderView();
            break;
        case 'full':
            html = generateFullView();
            break;
    }
    
    container.innerHTML = html;
}

/**
 * Generate encoder view HTML
 */
function generateEncoderView() {
    return `
        <div class="transformer-layer">
            <div class="layer-wrapper highlighted">
                <span class="layer-label">Encoder Block</span>
                <div class="layer-content">
                    <div class="arch-component mha active">
                        <div class="component-icon">🧠</div>
                        <div class="component-name">Multi-Head<br>Self-Attention</div>
                        <div class="component-detail">h heads</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-component norm">
                        <div class="component-icon">+</div>
                        <div class="component-name">Add & Norm</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-component ffn">
                        <div class="component-icon">⚡</div>
                        <div class="component-name">Feed<br>Forward</div>
                        <div class="component-detail">2 linear layers</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-component norm">
                        <div class="component-icon">+</div>
                        <div class="component-name">Add & Norm</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="layer-repeat-indicator">
            <span>↑ Repeat</span>
            <strong>N×</strong>
            <span>(typically 6-12 layers)</span>
        </div>
    `;
}

/**
 * Generate decoder view HTML
 */
function generateDecoderView() {
    return `
        <div class="transformer-layer">
            <div class="layer-wrapper">
                <span class="layer-label">Decoder Block</span>
                <div class="layer-content" style="flex-direction: column; gap: 15px;">
                    <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                        <div class="arch-component mha">
                            <div class="component-icon">🧠</div>
                            <div class="component-name">Masked<br>Self-Attention</div>
                            <div class="component-detail">causal mask</div>
                        </div>
                        <div class="arch-arrow">→</div>
                        <div class="arch-component norm">
                            <div class="component-icon">+</div>
                            <div class="component-name">Add & Norm</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                        <div class="arch-component mha active">
                            <div class="component-icon">🔗</div>
                            <div class="component-name">Cross<br>Attention</div>
                            <div class="component-detail">K,V from encoder</div>
                        </div>
                        <div class="arch-arrow">→</div>
                        <div class="arch-component norm">
                            <div class="component-icon">+</div>
                            <div class="component-name">Add & Norm</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                        <div class="arch-component ffn">
                            <div class="component-icon">⚡</div>
                            <div class="component-name">Feed Forward</div>
                        </div>
                        <div class="arch-arrow">→</div>
                        <div class="arch-component norm">
                            <div class="component-icon">+</div>
                            <div class="component-name">Add & Norm</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="layer-repeat-indicator">
            <span>↑ Repeat</span>
            <strong>N×</strong>
            <span>(typically 6-12 layers)</span>
        </div>
    `;
}

/**
 * Generate full transformer view HTML
 */
function generateFullView() {
    return `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; width: 100%;">
            <div>
                <div style="text-align: center; color: #00d4ff; font-weight: 700; margin-bottom: 15px; font-size: 14px;">ENCODER</div>
                <div class="transformer-layer">
                    <div class="layer-wrapper">
                        <span class="layer-label">Encoder ×N</span>
                        <div class="layer-content" style="flex-direction: column; gap: 10px;">
                            <div class="arch-component mha" style="width: 100%;">
                                <div class="component-name">Self-Attention</div>
                            </div>
                            <div class="arch-component ffn" style="width: 100%;">
                                <div class="component-name">FFN</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div>
                <div style="text-align: center; color: #00d4ff; font-weight: 700; margin-bottom: 15px; font-size: 14px;">DECODER</div>
                <div class="transformer-layer">
                    <div class="layer-wrapper">
                        <span class="layer-label">Decoder ×N</span>
                        <div class="layer-content" style="flex-direction: column; gap: 10px;">
                            <div class="arch-component mha" style="width: 100%;">
                                <div class="component-name">Masked Self-Attn</div>
                            </div>
                            <div class="arch-component mha active" style="width: 100%;">
                                <div class="component-name">Cross-Attention</div>
                            </div>
                            <div class="arch-component ffn" style="width: 100%;">
                                <div class="component-name">FFN</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 20px; color: rgba(255,255,255,0.6);">
            <span style="color: #667eea;">━━━━━</span> 
            <span style="color: #00d4ff;">Cross-Attention connects Encoder ↔ Decoder</span>
            <span style="color: #667eea;">━━━━━</span>
        </div>
    `;
}

/**
 * Render heads parallel view
 * @param {number} numHeads - Number of attention heads
 * @param {number} headDim - Dimension per head
 */
export function renderHeadsParallelView(numHeads, headDim) {
    const container = document.getElementById('headsParallelView');
    if (!container) return;
    
    container.innerHTML = Array.from({length: numHeads}, (_, h) => `
        <div class="head-unit" onclick="window.selectHead(${h})">
            <div class="head-number">Head ${h + 1}</div>
            <div class="head-qkv">
                <span class="head-qkv-item q">Q</span>
                <span class="head-qkv-item k">K</span>
                <span class="head-qkv-item v">V</span>
            </div>
            <div class="head-dim">${headDim}d</div>
        </div>
    `).join('');
}

/**
 * Render architecture stats
 * @param {Object} mhsa - MHSA instance
 * @param {number} seqLen - Sequence length
 */
export function renderArchStats(mhsa, seqLen) {
    const container = document.getElementById('archStats');
    if (!container) return;
    
    container.innerHTML = `
        <div class="arch-stat">
            <div class="arch-stat-value">${seqLen}</div>
            <div class="arch-stat-label">Sequence Length</div>
        </div>
        <div class="arch-stat">
            <div class="arch-stat-value">${mhsa.embedDim}</div>
            <div class="arch-stat-label">d_model</div>
        </div>
        <div class="arch-stat">
            <div class="arch-stat-value">${mhsa.numHeads}</div>
            <div class="arch-stat-label">Attention Heads</div>
        </div>
        <div class="arch-stat">
            <div class="arch-stat-value">${mhsa.headDim}</div>
            <div class="arch-stat-label">d_k per head</div>
        </div>
    `;
}

/**
 * Update flow dimensions in the architecture view
 * @param {number} seqLen - Sequence length
 * @param {number} embedDim - Embedding dimension
 * @param {number} numHeads - Number of heads
 * @param {number} headDim - Head dimension
 */
export function updateFlowDimensions(seqLen, embedDim, numHeads, headDim) {
    const elements = {
        'flowInputDim': `[${seqLen} × ${embedDim}]`,
        'flowHeadsDim': `[${numHeads} × ${seqLen} × ${headDim}]`,
        'flowOutputDim': `[${seqLen} × ${embedDim}]`
    };
    
    for (const [id, value] of Object.entries(elements)) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }
}

/**
 * Show loading indicator
 * @param {string} message - Loading message
 */
export function showLoading(message = 'Loading...') {
    const content = document.querySelector('.content');
    if (!content) return;
    
    const existingLoader = document.querySelector('.loading-overlay');
    if (existingLoader) return;
    
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;
    overlay.innerHTML = `
        <div class="loading-indicator" style="background: white; padding: 30px 50px; border-radius: 12px;">
            <div class="loading-spinner"></div>
            <div class="loading-text">${message}</div>
        </div>
    `;
    document.body.appendChild(overlay);
}

/**
 * Hide loading indicator
 */
export function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Update model status display
 * @param {string} status - Status message
 * @param {string} type - Status type ('loading', 'loaded', 'error')
 */
export function updateModelStatus(status, type = 'info') {
    const statusEl = document.getElementById('modelStatus');
    if (!statusEl) return;
    
    statusEl.textContent = status;
    statusEl.className = `model-status ${type}`;
}

/**
 * Display tokens
 * @param {string[]} tokens - Array of tokens
 */
export function displayTokens(tokens) {
    const container = document.getElementById('tokenDisplay');
    if (!container) return;
    
    container.innerHTML = tokens.map(
        (t, i) => `<div class="token">${i+1}. ${t}</div>`
    ).join('');
}

/**
 * Display architecture info
 * @param {Object} mhsa - MHSA instance
 * @param {number} seqLen - Sequence length
 */
export function displayArchitectureInfo(mhsa, seqLen) {
    const container = document.getElementById('architectureInfo');
    if (!container) return;
    
    container.innerHTML = `
        <div class="info-box">
            <strong>Configuration:</strong><br>
            • Sequence Length: ${seqLen} tokens<br>
            • Embedding Dimension (d_model): ${mhsa.embedDim}<br>
            • Number of Heads: ${mhsa.numHeads}<br>
            • Head Dimension (d_k = d_model/num_heads): ${mhsa.headDim}<br>
            • Temperature: ${mhsa.temperature}
            ${mhsa.isRealModel ? `<br>• <strong style="color: #28a745;">Using Real Model: ${mhsa.modelName}</strong>` : ''}
        </div>
    `;
}

export default {
    initializeStepNavigator,
    renderStep,
    nextStep,
    previousStep,
    goToStep,
    resetSequence,
    getCurrentStep,
    switchTab,
    setArchView,
    renderArchitectureView,
    renderHeadsParallelView,
    renderArchStats,
    updateFlowDimensions,
    showLoading,
    hideLoading,
    updateModelStatus,
    displayTokens,
    displayArchitectureInfo
};

