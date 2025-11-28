/**
 * MHSA Visualizer - Step Content Generation Module
 * Generates HTML content for each step of the visualization sequence
 */

import { Config } from './config.js';

/**
 * Generate content for Step 0: Input Tokens
 */
export function generateStep0_InputTokens(tokens) {
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">1</span>
                Input Tokens
            </div>
            <div class="step-description">
                We start with a sequence of tokens (words or subwords) from our input text.
                Each token will be processed independently but will interact through attention.
            </div>

            <div class="dimension-tracker">
                <h4>📐 Current Data Dimensions</h4>
                <div class="dim-row">
                    <div class="dim-badge">
                        <span class="dim-label">Sequence Length:</span>
                        <span class="dim-shape">n = ${tokens.length}</span>
                    </div>
                    <div class="dim-badge">
                        <span class="dim-label">Data Type:</span>
                        <span class="dim-shape">Discrete Tokens</span>
                    </div>
                </div>
            </div>

            <div class="computation-visual">
                <h4 style="margin-bottom: 15px;">Input Sequence</h4>
                <div class="data-flow">
                    ${tokens.map((token, idx) => `
                        <div class="data-box active">
                            <div class="data-label">Token ${idx + 1}</div>
                            <div class="data-value">"${token}"</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> Raw text string</li>
                    <li><strong>Output:</strong> ${tokens.length} discrete tokens</li>
                    <li><strong>Next:</strong> Convert tokens to dense vector embeddings</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p>Each token is currently just a symbol (like a word). Before the attention mechanism can work, 
                we need to convert these into continuous vectors that capture semantic meaning.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 1: Token Embeddings
 */
export function generateStep1_Embeddings(tokens, embeddings, mhsa) {
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">2</span>
                Token Embeddings
            </div>
            <div class="step-description">
                Each token is converted into a dense vector of dimension d<sub>model</sub> = ${mhsa.embedDim}.
                This embedding captures semantic information about the token.
            </div>

            <div class="dimension-tracker">
                <h4>📐 Current Data Dimensions</h4>
                <div class="dim-row">
                    <div class="dim-badge embedding">
                        <span class="dim-label">Embedding Matrix X:</span>
                        <span class="dim-shape">[${tokens.length} × ${mhsa.embedDim}]</span>
                    </div>
                </div>
                <div class="dim-row">
                    <div class="dim-badge">
                        <span class="dim-label">${tokens.length} rows</span>
                        <span class="dim-shape">= ${tokens.length} tokens</span>
                    </div>
                    <div class="dim-badge">
                        <span class="dim-label">${mhsa.embedDim} columns</span>
                        <span class="dim-shape">= embedding dimension (d_model)</span>
                    </div>
                </div>
            </div>

            <div class="matrix-shape-display">
                <div class="matrix-block">
                    <div class="matrix-rect medium embedding" style="width: 40px; height: 80px;">
                        <span class="matrix-dim-label top">${tokens.length}</span>
                        <span class="matrix-dim-label right">${mhsa.embedDim}</span>
                        X
                    </div>
                    <div class="matrix-name">Embedding Matrix</div>
                    <div class="matrix-dims">[n × d_model]</div>
                </div>
            </div>

            <div class="computation-visual">
                <h4 style="margin-bottom: 15px;">Embedding Transformation</h4>
                ${tokens.slice(0, 4).map((token, idx) => `
                    <div class="token-embedding-viz">
                        <div class="token-box">${token}</div>
                        <div class="projection-arrow">→</div>
                        <div class="embedding-viz">
                            ${embeddings[idx].slice(0, 16).map(val => 
                                `<div class="embedding-bar" style="height: ${Math.abs(val) * 60 + 15}px;"></div>`
                            ).join('')}
                        </div>
                        <small style="margin-left: 10px; font-weight: 600; color: #667eea;">[${mhsa.embedDim}d vector]</small>
                    </div>
                `).join('')}
                ${tokens.length > 4 ? `<div style="text-align: center; padding: 10px; color: #6c757d;">... and ${tokens.length - 4} more tokens</div>` : ''}
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> ${tokens.length} discrete tokens</li>
                    <li><strong>Operation:</strong> Lookup in embedding table</li>
                    <li><strong>Output:</strong> Matrix X of shape <code>[${tokens.length} × ${mhsa.embedDim}]</code></li>
                    <li><strong>Next:</strong> Project embeddings to Q, K, V spaces</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p>Each token is now a ${mhsa.embedDim}-dimensional vector. Semantically similar words will have similar vectors. 
                This dense representation allows mathematical operations on word meanings.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 2: Linear Projections
 */
export function generateStep2_Projections(tokens, embeddings, mhsa, head0) {
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">3</span>
                Linear Projections to Q, K, V
            </div>
            <div class="step-description">
                Each embedding is transformed into three different representations using learned weight matrices:
                <strong>Query (Q)</strong>, <strong>Key (K)</strong>, and <strong>Value (V)</strong>.
            </div>

            <div class="dimension-tracker">
                <h4>📐 Matrix Dimensions (Per Head)</h4>
                <div class="dim-row">
                    <div class="dim-badge embedding">
                        <span class="dim-label">Input X:</span>
                        <span class="dim-shape">[${tokens.length} × ${mhsa.embedDim}]</span>
                    </div>
                </div>
                <div class="dim-row">
                    <div class="dim-badge query">
                        <span class="dim-label">W<sup>Q</sup>:</span>
                        <span class="dim-shape">[${mhsa.embedDim} × ${mhsa.headDim}]</span>
                    </div>
                    <div class="dim-badge key">
                        <span class="dim-label">W<sup>K</sup>:</span>
                        <span class="dim-shape">[${mhsa.embedDim} × ${mhsa.headDim}]</span>
                    </div>
                    <div class="dim-badge value">
                        <span class="dim-label">W<sup>V</sup>:</span>
                        <span class="dim-shape">[${mhsa.embedDim} × ${mhsa.headDim}]</span>
                    </div>
                </div>
                <div class="dim-row">
                    <div class="dim-badge query">
                        <span class="dim-label">Q = X·W<sup>Q</sup>:</span>
                        <span class="dim-shape">[${tokens.length} × ${mhsa.headDim}]</span>
                    </div>
                    <div class="dim-badge key">
                        <span class="dim-label">K = X·W<sup>K</sup>:</span>
                        <span class="dim-shape">[${tokens.length} × ${mhsa.headDim}]</span>
                    </div>
                    <div class="dim-badge value">
                        <span class="dim-label">V = X·W<sup>V</sup>:</span>
                        <span class="dim-shape">[${tokens.length} × ${mhsa.headDim}]</span>
                    </div>
                </div>
            </div>

            <div class="qkv-grid">
                <div class="qkv-card qkv-query">
                    <h3>🔍 Queries <span style="font-weight: normal; font-size: 12px;">[${tokens.length}×${mhsa.headDim}]</span></h3>
                    <div class="mini-matrix">
                        ${tokens.slice(0, 3).map((t, i) => 
                            `<strong>${t}:</strong> [${head0.Q[i].slice(0, 4).map(v => v.toFixed(2)).join(', ')}...]`
                        ).join('<br>')}
                        ${tokens.length > 3 ? '<br>...' : ''}
                    </div>
                </div>
                <div class="qkv-card qkv-key">
                    <h3>🔑 Keys <span style="font-weight: normal; font-size: 12px;">[${tokens.length}×${mhsa.headDim}]</span></h3>
                    <div class="mini-matrix">
                        ${tokens.slice(0, 3).map((t, i) => 
                            `<strong>${t}:</strong> [${head0.K[i].slice(0, 4).map(v => v.toFixed(2)).join(', ')}...]`
                        ).join('<br>')}
                        ${tokens.length > 3 ? '<br>...' : ''}
                    </div>
                </div>
                <div class="qkv-card qkv-value">
                    <h3>💎 Values <span style="font-weight: normal; font-size: 12px;">[${tokens.length}×${mhsa.headDim}]</span></h3>
                    <div class="mini-matrix">
                        ${tokens.slice(0, 3).map((t, i) => 
                            `<strong>${t}:</strong> [${head0.V[i].slice(0, 4).map(v => v.toFixed(2)).join(', ')}...]`
                        ).join('<br>')}
                        ${tokens.length > 3 ? '<br>...' : ''}
                    </div>
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> Embedding matrix X <code>[${tokens.length} × ${mhsa.embedDim}]</code></li>
                    <li><strong>Operation:</strong> Three matrix multiplications (one per Q, K, V)</li>
                    <li><strong>Output:</strong> Q, K, V matrices, each <code>[${tokens.length} × ${mhsa.headDim}]</code></li>
                    <li><strong>Per Head:</strong> Dimension reduced from ${mhsa.embedDim} → ${mhsa.headDim}</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p><strong>Query:</strong> "What am I looking for?" | <strong>Key:</strong> "What do I offer?" | <strong>Value:</strong> "What's my content?"
                <br>These three projections let each token play different roles in the attention mechanism.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 3: Dot Product
 */
export function generateStep3_DotProduct(tokens, head0, focusToken) {
    const focusTokenName = tokens[focusToken];
    const scores = tokens.map((_, idx) => 
        head0.Q[focusToken].reduce((sum, q, i) => sum + q * head0.K[idx][i], 0)
    );
    const maxScore = Math.max(...scores.map(Math.abs));
    
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">4</span>
                Compute Query-Key Dot Product (Q · K<sup>T</sup>)
            </div>
            <div class="step-description">
                For each query vector, we compute its dot product with all key vectors.
                This measures how similar/relevant each key is to the query.
            </div>

            <div class="dimension-tracker">
                <h4>📐 Matrix Dimensions</h4>
                <div class="dim-row">
                    <div class="dim-badge query">
                        <span class="dim-label">Q:</span>
                        <span class="dim-shape">[${tokens.length} × ${head0.Q[0].length}]</span>
                    </div>
                    <div class="dim-operator">×</div>
                    <div class="dim-badge key">
                        <span class="dim-label">K<sup>T</sup>:</span>
                        <span class="dim-shape">[${head0.K[0].length} × ${tokens.length}]</span>
                    </div>
                    <div class="dim-operator">=</div>
                    <div class="dim-badge scores">
                        <span class="dim-label">Scores:</span>
                        <span class="dim-shape">[${tokens.length} × ${tokens.length}]</span>
                    </div>
                </div>
            </div>

            <div class="matrix-visual-box" data-label="Focus: Token '${focusTokenName}' (Row ${focusToken + 1})">
                <h4 style="margin-bottom: 15px; color: #667eea;">Computing one row of the score matrix:</h4>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;">
                        <div style="background: #FF6B6B; color: white; padding: 8px 16px; border-radius: 6px; font-weight: 600;">
                            Query: "${focusTokenName}"
                        </div>
                        <span style="font-size: 20px; color: #667eea;">⊙</span>
                        <span style="color: #4ECDC4; font-weight: 600;">each Key</span>
                        <span style="font-size: 20px; color: #667eea;">→</span>
                        <span style="color: #ffc107; font-weight: 600;">Similarity Score</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;">
                        ${tokens.map((token, idx) => {
                            const score = scores[idx];
                            const normalizedScore = Math.abs(score) / (maxScore + 0.001);
                            const bgColor = `rgba(255, 193, 7, ${0.1 + normalizedScore * 0.6})`;
                            return `
                                <div style="background: ${bgColor}; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid ${normalizedScore > 0.7 ? '#ffc107' : '#dee2e6'};">
                                    <div style="font-weight: 600; color: #4ECDC4; font-size: 13px;">${token}</div>
                                    <div style="font-size: 18px; font-weight: 700; color: #333; margin-top: 5px;">${score.toFixed(3)}</div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> Q <code>[${tokens.length}×${head0.Q[0].length}]</code> and K <code>[${tokens.length}×${head0.K[0].length}]</code></li>
                    <li><strong>Operation:</strong> Matrix multiplication Q × K<sup>T</sup></li>
                    <li><strong>Output:</strong> Score matrix <code>[${tokens.length} × ${tokens.length}]</code> (one score per token pair)</li>
                    <li><strong>Meaning:</strong> scores[i][j] = how relevant token j is to token i</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p>Each query asks "who is relevant to me?" and each key answers. 
                High dot product = high similarity = that token is relevant. 
                The result is an <strong>${tokens.length}×${tokens.length}</strong> matrix where each row shows one token's "relevance scores" for all other tokens.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 4: Scaling
 */
export function generateStep4_Scaling(tokens, head0, mhsa, focusToken) {
    const focusTokenName = tokens[focusToken];
    const scale = Math.sqrt(mhsa.headDim);
    const rawScores = tokens.map((_, idx) => 
        head0.Q[focusToken].reduce((sum, q, i) => sum + q * head0.K[idx][i], 0)
    );
    
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">5</span>
                Scale by √d<sub>k</sub>
            </div>
            <div class="step-description">
                Divide attention scores by √d<sub>k</sub> = √${mhsa.headDim} = <strong>${scale.toFixed(2)}</strong>.
                This prevents dot products from becoming too large before softmax.
            </div>

            <div class="dimension-tracker">
                <h4>📐 Matrix Dimensions (Unchanged)</h4>
                <div class="dim-row">
                    <div class="dim-badge scores">
                        <span class="dim-label">Raw Scores:</span>
                        <span class="dim-shape">[${tokens.length} × ${tokens.length}]</span>
                    </div>
                    <div class="dim-operator">÷</div>
                    <div class="dim-badge">
                        <span class="dim-label">Scale Factor:</span>
                        <span class="dim-shape">√${mhsa.headDim} = ${scale.toFixed(2)}</span>
                    </div>
                    <div class="dim-operator">=</div>
                    <div class="dim-badge scores">
                        <span class="dim-label">Scaled Scores:</span>
                        <span class="dim-shape">[${tokens.length} × ${tokens.length}]</span>
                    </div>
                </div>
            </div>

            <div class="matrix-visual-box" data-label="Before vs After Scaling (Row: '${focusTokenName}')">
                <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 20px; align-items: start;">
                    <div>
                        <h4 style="color: #dc3545; margin-bottom: 15px;">❌ Before Scaling</h4>
                        <div style="display: grid; gap: 8px;">
                            ${tokens.map((token, idx) => `
                                <div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: #fff5f5; border-radius: 6px;">
                                    <span style="font-weight: 600; min-width: 50px;">${token}</span>
                                    <div style="flex: 1; height: 8px; background: #eee; border-radius: 4px; overflow: hidden;">
                                        <div style="width: ${Math.min(Math.abs(rawScores[idx]) * 20, 100)}%; height: 100%; background: #dc3545;"></div>
                                    </div>
                                    <span style="font-family: monospace; font-size: 12px;">${rawScores[idx].toFixed(3)}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px;">
                        <div style="font-size: 28px; color: #667eea;">÷ ${scale.toFixed(1)}</div>
                        <div style="font-size: 32px; color: #667eea; margin-top: 10px;">→</div>
                    </div>
                    <div>
                        <h4 style="color: #28a745; margin-bottom: 15px;">✓ After Scaling</h4>
                        <div style="display: grid; gap: 8px;">
                            ${tokens.map((token, idx) => `
                                <div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: #f0fff4; border-radius: 6px;">
                                    <span style="font-weight: 600; min-width: 50px;">${token}</span>
                                    <div style="flex: 1; height: 8px; background: #eee; border-radius: 4px; overflow: hidden;">
                                        <div style="width: ${Math.min(Math.abs(head0.scores[focusToken][idx]) * 40, 100)}%; height: 100%; background: #28a745;"></div>
                                    </div>
                                    <span style="font-family: monospace; font-size: 12px;">${head0.scores[focusToken][idx].toFixed(3)}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> Raw score matrix <code>[${tokens.length} × ${tokens.length}]</code></li>
                    <li><strong>Operation:</strong> Divide each element by √${mhsa.headDim} = ${scale.toFixed(2)}</li>
                    <li><strong>Output:</strong> Scaled score matrix <code>[${tokens.length} × ${tokens.length}]</code></li>
                    <li><strong>Why:</strong> Prevents extreme values that would cause vanishing gradients</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p>As the dimension d_k gets larger, dot products tend to get larger too. 
                Without scaling, softmax would produce nearly one-hot distributions (all attention on one token). 
                Scaling keeps the scores in a reasonable range for stable training.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 5: Softmax
 */
export function generateStep5_Softmax(tokens, head0, focusToken) {
    const focusTokenName = tokens[focusToken];
    const weights = head0.attentionWeights[focusToken];
    const maxWeight = Math.max(...weights);
    
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">6</span>
                Apply Softmax Normalization
            </div>
            <div class="step-description">
                Convert scaled scores to a probability distribution using softmax.
                The attention weights now sum to 1 and represent "how much to attend to each token."
            </div>

            <div class="dimension-tracker">
                <h4>📐 Matrix Dimensions</h4>
                <div class="dim-row">
                    <div class="dim-badge scores">
                        <span class="dim-label">Scaled Scores:</span>
                        <span class="dim-shape">[${tokens.length} × ${tokens.length}]</span>
                    </div>
                    <div class="dim-operator">→</div>
                    <div class="dim-badge">
                        <span class="dim-label">softmax()</span>
                        <span class="dim-shape">row-wise</span>
                    </div>
                    <div class="dim-operator">→</div>
                    <div class="dim-badge attention">
                        <span class="dim-label">Attention Weights:</span>
                        <span class="dim-shape">[${tokens.length} × ${tokens.length}]</span>
                    </div>
                </div>
            </div>

            <div class="computation-visual">
                <h4 style="margin-bottom: 20px;">Softmax Transformation for "${focusTokenName}"</h4>
                <canvas id="stepSoftmaxCanvas" width="600" height="300" style="display: block; margin: 0 auto;"></canvas>
            </div>

            <div class="matrix-visual-box" data-label="Attention Distribution for '${focusTokenName}'">
                <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; padding: 10px;">
                    ${tokens.map((token, idx) => {
                        const weight = weights[idx];
                        const barHeight = Math.max(weight / maxWeight * 100, 5);
                        const isHighest = weight === maxWeight;
                        return `
                            <div style="display: flex; flex-direction: column; align-items: center; min-width: 70px;">
                                <div style="height: 120px; display: flex; align-items: flex-end;">
                                    <div style="width: 45px; height: ${barHeight}%; background: ${isHighest ? 'linear-gradient(to top, #667eea, #764ba2)' : 'linear-gradient(to top, #a8c0ff, #3f5efb)'}; border-radius: 6px 6px 0 0; transition: height 0.3s;"></div>
                                </div>
                                <div style="font-weight: 700; margin-top: 8px; font-size: 11px; color: #495057;">${token}</div>
                                <div style="font-size: 14px; font-weight: 700; color: ${isHighest ? '#667eea' : '#6c757d'};">${(weight * 100).toFixed(1)}%</div>
                            </div>
                        `;
                    }).join('')}
                </div>
                <div style="text-align: center; margin-top: 15px; padding: 12px; background: #e7f3ff; border-radius: 8px; font-weight: 600;">
                    ✓ Sum of all weights = ${weights.reduce((a, b) => a + b, 0).toFixed(4)} ≈ 1.0
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> Scaled scores <code>[${tokens.length} × ${tokens.length}]</code></li>
                    <li><strong>Operation:</strong> Apply softmax to each row independently</li>
                    <li><strong>Output:</strong> Attention weights <code>[${tokens.length} × ${tokens.length}]</code></li>
                    <li><strong>Property:</strong> Each row is a probability distribution (sums to 1)</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p>Softmax converts raw scores into probabilities. For token "${focusTokenName}", it pays 
                <strong>${(maxWeight * 100).toFixed(1)}%</strong> attention to the most relevant token. 
                All ${tokens.length} weights sum to exactly 1, creating a weighted average in the next step.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 6: Attention × Values
 */
export function generateStep6_AttentionValues(tokens, head0, focusToken) {
    const focusTokenName = tokens[focusToken];
    const weights = head0.attentionWeights[focusToken];
    const headDim = head0.V[0].length;
    
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">7</span>
                Multiply Attention × Values
            </div>
            <div class="step-description">
                Use attention weights to compute a weighted sum of value vectors.
                This creates a <strong>context-aware representation</strong> for each token.
            </div>

            <div class="dimension-tracker">
                <h4>📐 Matrix Dimensions</h4>
                <div class="dim-row">
                    <div class="dim-badge attention">
                        <span class="dim-label">Attention A:</span>
                        <span class="dim-shape">[${tokens.length} × ${tokens.length}]</span>
                    </div>
                    <div class="dim-operator">×</div>
                    <div class="dim-badge value">
                        <span class="dim-label">Values V:</span>
                        <span class="dim-shape">[${tokens.length} × ${headDim}]</span>
                    </div>
                    <div class="dim-operator">=</div>
                    <div class="dim-badge output">
                        <span class="dim-label">Head Output:</span>
                        <span class="dim-shape">[${tokens.length} × ${headDim}]</span>
                    </div>
                </div>
            </div>

            <div class="matrix-visual-box" data-label="Weighted Sum for '${focusTokenName}'">
                <div style="padding: 10px;">
                    <div style="text-align: center; margin-bottom: 20px; font-size: 14px; color: #495057;">
                        <strong>output[${focusTokenName}]</strong> = weighted combination of all value vectors
                    </div>
                    
                    ${tokens.map((token, idx) => {
                        const weight = weights[idx];
                        const isSignificant = weight > 0.1;
                        return `
                            <div style="display: grid; grid-template-columns: 80px 100px 1fr 180px; gap: 15px; align-items: center; 
                                        margin: 12px 0; padding: 12px 15px; 
                                        background: ${isSignificant ? 'linear-gradient(90deg, rgba(102,126,234,0.1), rgba(102,126,234,0.05))' : '#f8f9fa'}; 
                                        border-radius: 8px; border-left: 4px solid ${isSignificant ? '#667eea' : '#dee2e6'};">
                                    <div style="font-weight: 700; color: ${isSignificant ? '#667eea' : '#6c757d'};">${token}</div>
                                    <div style="text-align: center;">
                                        <div style="font-size: 18px; font-weight: 700; color: ${isSignificant ? '#667eea' : '#aaa'};">
                                            ${(weight * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div style="position: relative; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden;">
                                        <div style="position: absolute; left: 0; top: 0; height: 100%; width: ${weight * 100}%; 
                                                    background: ${isSignificant ? 'linear-gradient(90deg, #667eea, #764ba2)' : '#adb5bd'}; 
                                                    border-radius: 10px; transition: width 0.3s;"></div>
                                    </div>
                                    <div style="font-family: monospace; font-size: 10px; color: #6c757d; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                        × [${head0.V[idx].slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]
                                    </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>

            <div class="result-box">
                <h4 style="margin-bottom: 10px;">🎯 Output Vector for "${focusTokenName}"</h4>
                <div style="font-family: monospace; font-size: 13px; background: rgba(0,0,0,0.1); padding: 12px; border-radius: 6px; margin: 10px 0;">
                    [${head0.output[focusToken].slice(0, 8).map(v => v.toFixed(3)).join(', ')}...]
                </div>
                <div style="margin-top: 10px; font-size: 14px;">
                    Dimension: <strong>${headDim}</strong> (same as d_k)
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> Attention weights <code>[${tokens.length}×${tokens.length}]</code> and Values <code>[${tokens.length}×${headDim}]</code></li>
                    <li><strong>Operation:</strong> Matrix multiplication A × V</li>
                    <li><strong>Output:</strong> Head output <code>[${tokens.length} × ${headDim}]</code></li>
                    <li><strong>Each row:</strong> Weighted average of all value vectors</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p>This is the core of attention: each token's output is a weighted combination of <em>all</em> value vectors. 
                High attention weight = more contribution. Token "${focusTokenName}" now contains information 
                from tokens it deemed relevant, creating a <strong>context-aware representation</strong>.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 7: Concatenate Heads
 */
export function generateStep7_Concatenate(tokens, result, mhsa) {
    const headDim = mhsa.headDim;
    const concatDim = mhsa.numHeads * headDim;
    const headColors = Config.colors.heads;
    
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">8</span>
                Concatenate Multiple Heads
            </div>
            <div class="step-description">
                Each of the <strong>${mhsa.numHeads} heads</strong> has produced its own output.
                We concatenate these outputs along the feature dimension to combine all perspectives.
            </div>

            <div class="dimension-tracker">
                <h4>📐 Matrix Dimensions</h4>
                <div class="dim-row">
                    ${Array.from({length: Math.min(mhsa.numHeads, 4)}, (_, h) => `
                        <div class="dim-badge" style="border-color: ${headColors[h]}; background: ${headColors[h]}22;">
                            <span class="dim-label">Head ${h + 1}:</span>
                            <span class="dim-shape">[${tokens.length}×${headDim}]</span>
                        </div>
                    `).join('')}
                    ${mhsa.numHeads > 4 ? `<div class="dim-badge"><span class="dim-shape">... +${mhsa.numHeads - 4} more</span></div>` : ''}
                </div>
                <div class="dim-row">
                    <div class="dim-operator">⊕ CONCAT ⊕</div>
                </div>
                <div class="dim-row">
                    <div class="dim-badge output">
                        <span class="dim-label">Combined:</span>
                        <span class="dim-shape">[${tokens.length} × ${concatDim}]</span>
                    </div>
                    <div class="dim-badge">
                        <span class="dim-label">Calculation:</span>
                        <span class="dim-shape">${mhsa.numHeads} heads × ${headDim} dims = ${concatDim}</span>
                    </div>
                </div>
            </div>

            <div class="matrix-visual-box" data-label="Visual: Concatenation Process">
                <div style="text-align: center; margin-bottom: 20px;">
                    <strong>Each token gets a longer vector by joining all head outputs:</strong>
                </div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 15px; flex-wrap: wrap; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    ${Array.from({length: mhsa.numHeads}, (_, h) => `
                        <div style="width: ${Math.max(30, 80/mhsa.numHeads)}px; height: 80px; background: ${headColors[h]}; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px;">
                            ${headDim}d
                        </div>
                    `).join('<div style="font-size: 20px; color: #667eea;">+</div>')}
                    <div style="font-size: 28px; color: #667eea; margin: 0 15px;">=</div>
                    <div style="width: 120px; height: 80px; background: linear-gradient(90deg, ${headColors.slice(0, mhsa.numHeads).join(', ')}); border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 14px;">
                        ${concatDim}d
                    </div>
                </div>
            </div>

            <div style="margin: 30px 0;">
                <h4 style="color: #667eea; margin-bottom: 15px;">🧠 Why Multiple Heads?</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
                    ${Array.from({length: Math.min(mhsa.numHeads, 4)}, (_, h) => {
                        const focuses = [
                            'Syntactic patterns<br>(grammar, structure)',
                            'Semantic relations<br>(meaning, concepts)',
                            'Long-range deps<br>(distant context)',
                            'Local context<br>(adjacent words)'
                        ];
                        return `
                            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid ${headColors[h]};">
                                <strong style="color: ${headColors[h]};">Head ${h + 1}</strong><br>
                                <span style="font-size: 12px; color: #6c757d;">${focuses[h]}</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Step Summary</h4>
                <ul>
                    <li><strong>Input:</strong> ${mhsa.numHeads} head outputs, each <code>[${tokens.length}×${headDim}]</code></li>
                    <li><strong>Operation:</strong> Concatenate along feature dimension (axis=1)</li>
                    <li><strong>Output:</strong> Combined matrix <code>[${tokens.length} × ${concatDim}]</code></li>
                    <li><strong>Purpose:</strong> Combine different attention perspectives</li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 What's Happening</h4>
                <p>By running ${mhsa.numHeads} attention mechanisms in parallel, each head can learn to focus on different aspects. 
                Concatenation preserves all this information. One head might capture syntax, another semantics, 
                another long-range dependencies—all combined into one rich representation.</p>
            </div>
        </div>
    `;
}

/**
 * Generate content for Step 8: Final Output
 */
export function generateStep8_FinalOutput(tokens, embeddings, result) {
    const numHeads = result.headOutputs.length;
    const headDim = result.headOutputs[0].output[0].length;
    const concatDim = numHeads * headDim;
    const embedDim = embeddings[0].length;
    
    return `
        <div class="step-content">
            <div class="step-title">
                <span class="step-number">9</span>
                Final Output Projection
            </div>
            <div class="step-description">
                Apply a final linear transformation W<sup>O</sup> to project the concatenated heads
                back to the original embedding dimension. This completes the multi-head self-attention!
            </div>

            <div class="dimension-tracker">
                <h4>📐 Final Matrix Dimensions</h4>
                <div class="dim-row">
                    <div class="dim-badge" style="border-color: #764ba2; background: rgba(118,75,162,0.15);">
                        <span class="dim-label">Concatenated:</span>
                        <span class="dim-shape">[${tokens.length} × ${concatDim}]</span>
                    </div>
                    <div class="dim-operator">×</div>
                    <div class="dim-badge">
                        <span class="dim-label">W<sup>O</sup>:</span>
                        <span class="dim-shape">[${concatDim} × ${embedDim}]</span>
                    </div>
                    <div class="dim-operator">=</div>
                    <div class="dim-badge output">
                        <span class="dim-label">Final Output:</span>
                        <span class="dim-shape">[${tokens.length} × ${embedDim}]</span>
                    </div>
                </div>
                <div class="dim-row" style="margin-top: 15px;">
                    <div class="dim-badge" style="background: rgba(132,250,176,0.2); border-color: #28a745;">
                        <span class="dim-shape">✓ Output shape = Input shape: [${tokens.length} × ${embedDim}]</span>
                    </div>
                </div>
            </div>

            <div class="matrix-visual-box" data-label="Before vs After: Input → Output Comparison">
                <div style="overflow-x: auto;">
                    ${tokens.slice(0, Math.min(tokens.length, 4)).map((token, idx) => `
                        <div style="margin: 15px 0; padding: 18px; background: linear-gradient(135deg, #f8f9fa, #fff); border-radius: 10px; border: 2px solid #e9ecef;">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                                <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                                    Token ${idx + 1}
                                </span>
                                <span style="font-weight: 700; font-size: 16px; color: #333;">"${token}"</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 50px 1fr; gap: 10px; align-items: center;">
                                <div>
                                    <div style="font-size: 11px; color: #6c757d; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">Input Embedding</div>
                                    <div style="font-family: monospace; font-size: 10px; background: #f1f3f5; padding: 10px; border-radius: 6px; border-left: 3px solid #adb5bd;">
                                        [${embeddings[idx].slice(0, 5).map(v => v.toFixed(2)).join(', ')}...]
                                    </div>
                                    <div style="font-size: 10px; color: #868e96; margin-top: 4px;">Static • [${embedDim}d]</div>
                                </div>
                                <div style="text-align: center; font-size: 24px; color: #667eea;">→</div>
                                <div>
                                    <div style="font-size: 11px; color: #667eea; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Output (Contextualized)</div>
                                    <div style="font-family: monospace; font-size: 10px; background: linear-gradient(135deg, #e7f3ff, #f0e7ff); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                                        [${result.output[idx].slice(0, 5).map(v => v.toFixed(2)).join(', ')}...]
                                    </div>
                                    <div style="font-size: 10px; color: #667eea; margin-top: 4px; font-weight: 600;">Context-aware ✨ • [${embedDim}d]</div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                    ${tokens.length > 4 ? `<div style="text-align: center; color: #6c757d; padding: 10px;">... and ${tokens.length - 4} more tokens</div>` : ''}
                </div>
            </div>

            <div class="result-box" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 50%, #a8edea 100%);">
                <h4 style="margin-bottom: 15px; font-size: 20px;">🎉 Multi-Head Self-Attention Complete!</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; text-align: left; font-size: 14px;">
                    <div style="background: rgba(255,255,255,0.3); padding: 12px; border-radius: 8px;">
                        ✓ <strong>${tokens.length} tokens</strong> processed
                    </div>
                    <div style="background: rgba(255,255,255,0.3); padding: 12px; border-radius: 8px;">
                        ✓ <strong>${numHeads} attention heads</strong> combined
                    </div>
                    <div style="background: rgba(255,255,255,0.3); padding: 12px; border-radius: 8px;">
                        ✓ Output: <strong>[${tokens.length} × ${embedDim}]</strong>
                    </div>
                    <div style="background: rgba(255,255,255,0.3); padding: 12px; border-radius: 8px;">
                        ✓ <strong>Contextualized</strong> representations
                    </div>
                </div>
            </div>

            <div class="step-summary">
                <h4>📋 Complete Pipeline Summary</h4>
                <ul>
                    <li><strong>Step 1-2:</strong> Tokens → Embeddings <code>[${tokens.length}×${embedDim}]</code></li>
                    <li><strong>Step 3:</strong> Embeddings → Q, K, V <code>[${tokens.length}×${headDim}]</code> per head</li>
                    <li><strong>Step 4-5:</strong> Q×K<sup>T</sup> → Scaled Scores <code>[${tokens.length}×${tokens.length}]</code></li>
                    <li><strong>Step 6:</strong> Scores → Attention Weights (softmax)</li>
                    <li><strong>Step 7:</strong> Attention × V → Head Outputs <code>[${tokens.length}×${headDim}]</code></li>
                    <li><strong>Step 8:</strong> Concatenate ${numHeads} heads → <code>[${tokens.length}×${concatDim}]</code></li>
                    <li><strong>Step 9:</strong> Project → Final Output <code>[${tokens.length}×${embedDim}]</code></li>
                </ul>
            </div>

            <div class="whats-happening">
                <h4>🎯 The Big Picture</h4>
                <p>Each token started as an isolated embedding. After self-attention, each token's representation 
                now contains information from the <strong>entire sequence</strong>, weighted by relevance. 
                This is how transformers understand context and relationships between words!</p>
            </div>
        </div>
    `;
}

/**
 * Generate step content based on step number
 */
export function generateStepContent(step, data) {
    const { tokens, embeddings, mhsa, result, focusToken } = data;
    const head0 = result.headOutputs[0];

    switch(step) {
        case 0: return generateStep0_InputTokens(tokens);
        case 1: return generateStep1_Embeddings(tokens, embeddings, mhsa);
        case 2: return generateStep2_Projections(tokens, embeddings, mhsa, head0);
        case 3: return generateStep3_DotProduct(tokens, head0, focusToken);
        case 4: return generateStep4_Scaling(tokens, head0, mhsa, focusToken);
        case 5: return generateStep5_Softmax(tokens, head0, focusToken);
        case 6: return generateStep6_AttentionValues(tokens, head0, focusToken);
        case 7: return generateStep7_Concatenate(tokens, result, mhsa);
        case 8: return generateStep8_FinalOutput(tokens, embeddings, result);
        default: return '';
    }
}

export default {
    generateStep0_InputTokens,
    generateStep1_Embeddings,
    generateStep2_Projections,
    generateStep3_DotProduct,
    generateStep4_Scaling,
    generateStep5_Softmax,
    generateStep6_AttentionValues,
    generateStep7_Concatenate,
    generateStep8_FinalOutput,
    generateStepContent
};

