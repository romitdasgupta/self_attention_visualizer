/**
 * MHSA Visualizer - Visualization Module
 * Canvas-based visualizations for attention patterns and matrices
 */

import { Config } from './config.js';

/**
 * Draw an attention heatmap on a canvas
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {number[][]} attentionWeights - Attention weight matrix
 * @param {string[]} tokens - Token labels
 * @param {Object} options - Drawing options
 */
export function drawAttentionHeatmap(canvas, attentionWeights, tokens, options = {}) {
    const ctx = canvas.getContext('2d');
    const size = attentionWeights.length;
    const cellSize = Math.min(options.cellSize || 400 / size, Config.canvas.maxCellSize);
    const margin = options.margin || Config.canvas.heatmapMargin;

    canvas.width = size * cellSize + margin * 2;
    canvas.height = size * cellSize + margin * 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw heatmap cells
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const weight = attentionWeights[i][j];
            const intensity = Math.floor(weight * 255);
            
            // Purple-blue gradient based on attention weight
            ctx.fillStyle = `rgb(${255-intensity}, ${255-intensity/2}, 255)`;
            ctx.fillRect(margin + j * cellSize, margin + i * cellSize, cellSize, cellSize);

            // Draw border
            ctx.strokeStyle = '#dee2e6';
            ctx.strokeRect(margin + j * cellSize, margin + i * cellSize, cellSize, cellSize);

            // Draw value if cell is large enough
            if (cellSize > 25) {
                ctx.fillStyle = weight > 0.5 ? 'white' : 'black';
                ctx.font = `${Math.min(cellSize/3, 12)}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(
                    weight.toFixed(2),
                    margin + j * cellSize + cellSize/2,
                    margin + i * cellSize + cellSize/2
                );
            }
        }
    }

    // Draw row labels (queries)
    ctx.fillStyle = '#495057';
    ctx.font = '14px Arial';
    ctx.textAlign = 'right';
    for (let i = 0; i < size; i++) {
        const label = tokens[i].length > 8 ? tokens[i].slice(0, 7) + '…' : tokens[i];
        ctx.fillText(label, margin - 10, margin + i * cellSize + cellSize/2);
    }

    // Draw column labels (keys)
    ctx.textAlign = 'center';
    for (let j = 0; j < size; j++) {
        ctx.save();
        ctx.translate(margin + j * cellSize + cellSize/2, margin - 10);
        ctx.rotate(-Math.PI / 4);
        const label = tokens[j].length > 8 ? tokens[j].slice(0, 7) + '…' : tokens[j];
        ctx.fillText(label, 0, 0);
        ctx.restore();
    }

    // Draw axis labels
    ctx.font = 'bold 16px Arial';
    ctx.fillStyle = Config.colors.primary;
    ctx.textAlign = 'center';
    ctx.fillText('Keys (attending to)', canvas.width / 2, 30);
    
    ctx.save();
    ctx.translate(30, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Queries (attending from)', 0, 0);
    ctx.restore();
}

/**
 * Draw softmax visualization comparing raw scores to attention weights
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {string[]} tokens - Token labels
 * @param {number} focusIdx - Index of the focus token
 * @param {Object} headData - Head output data with scores and weights
 */
export function drawSoftmaxVisualization(canvas, tokens, focusIdx, headData) {
    const ctx = canvas.getContext('2d');
    canvas.width = 600;
    canvas.height = 300;

    const scores = headData.scores[focusIdx];
    const weights = headData.attentionWeights[focusIdx];

    const barWidth = Math.min(80, 500 / tokens.length);
    const gap = 10;
    const startX = (canvas.width - tokens.length * (barWidth + gap)) / 2;
    const baseY = 250;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bars for each token
    tokens.forEach((token, i) => {
        const x = startX + i * (barWidth + gap);
        
        // Raw score (red, left half)
        const scoreHeight = Math.min(Math.abs(scores[i]) * 30, 60);
        ctx.fillStyle = 'rgba(255, 107, 107, 0.6)';
        ctx.fillRect(x, baseY - scoreHeight - 80, barWidth / 2 - 5, scoreHeight);

        // Softmax weight (blue, right half)
        const weightHeight = weights[i] * 120;
        ctx.fillStyle = 'rgba(102, 126, 234, 0.8)';
        ctx.fillRect(x + barWidth / 2, baseY - weightHeight - 80, barWidth / 2 - 5, weightHeight);

        // Token label
        ctx.fillStyle = '#495057';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(token.slice(0, 8), x + barWidth / 2, baseY - 60);
        
        // Value labels
        ctx.font = '10px Arial';
        ctx.fillText(`${scores[i].toFixed(2)}`, x + barWidth / 4, baseY - scoreHeight - 85);
        ctx.fillText(`${weights[i].toFixed(2)}`, x + 3 * barWidth / 4, baseY - weightHeight - 85);
    });

    // Legend
    ctx.fillStyle = 'rgba(255, 107, 107, 0.6)';
    ctx.fillRect(20, 20, 15, 15);
    ctx.fillStyle = '#495057';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Raw Scores (Q·K)', 40, 32);

    ctx.fillStyle = 'rgba(102, 126, 234, 0.8)';
    ctx.fillRect(20, 40, 15, 15);
    ctx.fillStyle = '#495057';
    ctx.fillText('After Softmax (Attention Weights)', 40, 52);
}

/**
 * Draw weight matrix visualization
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {Object} mhsa - MHSA instance
 */
export function drawWeightMatrixVisualization(canvas, mhsa) {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const head = mhsa.heads[0];

    canvas.width = 900;
    canvas.height = 250;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const matrices = [
        { matrix: head.Wq, name: 'W^Q', color: Config.colors.query, x: 50 },
        { matrix: head.Wk, name: 'W^K', color: Config.colors.key, x: 350 },
        { matrix: head.Wv, name: 'W^V', color: Config.colors.value, x: 650 }
    ];

    matrices.forEach(({ matrix, name, color, x }) => {
        const rows = Math.min(matrix.length, 16);
        const cols = Math.min(matrix[0].length, 8);
        const cellSize = 8;

        // Draw matrix cells
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const val = matrix[i][j];
                const normalized = (val + 0.5);
                ctx.fillStyle = hexToRgba(color, normalized);
                ctx.fillRect(x + j * cellSize, 80 + i * cellSize, cellSize - 1, cellSize - 1);
            }
        }

        // Draw labels
        ctx.fillStyle = '#495057';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(name, x + cols * cellSize / 2, 60);

        ctx.font = '12px Arial';
        ctx.fillText(`${matrix.length} × ${matrix[0].length}`, x + cols * cellSize / 2, 220);

        // Draw border
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, 80, cols * cellSize, rows * cellSize);
    });

    // Description
    ctx.fillStyle = '#495057';
    ctx.font = '13px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Each matrix learns to extract different semantic aspects during training', canvas.width / 2, 245);
}

/**
 * Draw aggregation visualization showing weighted value combination
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {string[]} tokens - Token labels
 * @param {number} focusIdx - Focus token index
 * @param {Object} headData - Head output data
 */
export function drawAggregationVisualization(canvas, tokens, focusIdx, headData) {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 600;
    canvas.height = 400;

    const weights = headData.attentionWeights[focusIdx];
    const values = headData.V;
    const numDims = 8;
    const barWidth = 60;

    let startY = 50;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    tokens.forEach((token, tokenIdx) => {
        const weight = weights[tokenIdx];
        const x = 50;

        // Token label
        ctx.fillStyle = '#495057';
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(`${token.slice(0, 8)} (${weight.toFixed(2)}×)`, x - 10, startY + 15);

        // Draw value vector bars
        for (let dim = 0; dim < numDims; dim++) {
            const val = values[tokenIdx][dim];
            const weightedVal = val * weight;

            // Original value (faint)
            ctx.fillStyle = 'rgba(149, 225, 211, 0.3)';
            ctx.fillRect(x + dim * (barWidth / numDims), startY, barWidth / numDims - 2, 10);

            // Weighted value
            const hue = weightedVal > 0 ? 150 : 0;
            ctx.fillStyle = `hsla(${hue}, 70%, 50%, ${Math.abs(weight)})`;
            ctx.fillRect(x + dim * (barWidth / numDims), startY, 
                (barWidth / numDims - 2) * Math.min(Math.abs(weightedVal) * 5, 1), 10);
        }

        startY += 20;
    });

    // Title
    ctx.fillStyle = '#495057';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Each row: Value vector × Attention weight', 50, 30);

    ctx.font = '11px Arial';
    ctx.fillText('Higher attention weight = more contribution to final output', 50, startY + 20);
}

/**
 * Draw comparison visualization (before vs after attention)
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {number[][]} embeddings - Input embeddings
 * @param {number[][]} outputs - Output embeddings
 * @param {number} focusIdx - Focus token index
 * @param {string} token - Token label
 */
export function drawComparisonVisualization(canvas, embeddings, outputs, focusIdx, token) {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 600;
    canvas.height = 300;

    const numDims = Math.min(embeddings[focusIdx].length, 32);
    const barWidth = 500 / numDims;
    const startX = 50;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw input embedding
    let y = 80;
    ctx.fillStyle = '#495057';
    ctx.font = 'bold 13px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Input Embedding of "${token}"`, startX, y - 10);

    for (let i = 0; i < numDims; i++) {
        const val = embeddings[focusIdx][i];
        const height = Math.abs(val) * 50;
        ctx.fillStyle = val > 0 ? 'rgba(102, 126, 234, 0.6)' : 'rgba(234, 102, 126, 0.6)';
        ctx.fillRect(startX + i * barWidth, y + 50 - height, barWidth - 1, height);
    }

    // Draw output
    y = 180;
    ctx.fillStyle = '#495057';
    ctx.font = 'bold 13px Arial';
    ctx.fillText(`Contextualized Output of "${token}"`, startX, y - 10);

    for (let i = 0; i < numDims; i++) {
        const val = outputs[focusIdx][i];
        const height = Math.abs(val) * 50;
        ctx.fillStyle = val > 0 ? 'rgba(118, 75, 162, 0.8)' : 'rgba(162, 75, 118, 0.8)';
        ctx.fillRect(startX + i * barWidth, y + 50 - height, barWidth - 1, height);
    }

    // Arrow
    ctx.strokeStyle = Config.colors.primary;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(20, 110);
    ctx.lineTo(20, 190);
    ctx.stroke();

    // Arrowhead
    ctx.beginPath();
    ctx.moveTo(15, 180);
    ctx.lineTo(20, 190);
    ctx.lineTo(25, 180);
    ctx.fillStyle = Config.colors.primary;
    ctx.fill();

    ctx.fillStyle = Config.colors.primary;
    ctx.font = 'bold 12px Arial';
    ctx.save();
    ctx.translate(10, 150);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('ATTENTION', 0, 0);
    ctx.restore();
}

/**
 * Draw step-specific softmax canvas for sequence visualization
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {string[]} tokens - Token labels
 * @param {number} focusIdx - Focus token index
 * @param {Object} headData - Head output data
 */
export function drawStepSoftmaxCanvas(canvas, tokens, focusIdx, headData) {
    const ctx = canvas.getContext('2d');
    const scores = headData.scores[focusIdx];
    const weights = headData.attentionWeights[focusIdx];

    const barWidth = Math.min(80, 500 / tokens.length);
    const gap = 10;
    const startX = (canvas.width - tokens.length * (barWidth + gap)) / 2;
    const baseY = 250;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    tokens.forEach((token, i) => {
        const x = startX + i * (barWidth + gap);
        
        // Scaled score (before softmax)
        const scoreHeight = Math.min(Math.abs(scores[i]) * 40, 80);
        ctx.fillStyle = 'rgba(255, 107, 107, 0.6)';
        ctx.fillRect(x, baseY - scoreHeight - 80, barWidth / 2 - 2, scoreHeight);

        // Softmax weight
        const weightHeight = weights[i] * 120;
        ctx.fillStyle = 'rgba(102, 126, 234, 0.9)';
        ctx.fillRect(x + barWidth / 2, baseY - weightHeight - 80, barWidth / 2 - 2, weightHeight);

        // Labels
        ctx.fillStyle = '#495057';
        ctx.font = '11px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(token.slice(0, 8), x + barWidth / 2, baseY - 60);
        
        ctx.font = '9px Arial';
        ctx.fillText(scores[i].toFixed(2), x + barWidth / 4, baseY - scoreHeight - 85);
        ctx.fillText((weights[i] * 100).toFixed(0) + '%', x + 3 * barWidth / 4, baseY - weightHeight - 85);
    });

    // Legend
    ctx.fillStyle = 'rgba(255, 107, 107, 0.6)';
    ctx.fillRect(20, 20, 15, 15);
    ctx.fillStyle = '#495057';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Scaled Scores', 40, 32);

    ctx.fillStyle = 'rgba(102, 126, 234, 0.9)';
    ctx.fillRect(20, 40, 15, 15);
    ctx.fillText('Attention Weights (%)', 40, 52);
}

/**
 * Helper: Convert hex color to rgba
 * @param {string} hex - Hex color code
 * @param {number} alpha - Alpha value
 * @returns {string} RGBA color string
 */
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Format a matrix for HTML display
 * @param {number[][]} matrix - Matrix to format
 * @param {string} label - Matrix label
 * @returns {string} HTML string
 */
export function formatMatrix(matrix, label) {
    let html = `<h4>${label}</h4><div class="matrix-display">`;
    html += matrix.map((row, i) =>
        `<div class="matrix-row">[${row.map(v =>
            `<span class="matrix-value">${v.toFixed(4)}</span>`
        ).join(' ')}]</div>`
    ).join('');
    html += '</div>';
    return html;
}

export default {
    drawAttentionHeatmap,
    drawSoftmaxVisualization,
    drawWeightMatrixVisualization,
    drawAggregationVisualization,
    drawComparisonVisualization,
    drawStepSoftmaxCanvas,
    formatMatrix
};

