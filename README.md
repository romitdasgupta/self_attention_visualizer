# Multi-Head Self-Attention Visualizer

An interactive visualization tool for understanding how Multi-Head Self-Attention works in Transformer models. Explore attention patterns using deterministic embeddings or **real pre-trained DistilBERT weights** with transformer tokenization via [transformers.js](https://huggingface.co/docs/transformers.js).

## 🚀 Features

- **Step-by-step visualization** of the complete 9-step MHSA pipeline
- **Real model weights** - Pre-extracted DistilBERT attention weights from trained models
- **Interactive attention heatmaps** showing where each token attends
- **Q/K/V projection visualization** to understand learned representations
- **Mathematical breakdown** with dimension tracking at each step
- **Architecture diagrams** showing encoder/decoder structure
- **Multiple attention heads** - Compare how different heads learn different patterns
- **Temperature control** - Adjust softmax sharpness to see how it affects attention

## 📁 Project Structure

```
mhsa/
├── index.html              # Main HTML page
├── css/
│   └── styles.css          # All styling
├── js/
│   ├── config.js           # Configuration & model definitions
│   ├── math.js             # Matrix operations & utilities
│   ├── mhsa.js             # Multi-Head Self-Attention implementation
│   ├── embeddings.js       # Embedding generation & transformers.js integration
│   ├── model-weights.js    # Pre-extracted weight loading & caching
│   ├── visualization.js    # Canvas-based visualizations
│   ├── steps.js            # Step-by-step content generation
│   ├── ui.js               # DOM interactions & UI state
│   └── main.js             # Application entry point
├── models/
│   ├── distilbert-4head.json   # DistilBERT layer 0, 4 heads (768-dim, 64 head-dim)
│   ├── distilbert-2head.json   # DistilBERT layer 0, 2 heads
│   ├── distilbert-1head.json   # DistilBERT layer 0, 1 head
│   └── demo-tiny-weights.json  # Small demo model (64-dim, 4 heads)
├── scripts/
│   ├── extract_model_weights.py   # Extract weights from HuggingFace models
│   ├── create_compact_weights.py  # Create compact weight files for web
│   └── requirements.txt           # Python dependencies
├── LICENSE
└── README.md
```

## 🏃 Getting Started

### Quick Start (Local Server Required)

The visualizer uses ES modules and requires serving via HTTP:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve .

# Then open http://localhost:8000
```

### With Real Model Embeddings

For real-time DistilBERT embeddings via transformers.js, the model will be downloaded on first use (~50MB, cached in browser).

## 🧠 Model Sources

The visualizer supports multiple model configurations:

| Source | Description | Speed |
|--------|-------------|-------|
| **Deterministic** | Hash-based embeddings with random weights | Instant |
| **Random** | Xavier-initialized random embeddings & weights | Instant |
| **DistilBERT (4 heads)** | Real DistilBERT weights + transformers.js tokenization | ~5s first load |
| **DistilBERT (2 heads)** | Subset of DistilBERT heads for clearer visualization | ~5s first load |
| **Demo (Tiny)** | Small 64-dim model for fast experiments | Instant |

> **Note:** Real model weights are pre-extracted JSON files (~2-5MB). The transformers.js model for tokenization is cached in the browser after first load.

## 🔧 Module Overview

### `config.js`
Central configuration including:
- Default parameters (embedding dimension, heads, temperature)
- Model definitions with weight file paths
- Color schemes for visualization
- Step names for the 9-step sequence

### `math.js`
Matrix operations:
- `matmul()` - Matrix multiplication
- `transpose()` - Matrix transpose
- `softmax()` - Row-wise softmax with temperature
- `randomMatrix()` - Xavier initialization
- `averageAttentions()` - Average multiple attention matrices

### `mhsa.js`
Multi-Head Self-Attention implementation:
```javascript
// With random weights
const mhsa = createRandomMHSA(embedDim, numHeads, temperature);

// With pre-extracted weights
const mhsa = createMHSAFromWeights(loadedWeights, temperature);

const result = mhsa.forward(embeddings);
```

Supports:
- Per-head Q, K, V, O projection weights
- Optional biases (used by real models)
- Temperature-controlled softmax

### `embeddings.js`
Embedding generation and real model loading:
```javascript
// Deterministic/random embeddings
const { embeddings, tokens } = await getEmbeddings(tokens, dim, source);

// Real DistilBERT embeddings via transformers.js
const { embeddings, tokens, hiddenSize } = await getDistilBertEmbeddings(text);
```

### `model-weights.js`
Pre-extracted weight management:
```javascript
// Load weights with progress callback
const weights = await loadModelWeights('distilbert-4head', (progress) => {
    console.log(`Loading: ${progress.progress}%`);
});

// Weights are cached in memory
const cached = isCached('distilbert-4head');
```

### `visualization.js`
Canvas-based drawing functions:
- `drawAttentionHeatmap()` - Attention weight grid with color intensity
- `drawSoftmaxVisualization()` - Before/after softmax comparison
- `drawWeightMatrixVisualization()` - W^Q, W^K, W^V matrix display
- `drawAggregationVisualization()` - Value aggregation with weights
- `drawComparisonVisualization()` - Input vs output embeddings

### `ui.js`
UI state and DOM manipulation:
- Tab switching (Overview, Step-by-Step, Heads, Mathematics)
- Step navigation with progress bar
- Architecture view rendering (Encoder/Decoder/Full)
- Dynamic dimension displays

### `main.js`
Application entry point:
- Initializes all modules
- Coordinates the attention computation pipeline
- Manages embedding source switching
- Exposes global functions for UI handlers

## 🎨 Visualization Tabs

### Overview
- Combined attention heatmap (averaged across all heads)
- Quick view of overall attention patterns

### Step-by-Step Sequence
Interactive 9-step walkthrough:
1. Input Tokens
2. Token Embeddings
3. Linear Projections (Q, K, V)
4. Query-Key Dot Product
5. Scaling (÷√d_k)
6. Softmax
7. Attention × Values
8. Concatenate Heads
9. Final Output

### Individual Heads
- Side-by-side attention heatmaps for each head
- See how different heads focus on different relationships

### Mathematics
- Detailed formulas with actual computed values
- Matrix dimensions at each step
- First-head deep dive with Q, K, V matrices

## 🔬 How It Works

1. **Input** - Text is split into tokens (or subword tokens with DistilBERT)
2. **Embedding** - Tokens → dense vectors (768-dim for DistilBERT)
3. **Projection** - X → Q, K, V via learned weight matrices per head
4. **Attention Scores** - Q · K^T (query-key similarity)
5. **Scaling** - Divide by √d_k to stabilize gradients
6. **Softmax** - Convert scores to attention probabilities
7. **Weighted Sum** - Attention weights × V (gather relevant information)
8. **Concatenate** - Combine outputs from all heads
9. **Output Projection** - Final linear transformation

## 🛠 Extracting Custom Model Weights

Use the provided Python scripts to extract weights from any HuggingFace model:

```bash
cd scripts
pip install -r requirements.txt

# Extract DistilBERT weights
python extract_model_weights.py

# Create compact per-head weight files
python create_compact_weights.py
```

### Adding a New Model

1. Extract weights using the Python scripts
2. Add configuration to `js/config.js`:

```javascript
models: {
    'my-model': {
        name: 'My Model Name',
        description: 'Description here',
        weightsFile: 'models/my-model-weights.json',
        embedDim: 768,
        numHeads: 4,
        headDim: 64,
        isRealModel: true
    }
}
```

## 📊 Understanding the Visualizations

### Attention Heatmaps
- **Rows** = Query tokens (what's asking)
- **Columns** = Key tokens (what's being attended to)
- **Intensity** = Attention weight (0 to 1)
- Each row sums to 1 after softmax

### Real Model Patterns
When using DistilBERT weights, you'll observe learned patterns:
- **Diagonal attention** - Tokens attending to themselves
- **Adjacent attention** - Local context (nearby words)
- **Long-range attention** - Distant but related tokens
- **Special token patterns** - [CLS] often aggregates globally

## 🌐 Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 15+
- Edge 90+

Requires ES modules and WebAssembly support for transformers.js.

## 📄 License

See [LICENSE](LICENSE) file.
