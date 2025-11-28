# Multi-Head Self-Attention Visualizer

An interactive visualization tool for understanding how Multi-Head Self-Attention works in Transformer models. This tool allows you to visualize attention patterns using either deterministic embeddings or **real pre-trained transformer models** via [transformers.js](https://huggingface.co/docs/transformers.js).

## 🚀 Features

- **Step-by-step visualization** of the complete MHSA pipeline
- **Real model support** via transformers.js (BERT, DistilBERT, GPT-2, etc.)
- **Interactive attention heatmaps** showing where each token attends
- **Q/K/V projection visualization** to understand learned representations
- **Mathematical breakdown** with dimension tracking at each step
- **Architecture diagrams** showing encoder/decoder structure

## 📁 Project Structure

```
mhsa/
├── index.html          # Main HTML page
├── css/
│   └── styles.css      # All styling (1500+ lines extracted)
├── js/
│   ├── config.js       # Configuration & model definitions
│   ├── math.js         # Matrix operations & utilities
│   ├── mhsa.js         # Multi-Head Self-Attention class
│   ├── embeddings.js   # Embedding generation & transformers.js integration
│   ├── visualization.js # Canvas-based visualizations
│   ├── steps.js        # Step-by-step content generation
│   ├── ui.js           # DOM interactions & UI state
│   └── main.js         # Application entry point
├── LICENSE
└── README.md
```

## 🏃 Getting Started

### Basic Usage (No Server Required)

For basic functionality with deterministic embeddings, simply open `index.html` in a browser.

### With Real Model Support

To use real transformer models, you need to serve the files via HTTP (due to ES modules and CORS):

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve .

# Then open http://localhost:8000
```

## 🧠 Embedding Sources

The visualizer supports multiple embedding sources:

| Source | Description | Speed |
|--------|-------------|-------|
| **Deterministic** | Hash-based embeddings (default) | Instant |
| **Random** | Xavier-initialized random embeddings | Instant |
| **BERT** | Real BERT-base-uncased embeddings | ~30s first load |
| **DistilBERT** | Smaller, faster BERT variant | ~15s first load |
| **GPT-2** | GPT-2 embeddings | ~30s first load |
| **MiniLM** | Fast sentence transformer | ~10s first load |

> **Note:** Real models are downloaded once and cached in the browser. Subsequent loads are much faster.

## 🔧 Module Overview

### `config.js`
Central configuration including:
- Default parameters
- Available model definitions (HuggingFace model IDs)
- Color schemes
- Step names

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
const mhsa = new MultiHeadSelfAttention(embedDim, numHeads, temperature);
mhsa.initializeRandomWeights();
const result = mhsa.forward(embeddings);
```

### `embeddings.js`
Embedding generation and real model loading:
```javascript
// Deterministic embeddings
const embeddings = generateDeterministicEmbeddings(tokens, dim);

// Real model embeddings
const result = await getRealAttention(text, 'bert');
```

### `visualization.js`
Canvas-based drawing functions:
- `drawAttentionHeatmap()` - Attention weight grid
- `drawSoftmaxVisualization()` - Before/after softmax comparison
- `drawWeightMatrixVisualization()` - W^Q, W^K, W^V matrices

### `ui.js`
UI state and DOM manipulation:
- Tab switching
- Step navigation
- Architecture view rendering
- Loading indicators

### `main.js`
Application entry point:
- Initializes all modules
- Exposes global functions for onclick handlers
- Coordinates attention computation pipeline

## 🎨 Customization

### Adding a New Model

Edit `js/config.js`:

```javascript
models: {
    'my-model': {
        name: 'My Custom Model',
        hfId: 'username/model-name',  // HuggingFace model ID
        embedDim: 768,
        numHeads: 12,
        numLayers: 12,
        description: 'My custom model description',
        maxLength: 512
    }
}
```

### Modifying Styles

All CSS is in `css/styles.css`. Key class prefixes:
- `.dim-*` - Dimension tracker badges
- `.matrix-*` - Matrix visualizations
- `.arch-*` - Architecture diagrams
- `.step-*` - Step navigation
- `.qkv-*` - Q/K/V cards

## 🔬 How It Works

1. **Tokenization**: Input text is split into tokens
2. **Embedding**: Tokens → dense vectors (deterministic or from real model)
3. **Projection**: X → Q, K, V via learned weight matrices
4. **Attention Scores**: Q · K^T / √d_k
5. **Softmax**: Convert scores to probabilities
6. **Weighted Sum**: Attention weights × V
7. **Concatenate**: Combine all heads
8. **Output Projection**: Final linear transformation

## 📊 Real Attention Weights

When using a real model, the visualizer:
1. Loads the model and tokenizer from HuggingFace
2. Runs the input through the model
3. Extracts attention weights from a specified layer
4. Displays actual learned attention patterns

This shows how real transformers attend to different tokens, revealing patterns like:
- Syntax-based attention (subject-verb relationships)
- Semantic attention (related concepts)
- Position-based attention (nearby tokens)

## 🌐 Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 15+
- Edge 90+

Requires ES modules and WebAssembly support for real models.

## 📄 License

See [LICENSE](LICENSE) file.

