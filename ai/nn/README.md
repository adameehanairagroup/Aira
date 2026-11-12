# 🧱 AIRA Neural Network Primitives (`ai/nn`)

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![Domain: ai/nn](https://img.shields.io/badge/Domain-Neural%20Layers-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)
[![Architect: Er. Adam Eehan](https://img.shields.io/badge/Architect-Er.%20Adam%20Eehan-gold.svg)](https://github.com/adameehanairagroup)

The `ai/nn` package provides the fundamental building blocks for constructing deep neural networks in pure `.aira`.

---

## 📐 Included Layers & Mechanics

### 1. `Linear(in_features, out_features, bias=true)`
Computes the linear affine transformation:
$$\mathbf{y} = \mathbf{x} \mathbf{W}^T + \mathbf{b}$$

```aira
import "ai/nn";

// Create a Linear layer with 784 inputs (e.g. MNIST) and 128 hidden neurons
let layer = new Linear(784, 128);
let output = layer.forward(input_vector);
```

### 2. `LayerNorm(normalized_shape, eps=1e-5)`
Normalizes the inputs across the features dimension for training stability in Transformers:
$$y = rac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + eta$$

```aira
let norm = new LayerNorm(512);
let normalized_tensor = norm.forward(hidden_states);
```

### 3. Activation Functions
* **`ReLU`:** $f(x) = \max(0, x)$ — Standard non-linear activation.
* **`GELU`:** $f(x) = x \cdot \Phi(x) pprox 0.5x(1 + 	anh(\sqrt{2/\pi}(x + 0.044715x^3)))$ — Used in GPT and modern LLMs.
* **`SiLU` (Swish):** $f(x) = x \cdot \sigma(x)$ — Used in LLaMA feed-forward networks.
* **`Softmax`:** Transforms arbitrary logits into valid probability distributions summing to 1.0.

```aira
let gelu = new GELU();
let activated = gelu.forward(raw_logits);
```
