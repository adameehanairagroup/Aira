# 🧠 AIRA AI Ecosystem — Building & Training AI from Scratch

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![AI Modules: 1,120](https://img.shields.io/badge/AI%20Modules-1%2C120%20Pure%20.aira-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)
[![Zero--Key AI: Active](https://img.shields.io/badge/Zero--Key%20Swarm-Embedded%20Native-brightgreen.svg)](https://github.com/adameehanairagroup/Airalang)
[![Architect: Er. Adam Eehan](https://img.shields.io/badge/Architect-Er.%20Adam%20Eehan-gold.svg)](https://github.com/adameehanairagroup)

> **"Intelligence is not a library in AIRA; it is a fundamental language primitive."**  
> Architect: **Er. Adam Eehan (Founder & CEO, Aira Group of Technology)**

---

## 🌟 The AIRA AI Philosophy

In legacy languages like Python or C++, building an AI model requires installing gigabytes of third-party C++ bindings (PyTorch, TensorFlow, NumPy, CUDA runtime) and configuring fragile virtual environments.

In **AIRA**, Artificial Intelligence is built **natively from scratch into the language syntax and standard library**:
1. **Pure Single-Language Implementation:** All **1,120 AI modules** in this directory are written in **100% Pure `.aira` code**.
2. **First-Class Tensor Math & Autograd:** Tensor manipulation, automatic backward differentiation, and graph execution are native language features.
3. **Build & Train Any Model:** From multi-layer perceptrons (MLP) and Convolutional Networks (CNN) to Generative Pre-Trained Transformers (GPT), Diffusion Models (DDPM), and State Space Models (Mamba)—every single layer is written and trainable in `.aira`.
4. **Zero-Key Swarm Inference:** When developers need instant intelligence without training, `ai.ask()` provides immediate zero-configuration access to sovereign neural swarms.

---

## 📖 The Complete Guide: Training a Neural Network from Scratch in AIRA

Here is how a developer writes, configures, and trains an AI model from scratch in pure AIRA syntax.

### 1. Defining Neural Network Layers

```aira
// Import core neural network primitives
import "ai/nn";
import "ai/loss";
import "ai/optim";

// Define a Multi-Layer Neural Network Class
class SovereignClassifier {
    fn init(in_dim, hidden_dim, out_dim) {
        this.fc1 = new Linear(in_dim, hidden_dim);
        this.relu = new ReLU();
        this.fc2 = new Linear(hidden_dim, out_dim);
    }

    // Forward propagation through layers
    fn forward(x) {
        let h1 = this.fc1.forward(x);
        let a1 = this.relu.forward(h1);
        let out = this.fc2.forward(a1);
        return out;
    }

    // Collect all learnable parameters for optimizer
    fn parameters() {
        return this.fc1.parameters() + this.fc2.parameters();
    }
}
```

### 2. The Complete Training Loop in Pure AIRA

```aira
// 1. Initialize dataset (features and labels)
let dataset_x = [
    [0.1, 0.2, 0.3, 0.4],
    [0.8, 0.7, 0.9, 0.6],
    [0.2, 0.1, 0.3, 0.2],
    [0.9, 0.8, 0.8, 0.7]
];
let dataset_y = [0, 1, 0, 1]; // Binary classes

// 2. Instantiate Model, Loss Function, and Optimizer
let model = new SovereignClassifier(4, 8, 2);
let criterion = new CrossEntropyLoss();
let optimizer = new AdamW(model.parameters(), 0.005, 0.01); // lr=0.005, weight_decay=0.01

say "🚀 Starting AIRA Sovereign Model Training...";

// 3. Training Loop across 50 Epochs
for epoch in range(1, 51) {
    let total_loss = 0.0;

    for i in range(0, len(dataset_x)) {
        let x = dataset_x[i];
        let target = dataset_y[i];

        // Step A: Forward pass
        let logits = model.forward(x);

        // Step B: Compute Loss
        let loss = criterion.forward(logits, target);
        total_loss = total_loss + loss.value;

        // Step C: Zero out previous gradients
        optimizer.zero_grad();

        // Step D: Backpropagation (Compute Gradients via Autograd)
        loss.backward();

        // Step E: Update Model Weights
        optimizer.step();
    }

    if (epoch % 10 == 0) {
        say "Epoch [" + str(epoch) + "/50] - Loss: " + str(total_loss / len(dataset_x));
    }
}

say "✅ Model Training Successfully Completed in Pure AIRA!";
```

---

## ⚡ Building a Generative Pre-Trained Transformer (GPT) in AIRA

AIRA provides complete Transformer blocks including Scaled Dot-Product Attention, Multi-Head Attention, and Layer Normalization:

```aira
import "ai/transformers";
import "ai/attention";

// Building a Transformer Decoder Block
class TransformerDecoderBlock {
    fn init(d_model, n_heads) {
        this.attn = new MultiHeadAttention(d_model, n_heads);
        this.ln1 = new LayerNorm(d_model);
        this.ffn1 = new Linear(d_model, d_model * 4);
        this.gelu = new GELU();
        this.ffn2 = new Linear(d_model * 4, d_model);
        this.ln2 = new LayerNorm(d_model);
    }

    fn forward(x, mask) {
        // Residual connection 1: Self-Attention
        let norm_x = this.ln1.forward(x);
        let attn_out = this.attn.forward(norm_x, norm_x, norm_x, mask);
        let h = x + attn_out;

        // Residual connection 2: Feed-Forward Network
        let norm_h = this.ln2.forward(h);
        let ffn_out = this.ffn2.forward(this.gelu.forward(this.ffn1.forward(norm_h)));
        return h + ffn_out;
    }
}
```

---

## 🧠 Instant Zero-Key AI Swarm (`ai.ask`)

When you do not need to train a model from scratch and want immediate reasoning or code generation:

```aira
import "ai";

// 1. Direct Zero-Key AI Question
let answer = ai.ask("Explain quantum teleportation in 2 sentences");
say answer;

// 2. Interactive Terminal AI Co-Pilot
while true {
    let prompt = input("Developer > ");
    if (prompt.lower() == "exit") {
        say "Terminating AI session.";
        break;
    }
    let reply = ai.ask(prompt);
    say "Aira > " + reply;
}
```

---

## 📂 AI Submodule Architecture Index (1,120 Modules)

Explore the dedicated documentation inside each package for complete API references and mathematical formulas:

| Package | Purpose & Architecture | Documentation |
| :--- | :--- | :--- |
| **`ai/nn/`** | Core Neural Layers: Linear, Conv2D, Dropout, LayerNorm, Activations | [👉 Read `ai/nn` Guide](./nn/README.md) |
| **`ai/attention/`** | Scaled Dot-Product Attention, Multi-Head Attention, Rotary Embeddings (RoPE) | [👉 Read `ai/attention` Guide](./attention/README.md) |
| **`ai/transformers/`** | Complete GPT, LLaMA, MoE (Mixture of Experts), Whisper Architectures | [👉 Read `ai/transformers` Guide](./transformers/README.md) |
| **`ai/loss/`** | Loss Functions: CrossEntropyLoss, MSELoss, BCE, HuberLoss, TripletLoss | [👉 Read `ai/loss` Guide](./loss/README.md) |
| **`ai/optim/`** | Optimization Algorithms: AdamW, SGD with Momentum, RMSprop, Schedulers | [👉 Read `ai/optim` Guide](./optim/README.md) |
| **`ai/diffusion/`** | Denoising Diffusion Probabilistic Models (DDPM), Noise Schedulers | [👉 Read `ai/diffusion` Guide](./diffusion/README.md) |
| **`ai/nlp/`** | Tokenization: Byte-Pair Encoding (BPE), WordPiece, Embedding Similarity | [👉 Read `ai/nlp` Guide](./nlp/README.md) |
| **`ai/mamba/`** | State Space Models (SSM), Selective Scan, O(N) Linear Attention | [👉 Read `ai/mamba` Guide](./mamba/README.md) |
| **`ai/rlhf/`** | RLHF, Direct Preference Optimization (DPO), PPO, Reward Modeling | [👉 Read `ai/rlhf` Guide](./rlhf/README.md) |
| **`ai/quantization/`** | GGUF 4-Bit / 8-Bit Quantization, Edge AI inference on Mobile / Termux | [👉 Read `ai/quantization` Guide](./quantization/README.md) |
| **`ai/agent/`** | Autonomous Swarm Agents, Guardrails, Tool-calling loops, Task Planning | [👉 Read `ai/agent` Guide](./agent/README.md) |
| **`ai/providers/`** | Native Zero-Key Swarm, OpenAI, Claude, Gemini, Groq, DeepSeek Bridges | [👉 Read `ai/providers` Guide](./providers/README.md) |
| **`ai/cv/`** | Computer Vision: Image Filters, Feature Detectors, Convolutional Kernels | [👉 Read `ai/cv` Guide](./cv/README.md) |
