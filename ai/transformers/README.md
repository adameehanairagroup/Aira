# 🚀 AIRA Transformers & LLM Architectures (`ai/transformers`)

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![Domain: ai/transformers](https://img.shields.io/badge/Domain-Large%20Language%20Models-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)
[![Architect: Er. Adam Eehan](https://img.shields.io/badge/Architect-Er.%20Adam%20Eehan-gold.svg)](https://github.com/adameehanairagroup)

The `ai/transformers` package houses complete, runnable architectures for Large Language Models written entirely in pure `.aira`.

---

## 🏛️ Supported Transformer Architectures

1. **`GPT` (Decoder-Only Generative Model):** Causal self-attention, learned position embeddings, GELU activations.
2. **`LLaMA` (Modern Open Weights Standard):** RMSNorm, Rotary Embeddings (RoPE), SwiGLU feed-forward networks.
3. **`MoE` (Mixture of Experts):** Top-2 sparse expert gating network routing tokens across 8+ specialized expert sub-networks.
4. **`Whisper`:** Encoder-decoder transformer for speech audio spectrogram-to-text transcription.

---

## 📝 Training and Text Generation Example

```aira
import "ai/transformers";
import "ai/nlp";

// 1. Initialize Tokenizer & Model
let tokenizer = new BPETokenizer();
let gpt = new MiniGPT(vocab_size=32000, d_model=768, n_layers=12, n_heads=12);

// 2. Autoregressive Text Generation
let prompt = "The future of sovereign programming is";
let tokens = tokenizer.encode(prompt);

for step in range(0, 30) {
    let logits = gpt.forward(tokens);
    let next_token = sample_top_p(logits[-1], temperature=0.7, top_p=0.9);
    tokens.append(next_token);
}

let generated_text = tokenizer.decode(tokens);
say generated_text;
```
