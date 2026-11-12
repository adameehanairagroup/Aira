# 🎯 AIRA Attention Mechanisms (`ai/attention`)

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![Domain: ai/attention](https://img.shields.io/badge/Domain-Multi--Head%20Attention-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)
[![Architect: Er. Adam Eehan](https://img.shields.io/badge/Architect-Er.%20Adam%20Eehan-gold.svg)](https://github.com/adameehanairagroup)

The `ai/attention` package provides native implementations of modern self-attention and cross-attention architectures.

---

## 🧮 Mathematical Formulation

### Scaled Dot-Product Attention
$$	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{Q K^T}{\sqrt{d_k}}ight) V$$

In pure `.aira`:

```aira
import "ai/attention";

// Instantiate MultiHeadAttention with 8 heads across 512 dimensions
let mha = new MultiHeadAttention(512, 8);

// Forward pass: Query, Key, Value, and Causal Attention Mask
let context_vectors = mha.forward(Q, K, V, causal_mask);
```

### Rotary Positional Embeddings (RoPE)
Enables transformers to dynamically encode relative token positions without hard positional tables:

```aira
let rope = new RotaryEmbedding(dim_per_head=64, max_seq_len=4096);
let q_rotated = rope.apply(Q, position_ids);
let k_rotated = rope.apply(K, position_ids);
```
