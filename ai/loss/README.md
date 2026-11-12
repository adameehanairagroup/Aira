# 📉 AIRA Loss Functions (`ai/loss`)

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![Domain: ai/loss](https://img.shields.io/badge/Domain-Objective%20Functions-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)

Loss functions calculate the scalar discrepancy between model predictions and ground-truth targets, guiding gradient backpropagation.

---

## 📐 Implemented Loss Functions

### 1. `CrossEntropyLoss` (Classification & Next-Token LLM Training)
$$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$
Utilizes numerically stable Log-Softmax with max-subtraction to prevent floating-point underflow:

```aira
import "ai/loss";

let criterion = new CrossEntropyLoss();
let loss = criterion.forward(logits, target_class);
loss.backward(); // Propagates dL/dLogits back into model layers
```

### 2. `MSELoss` (Mean Squared Error)
$$\mathcal{L} = rac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
Ideal for continuous regression, coordinate prediction, and Diffusion noise estimation.

### 3. `BCEWithLogitsLoss`
Binary cross-entropy combining a sigmoid activation and cross-entropy for numerical stability in multi-label classification.
