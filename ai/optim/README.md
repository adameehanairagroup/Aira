# ⚙️ AIRA Optimization Engines (`ai/optim`)

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![Domain: ai/optim](https://img.shields.io/badge/Domain-Gradient%20Optimizers-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)

The `ai/optim` package contains state-of-the-art gradient optimization algorithms used to update neural network weights during training.

---

## 🔬 `AdamW` (Decoupled Weight Decay Optimizer)

The gold standard for training Transformers and modern Deep Learning:

$$m_t = eta_1 m_{t-1} + (1 - eta_1) g_t$$
$$v_t = eta_2 v_{t-1} + (1 - eta_2) g_t^2$$
$$\hat{m}_t = rac{m_t}{1 - eta_1^t}, \quad \hat{v}_t = rac{v_t}{1 - eta_2^t}$$
$$	heta_t = 	heta_{t-1} - \eta \left( rac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda 	heta_{t-1} ight)$$

---

## 🚀 Usage in Training Loop

```aira
import "ai/optim";

// Initialize AdamW with parameters, learning rate, and weight decay
let optimizer = new AdamW(model.parameters(), lr=0.001, weight_decay=0.01);

// Step within the training epoch
optimizer.zero_grad(); // Reset accumulated gradients
loss.backward();       // Calculate gradients via autograd
optimizer.step();      // Update parameters using AdamW formula
```
