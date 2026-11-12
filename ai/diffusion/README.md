# 🎨 AIRA Generative Diffusion Models (`ai/diffusion`)

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![Domain: ai/diffusion](https://img.shields.io/badge/Domain-DDPM%20%26%20Generative%20AI-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)

Provides native implementations of Denoising Diffusion Probabilistic Models (DDPM) and noise schedules for generative image, audio, and tensor synthesis.

---

## 🌊 Diffusion Mechanics in AIRA

* **Forward Noise Schedule:** Adds Gaussian noise over $T$ timesteps according to variance schedule $eta_1, \dots, eta_T$:
  $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{ar{lpha}_t} x_0, (1 - ar{lpha}_t)\mathbf{I})$$
* **Reverse Denoising Process:** The neural network predicts the noise $\epsilon_	heta(x_t, t)$ to iteratively reconstruct clean latent data $x_0$.

```aira
import "ai/diffusion";

let scheduler = new DDPMScheduler(timesteps=1000, beta_start=0.0001, beta_end=0.02);
let noisy_sample = scheduler.add_noise(clean_latent, noise, timestep=500);
```
