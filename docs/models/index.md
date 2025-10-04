# Models

## What is a Model?

An **AI model** is a mathematical function that has learned patterns from data.  
It takes inputs (images, text, numbers) and produces outputs (classifications, predictions, or generated text).

!!! tip "Quick analogy"
    Think of a model as a **trained system** that makes decisions based on prior experience, just like you learn to recognize patterns from examples.

**Under the hood**:

* A model file contains **millions to billions of parameters** (*weights*)  
* These parameters represent the knowledge it has learned  
* File sizes range from a few MB to hundreds of GB

**Example**:  
An image classifier trained on 1M cat/dog photos can predict *"cat"* or *"dog"* with ~95% accuracy.

---

## Model vs. Checkpoint

Before deployment, models are often **saved in different forms**:  

- **Checkpoint:** Used during training to resume or track progress  
- **Inference-ready model:** Optimized for deployment and fast predictions

This section explains the differences.


### Checkpoint (Training)
A **checkpoint** is a saved snapshot of a model during training.


**What it contains?**

| Component             | Description                                        |
| --------------------- | -------------------------------------------------- |
| **Model weights**     | Neural network parameters                          |
| **Optimizer state**   | Training momentum and learning rates               |
| **Training metadata** | Epoch number, loss values, metrics                 |
| **RNG states**        | Random number generator states for reproducibility |


**Why use checkpoints?**

* **Resume training** — continue from the last checkpoint if interrupted  
* **Compare progress** — save at different stages to analyze improvements  
* **Rollback** — revert to a previous stable checkpoint if training diverges

!!! warning "Storage consideration"
    Checkpoints are **2-3× larger** than the final model.  
    Example: A 7B parameter checkpoint ≈ 28GB vs 14GB for the model alone.

!!! example "When to use"
    During **training or fine-tuning**, when you need resume, analysis, or rollback.

### Model (Inference-Ready)
The **final model** is optimized for deployment.

**What it contains?**

| Component                   | Description                                        |
| --------------------------- | -------------------------------------------------- |
| **Model weights**           | Only the trained parameters needed for predictions |
| **Architecture definition** | Model structure and configuration                  |
| **Metadata**                | Input/output shapes, tensor names, version info    |

**Why use checkpoints?**

* **Smaller file** — faster download and load  
* **Faster loading** — no optimizer state to load  
* **Production optimized** — can be quantized, pruned, or converted for better performance

!!! example "When to use"
    When **deploying to production**: servers, APIs, mobile apps, or edge devices.

---

## Quick Comparison

| Feature              | Checkpoint                     | Inference Model        |
| -------------------- | ------------------------------ | ---------------------- |
| **Primary use**      | Training and fine-tuning       | Production deployment  |
| **File size**        | Large (2-3× model size)        | Smaller (weights only) |
| **Contains**         | Weights + optimizer + metadata | Weights + config only  |
| **Loading speed**    | Slower                         | Faster                 |
| **Resume training**  | Yes                            | No                     |
| **Production ready** | No                             | Yes                    |

!!! tip "Backend Engineer Guidelines"
    **During training:** Save checkpoints regularly.  
    **For deployment:** Export to inference format and test thoroughly.  
    **Storage planning:** Budget for checkpoint storage; deploy lean models.
