<div align="center">

# Fine-Tuning SmolLM-1.7B-Instruct with LoRA for Domain Adaptation

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://github.com/huggingface/peft)

</div>
    
This repository provides a comprehensive guide and implementation for fine-tuning a pre-trained language model, **`SmolLM2-1.7B-Instruct`**, using Parameter-Efficient Fine-Tuning (PEFT), specifically Low-Rank Adaptation (LoRA). The project demonstrates how to adapt a general-purpose model to a specialized domain—in this case, an event-ticketing assistant—making it more accurate, context-aware, and efficient.

The primary focus is the **methodology of fine-tuning**, covering everything from data preparation and cleaning to the theoretical underpinnings of LoRA and its practical application.

<br>

<div align="center">
    
<img src="https://www.mygreatlearning.com/blog/wp-content/uploads/2025/04/fine-tuning-banner.jpg" width="700"/>

</div>

---

## 📜 Table of Contents

1.  [**Introduction to Fine-Tuning**](#1-introduction-to-fine-tuning)
    -   [What is Fine-Tuning?](#what-is-fine-tuning)
    -   [Why Fine-Tune a Language Model?](#why-fine-tune-a-language-model)
2.  [**Core Concepts: PEFT and LoRA**](#2-core-concepts-peft-and-lora)
    -   [What is PEFT?](#what-is-peft)
    -   [Deep Dive: LoRA (Low-Rank Adaptation)](#deep-dive-lora-low-rank-adaptation)
3.  [**The Fine-Tuning Workflow**](#3-the-fine-tuning-workflow)
    -   [Step 1: Environment Setup](#step-1-environment-setup)
    -   [Step 2: Data Preparation & Cleaning](#step-2-data-preparation--cleaning)
    -   [Step 3: Model and Tokenizer Loading](#step-3-model-and-tokenizer-loading)
    -   [Step 4: Configuring LoRA and the Trainer](#step-4-configuring-lora-and-the-trainer)
    -   [Step 5: Training and Saving the Model](#step-5-training-and-saving-the-model)
4.  [**How to Run This Project**](#4-how-to-run-this-project)
5.  [**Results: In-Domain vs. Out-of-Domain Performance**](#5-results-in-domain-vs-out-of-domain-performance)
6.  [**License**](#6-license)

<br>

---

## 1. Introduction to Fine-Tuning

### What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained model—one that has already been trained on a vast amount of general data—and training it further on a smaller, task-specific dataset. This adapts the model's general knowledge to excel at a particular task, such as understanding medical terminology, writing in a specific legal style, or, in this case, handling event ticketing queries.

### Why Fine-Tune a Language Model?

Pre-trained models like GPT or Llama are powerful but have limitations:
-   **Lack of Domain-Specific Knowledge:** They don't know your company's internal jargon, product names, or specific policies.
-   **Generic Tone and Style:** Their responses may not align with your brand's voice.
-   **Potential for Hallucinations:** When faced with niche queries, they might generate plausible but incorrect information.

Fine-tuning bridges this gap, leading to:
-   ✅ **Higher Accuracy:** More relevant and factually correct responses for your specific domain.
-   ✅ **Improved Performance:** Faster, more personalized answers tailored to user needs.
-   ✅ **Reduced Off-Topic Replies:** The model learns to stay within the bounds of its intended function.

<br>

---

## 2. Core Concepts: PEFT and LoRA

Training all the parameters of a massive LLM (full fine-tuning) is computationally expensive and requires significant hardware. **Parameter-Efficient Fine-Tuning (PEFT)** methods offer a solution by only updating a small subset of the model's parameters.

### What is PEFT?

PEFT is a collection of techniques that dramatically reduce the computational and storage costs of fine-tuning. Instead of modifying billions of parameters, PEFT methods freeze the original weights and insert small, trainable modules or "adapters" into the model.

Examples include: **LoRA**, Adapters, Prefix Tuning, and BitFit.

### Deep Dive: LoRA (Low-Rank Adaptation)

This project uses **LoRA**, one of the most popular PEFT techniques. LoRA works on the hypothesis that the change in weights during fine-tuning has a "low intrinsic rank." Therefore, instead of learning a large weight update matrix `ΔW`, LoRA learns two smaller, low-rank matrices `A` and `B` whose product approximates `ΔW`.

<div align="center">
  <img src="https://www.dailydoseofds.com/content/images/size/w1000/2024/02/image-283.png" width="600"/>
</div>

**How it Works Mathematically:**

-   A pre-trained weight matrix `W` is frozen.
-   The update `ΔW` is represented by a low-rank decomposition: `ΔW = A * B`, where `A` is a `d x r` matrix and `B` is an `r x k` matrix. The rank `r` is much smaller than `d` or `k`.
-   During training, only `A` and `B` are updated.
-   The forward pass is modified: `W'x = Wx + α * (AB)x`, where `α` is a scaling factor.

**Benefits of LoRA:**
-   **Massive Parameter Reduction:** We train `r * (d + k)` parameters instead of `d * k`, drastically reducing VRAM requirements.
-   **Faster Training:** Fewer parameters mean quicker convergence.
-   **Easy Deployment:** The original model remains unchanged. The small `A` and `B` matrices can be saved separately and loaded on top of the base model as needed, allowing for multiple task-specific adapters for a single base model.

<br>

---

## 3. The Fine-Tuning Workflow

This project follows a standard, reproducible workflow for fine-tuning an LLM.

### Step 1: Environment Setup

First, install the necessary libraries from the Hugging Face ecosystem and other data handling tools.

```bash
pip install -q transformers datasets peft trl wandb torch
```

### Step 2: Data Preparation & Cleaning

A high-quality dataset is the key to successful fine-tuning. The data should be in an instruction-response format.

1.  **Load Data:** Start with a structured dataset (e.g., CSV) containing columns for `instruction`, `intent`, and `response`.
2.  **Clean Data:**
    -   Remove duplicate samples.
    -   Sanitize text by removing profanity or irrelevant artifacts.
    -   Standardize formatting (e.g., capitalization, placeholder consistency).
3.  **Augment Data:** To make the model more robust, we concatenate our domain-specific data with a set of **out-of-domain** queries. For these, the desired response is a polite refusal, teaching the model to stay on task.
4.  **Format for Training:** Convert the data into the chat template required by the model.

```python
# Example of formatting a row into a chat template
def format_chat(row):
    messages = [
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

df["text"] = df.apply(format_chat, axis=1)
```

### Step 3: Model and Tokenizer Loading

Load the pre-trained model and its corresponding tokenizer. We use `SmolLM2-1.7B-Instruct`, a compact but powerful model suitable for fine-tuning on consumer-grade hardware.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
```

### Step 4: Configuring LoRA and the Trainer

This is where we define the LoRA parameters and set up the training process using `SFTTrainer` from the TRL library.

```python
from peft import LoraConfig
from trl import SFTTrainer

# Configure LoRA
peft_config = LoraConfig(
    r=32,                     # LoRA rank (low-rank dimension)
    lora_alpha=64,            # Scaling factor for LoRA weights
    lora_dropout=0.01,        # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear" # Apply LoRA to all linear layers
)

# Configure Training Arguments
training_arguments = TrainingArguments(...)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
)
```

### Step 5: Training and Saving the Model

Start the training process and save the resulting LoRA adapters.

```python
# Start training
trainer.train()

# Save the LoRA adapters and tokenizer
output_path = "./SmolLM2-finetuned-adapters"
trainer.model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```
The saved `output_path` will contain only the trained adapter weights (a few megabytes), not the entire 1.7B parameter model.

<br>

---

## 4. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MarpakaPradeepSai/Fine-tuning-HuggingFaceTB-SmolLM2-1.7B-Instruct-with-LoRA.git
    cd Fine-tuning-HuggingFaceTB-SmolLM2-1.7B-Instruct-with-LoRA
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Weights & Biases (Optional):**
    For logging metrics and training progress, log in to W&B.
    ```python
    import wandb
    wandb.login(key="YOUR_WANDB_API_KEY")
    ```

4.  **Run the Notebook:**
    Open the `Fine-Tuning-SmolLM.ipynb` notebook in a Jupyter or Colab environment and execute the cells sequentially.

<br>

---

## 5. Results: In-Domain vs. Out-of-Domain Performance

After fine-tuning, the model is significantly better at handling domain-specific queries while politely deflecting out-of-domain requests.

#### **✅ In-Domain Query**

> **User:** "How can I cancel my ticket for the upcoming cricket event in India?"

> **Fine-Tuned Assistant:** "To cancel your ticket for the upcoming cricket event in India, please adhere to the following steps:
> 1. Access [website] and sign in to your account.
> 2. Proceed to the <b>Ticket Cancellation</b> section.
> 3. Locate your scheduled events and choose the cricket event in India.
> 4. Select the <b>Cancel Ticket</b> option.
> 5. Complete the process by following the on-screen instructions for cancellation."

#### **❌ Out-of-Domain Query**

> **User:** "Explain the working of Large Language Models."

> **Fine-Tuned Assistant:** "I'm sorry, but I can't offer assistance on this topic. If you have questions about event ticketing, please let me know."

This demonstrates the model has successfully specialized its function, a direct result of the fine-tuning process.

<br>

---

## 6. License

This project is licensed under the [MIT License](LICENSE).

Note: This project uses the SmolLM2-1.7B-Instruct model, which is licensed under the [Apache 2.0 License](LICENSE). Any use of the model and its derivatives must comply with its original license.
