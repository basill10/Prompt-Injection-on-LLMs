# Prompt Injection on Large Language Models (LLMs)

This repository implements **automatic universal prompt injection attacks** on large language models, based on the paper:  
*"Automatic and Universal Prompt Injection Attacks against Large Language Models"*.  

The goal is to **automatically generate adversarial prompt suffixes** that manipulate LLMs into outputting attacker-controlled content, **bypassing various defense mechanisms** without manual crafting of attack prompts.

---

## Overview

Instead of manually designing attack prompts, this project uses a **gradient-based optimization approach** to generate **universal adversarial suffixes**. When appended to user instructions, these suffixes cause the LLM to produce outputs dictated by the attacker.

The process consists of three main phases:

### 1. Attack Generation (`universal_prompt_injection.py`)

- Initializes the attack environment, loading a target LLM (e.g., LLaMA 2) and its tokenizer.
- Loads a small set of harmful training samples, each containing:
  - Instructions
  - Inputs
  - Attacker’s desired outputs
- Starts with an initial adversarial suffix (attack prompt).
- Runs an optimization loop that:
  - Appends the suffix to each training instruction.
  - Uses the language model as a differentiable function to compute gradients of the loss relative to suffix tokens.
  - Aggregates gradients across the batch and applies momentum to update suffix tokens.
- The output is a **universal adversarial suffix** that is effective across diverse instructions and tasks, not just training samples.
- Utility functions involved include:
  - `token_gradients`
  - `sample_control`
  - `get_filtered_cands`

### 2. Attack Evaluation (`get_responses_universal.py`)

- Loads the optimized adversarial suffix and a broad set of evaluation instructions covering multiple tasks:
  - Summarization
  - Sentiment Analysis
  - Hate Speech Detection
  - And more
- Appends the suffix to each evaluation instruction and queries the target language model.
- Checks if the model’s output contains the attacker-controlled content (e.g., a specific phrase or URL).
- Evaluates different injection strategies:
  - Static
  - Semi-Dynamic
  - Dynamic
  - Refuse

### 3. Results Checking (`check_answers.py`)

- Loads evaluation results from the previous phase.
- Checks whether the attack target appears in each model output.
- Computes success rates per task and model configuration.
- Saves detailed success metrics for analysis.

---

## How to Use

1. **Generate Attack Suffix:**  
   Run `universal_prompt_injection.py` to produce the universal adversarial suffix.

2. **Evaluate Attack:**  
   Use `get_responses_universal.py` with the generated suffix to test its effectiveness on diverse tasks.

3. **Analyze Results:**  
   Execute `check_answers.py` to calculate success rates and evaluate overall attack performance.

---

## Requirements

- Python 3.x  
- PyTorch  
- Hugging Face Transformers  
- Other dependencies (see `requirements.txt`)  

---

## References

- Paper: [Automatic and Universal Prompt Injection Attacks against Large Language Models](https://arxiv.org/pdf/2403.04957)  
- Models: Google Flan T5, LLaMA 2, and others  

---

Feel free to contribute, report issues, or suggest improvements!
