# 4mulaMind-LLM

A lightweight, custom Transformer-based language model developed by **4mulaMind** for intent understanding and safe command generation for **4mulaQuery**.

## Overview

4mulaMind-LLM is a custom-trained language model designed to understand natural-language requests and map them to a small, controlled set of database operations.

Supported intents:

* `SEARCH`
* `INSERT`
* `DELETE`
* `ALL`
* `INVALID`

The model is integrated with a **Safe Pipeline** that validates the generated intent and arguments before producing a command for the 4mulaQuery engine.

## Architecture

```text
User Query
    ↓
4mulaMind LLM
    ↓
Intent Detection
    ↓
Argument Parser
    ↓
Validator
    ↓
Safe Pipeline
    ↓
Engine Command
```

Example:

```text
User:
I need user 123 right now

↓

Intent:
SEARCH

↓

Arguments:
id = 123

↓

Command:
search,123
```

## Model Configuration

| Parameter           | Value |
| ------------------- | ----: |
| Vocabulary Size     |  4096 |
| Context Length      |   128 |
| Embedding Dimension |   256 |
| Attention Heads     |     8 |
| Hidden Dimension    |  1024 |
| Transformer Layers  |     6 |
| Checkpoint Step     |  1100 |

## Training Data

The project contains SFT datasets for 4mulaQuery intent understanding.

Main datasets:

```text
data/4mulaquery_sft.jsonl
data/4mulaquery_sft_raw.jsonl
```

The dataset contains examples for:

* Search operations
* Insert operations
* Delete operations
* All-record operations
* Invalid requests

## Safe Pipeline

The Safe Pipeline prevents unrestricted model output from directly reaching the database engine.

Processing:

```text
Natural Language
      ↓
Intent
      ↓
Arguments
      ↓
Validation
      ↓
Safe Command
```

Only validated commands are allowed to reach the execution layer.

## Evaluation

The Safe Pipeline was evaluated on **112 challenge cases**.

```text
Intent Accuracy:       84 / 112  = 75.00%
Safe Command Accuracy: 107 / 112 = 95.54%
Safe Rejections:       29
Unsafe Commands:       1
```

The evaluation includes valid and invalid natural-language requests and checks whether the resulting command is safe before execution.

## Project Structure

```text
4mulaMind-LLM/
│
├── checkpoints/
│
├── data/
│   ├── 4mulaquery/
│   ├── 4mulaquery_sft.jsonl
│   ├── 4mulaquery_sft_raw.jsonl
│   ├── train.jsonl
│   └── validation.jsonl
│
├── integration/
│   ├── argument_parser.py
│   ├── bridge.py
│   ├── executor.py
│   ├── intent_router.py
│   ├── safe_pipeline.py
│   ├── validator.py
│   └── evaluate_safe_pipeline.py
│
├── tokenizers/
│   └── 4mulamind_bpe.json
│
├── tests/
│
└── README.md
```

## Running the Evaluation

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the Safe Pipeline evaluation:

```bash
python -m integration.evaluate_safe_pipeline
```

## Example Commands

The model can map natural-language requests to controlled engine commands.

```text
Find user 42
→ search,42
```

```text
Remove user 321
→ delete,321
```

```text
Show all users
→ all
```

Invalid requests are rejected by the safe pipeline instead of being executed.

## Goal

The goal of 4mulaMind-LLM is not to build a general-purpose chatbot.

It is a focused experiment in building a **small custom language model for structured intent understanding, command generation, and safe execution**.

## Status

**Current Status: Experimental / Research Prototype**

The model, tokenizer, intent pipeline, argument parser, validator, executor, and evaluation pipeline are implemented.

---

Built by **4mulaMind**
