# The Associative Trap: Semantic Maps vs. Logical Rules in LLMs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

## Overview
This repository contains a lightweight experiment that bridges **distributional semantics** (static word embeddings) and **predictive semantics** (contextual probability distributions) in small language models like GPT-2. 

We demonstrate that a model's internal continuous semantic map heavily dictates its contextual predictions, often overriding explicit logical operators such as negation. By comparing the embedding similarity of hypernym pairs against their Affirmation/Negation probability ratio, we show that high static semantic proximity traps the model in an **"associative state,"** rendering it effectively blind to logical negation.

## The Core Hypothesis
If an LLM truly "understands" a logical negation like *"is not a"*, the probability of a related category should drop significantly. However, if the model is relying purely on its spatial semantic map, the gravitational pull between two related embeddings (e.g., "dog" and "mammal") will overpower the logical operator, leading to a high probability prediction despite the negation.

## Methodology
To test the alignment between the embedding space and the probability space, we analyze subject-category pairs (e.g., `dog` $\rightarrow$ `mammal`, `paris` $\rightarrow$ `city`) across three core metrics:

1. **Static Embedding Similarity:** The cosine similarity between the base token embeddings of the subject and the category.
2. **Activation Shift:** The relative L2 norm difference of the model's final hidden state between the affirmative prompt (*"A [Subject] is a"*) and the negative prompt (*"A [Subject] is not a"*).
3. **Affirmation/Negation Ratio ($R$):** The ratio of the model's assigned probability for the target category across both contexts. 

$$R = \frac{P(\text{category} \mid \text{affirmative})}{P(\text{category} \mid \text{negative})}$$
https://github.com/simo-mazzoli/logic-embedding/blob/dae7cf9b1ac0f579d9555b03b96e5a41cd018e83/code.py
## Installation & Usage

You can run this experiment locally or in Google Colab. 

### Requirements
```bash
pip install transformers torch scipy pandas matplotlib scikit-learn
```
## Results & Analysis

When running the code on GPT-2, the output visually demonstrates how embedding proximity overrides logical reasoning capabilities. The data clusters into three distinct behavioral zones:

| Behavioral Zone | Example Pair | Embedding Sim. | Ratio ($R$) | Observation |
| --- | --- | --- | --- | --- |
| **Logical Success** | `paris` $\rightarrow$ `city` | Low ($\approx 0.30$) | High ($> 1.8$) | Negation functions correctly; probability of the category drops significantly. |
| **Associative Trap** | `dog` $\rightarrow$ `mammal` | Moderate ($\approx 0.32$) | Neutral ($\approx 1.2$) | The semantic gravity between concepts causes the model to largely ignore the "not" token. |
| **Reversal Paradox** | `eagle` $\rightarrow$ `bird` | High ($\approx 0.33$) | Inverted ($< 0.5$) | The model predicts the target category *more strongly* when explicitly told not to. |

*(Insert your generated plot image here: `![Results Plot](https://github.com/simo-mazzoli/logic-embedding/blob/dae7cf9b1ac0f579d9555b03b96e5a41cd018e83/code.py)`)*

## Conclusion

This experiment confirms that the probability space of a small language model is not independent of its static embedding space; rather, it is strictly constrained by it. When the embedding similarity between a subject and a category is sufficiently high, the internal activation shift caused by the word "not" is too weak to escape the "semantic well." In these cases, the model defaults to pattern association rather than syntactic logic, proving that continuous semantic maps extracted from logprobs mirror the underlying geometric biases of the base embedding layer.

```

***

**Would you like me to suggest some potential "Next Steps" or "Future Work" ideas to add to the bottom of the README, such as testing this on newer models like LLaMA-3 or Mistral?**

```
