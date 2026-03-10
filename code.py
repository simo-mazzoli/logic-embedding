import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.spatial.distance import cosine
import torch.nn.functional as F

# 1. Setup Model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        input_ids = inputs['input_ids'][0]
        return model.transformer.wte(input_ids).mean(dim=0).numpy()

def get_prompt_state_and_probs(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0, -1, :]
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        return last_hidden, probs

# 2. Test Data
test_data = [
    ("dog", "mammal"), ("eagle", "bird"), ("rose", "flower"),
    ("hammer", "tool"), ("table", "furniture"), ("python", "language"),
    ("paris", "city"), ("gold", "metal")
]

results = []

# 3. Execution
for sub, cat in test_data:
    sim = 1 - cosine(get_word_embedding(sub), get_word_embedding(cat))
    
    state_is, probs_is = get_prompt_state_and_probs(f"A {sub} is a")
    state_not, probs_not = get_prompt_state_and_probs(f"A {sub} is not a")
    
    cat_id = tokenizer.encode(" " + cat)[0]
    p_is, p_not = probs_is[cat_id].item(), probs_not[cat_id].item()
    ratio = p_is / p_not if p_not > 0 else float('inf')
    
    shift = torch.norm(state_not - state_is).item() / torch.norm(state_is).item()
    
    results.append({"Pair": f"{sub} -> {cat}", "Embed Sim": sim, "Act Shift %": shift * 100, "Logprob Ratio": ratio})

df = pd.DataFrame(results)

# 4. Visualization
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df["Embed Sim"], df["Logprob Ratio"], c=df["Act Shift %"], cmap='viridis_r', s=150, edgecolors='k')
for i, row in df.iterrows():
    plt.annotate(row["Pair"], (row["Embed Sim"], row["Logprob Ratio"]), xytext=(8, 8), textcoords='offset points')
plt.colorbar(scatter, label="Activation Shift %")
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
plt.title("The Associative Trap")
plt.xlabel("Embedding Similarity")
plt.ylabel("Affirmation/Negation Ratio")
plt.show()
