import jax.numpy as jnp
from jax import jit, vmap
from jax.nn import relu
from .dataset_creation import get_dataset_dicts
import torch
import pickle
import os


import jax.numpy as jnp
import os
import pickle


def tokenize_data(dataset, tokenizer, model, save_dir):
    """
    Tokenizes and generates embeddings for the dataset in a format compatible with JAX.
    Prompts are concatenated with responses (chosen/rejected) for DPO-style training.
    Saves embeddings to disk for reuse in future runs.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(save_dir, exist_ok=True)

    embeddings = []
    labels = []

    for idx, entry in enumerate(dataset):
        chosen_path = os.path.join(save_dir, f"chosen_{idx}.pkl")
        rejected_path = os.path.join(save_dir, f"rejected_{idx}.pkl")

        if os.path.exists(chosen_path) and os.path.exists(rejected_path):
            with open(chosen_path, "rb") as f:
                chosen_embedding = pickle.load(f)
            with open(rejected_path, "rb") as f:
                rejected_embedding = pickle.load(f)
        else:
            chosen_text = f"{entry['prompt']} Output: {entry['chosen']}"
            rejected_text = f"{entry['prompt']} Output: {entry['rejected']}"

            chosen_tokens = tokenizer(chosen_text, return_tensors="pt", padding=True, truncation=True)
            rejected_tokens = tokenizer(rejected_text, return_tensors="pt", padding=True, truncation=True)

            
            chosen_embedding = model(**chosen_tokens).last_hidden_state.mean(dim=1).squeeze(0).detach().numpy()
            rejected_embedding = model(**rejected_tokens).last_hidden_state.mean(dim=1).squeeze(0).detach().numpy()

            # save embeddings to disk
            with open(chosen_path, "wb") as f:
                pickle.dump(chosen_embedding, f)
            with open(rejected_path, "wb") as f:
                pickle.dump(rejected_embedding, f)

        embeddings.append(jnp.array(chosen_embedding))
        labels.append(1)  # chosen
        embeddings.append(jnp.array(rejected_embedding))
        labels.append(0)  # rejected

    embeddings = jnp.stack(embeddings)
    labels = jnp.array(labels)

    print(f"Final shapes: embeddings={embeddings.shape}, labels={labels.shape}")
    # Final shapes: embeddings=(9094, 1, 768), labels=(9094,) in embeddings3d without squeeze
    return embeddings, labels
