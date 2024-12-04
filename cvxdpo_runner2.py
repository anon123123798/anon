import jax.numpy as jnp
from flax.training import train_state
import jax
#import transformers
from transformers import GPT2Tokenizer, GPT2Model, FlaxGPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from solve.models.cvx_relu_mlp import CVX_ReLU_MLP  
from solve.optimizers.cvxdpo import run, init_cvxdpo_state 
from utils.data_utils import tokenize_data
from inference import inference
from utils.dataset_creation import get_dataset_dicts
from jax.extend.backend import get_backend
from jax import lax
import wandb
import time
from jax.lib import xla_bridge
import os
import pickle
from jax import random as jrn


# TO DO: check dtype of jax bfloat16 everywhere
# JAX version should be 0.4.33

print(get_backend().platform)
print("Running JAX Version =",jax.__version__)


# models: dilstilGPT, GPT2, GPT2-M, FlaxGPT2 
# methods: DPO, SFT, SimPO, ORPO
# datasets: imdB, StanfordSHP, edu-tutor 

SEED = 1024
P_S = 10 # or 20 or 40
PROJECT_DIR = os.getcwd()
SAVE_DIR = os.path.join(PROJECT_DIR, "embeddings")
os.makedirs(SAVE_DIR, exist_ok=True)  

MODEL_NAME = 'distilgpt2'
policy_dtype = jnp.bfloat16 

wandb.init(
    project="cvx_dpo",
    name="RUN1",
    config={
        "model": MODEL_NAME,
        "P_S": P_S,
        "optimizer": "CVX-DPO",
        "seed": SEED
    }
)

def initialize_pretrained_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, clean_up_tokenization_spaces=False)
    
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # check pad_token is explicitly set
    model = GPT2Model.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)

    return tokenizer, model



# cvxdpo: this is the main function being called in the loop below
def train_with_cvxdpo(embeddings, labels):
    """
    Trains a convex neural network using the cvxdpo optimizer in JAX.
    """
    print("Starting training...")

    n_samples = embeddings.shape[0]  # Number of samples
    feature_dim = embeddings.shape[-1]  # Feature dimension

    print(f"Training data shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    permutation = jax.random.permutation(jrn.PRNGKey(SEED), len(labels))
    embeddings = embeddings[permutation]
    labels = labels[permutation]

    model = CVX_ReLU_MLP(
        X=embeddings, 
        y=labels,  
        P_S=P_S, 
        beta=10**-3,
        rho=0.1,
        seed=jrn.PRNGKey(SEED)
    )

    model.init_model()

    state = init_cvxdpo_state(feature_dim, embeddings.shape[0], P_S)

    admm_params = {
        "rank": 10,  # Preconditioner rank
        "beta": 0.01,
        "gamma_ratio": 0.5,
        "admm_iters": 100,
        "pcg_iters": 20,
        "check_opt": True,
        "verbose": True
    }

    print("Training with cvxdpo...")
    (v, w), metrics = run(model, admm_params, model_type='CReLU')

    return v, w, metrics


# log VRAM
def get_vram_usage():
    backend = xla_bridge.get_backend()
    return backend.memory_info().used / (1024 ** 2)  # GPU VRAM usage in MB

# moved log tflops in def run 
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tokenizer, pretrained_model = initialize_pretrained_model()
    print("Finished loading model!")

    dataset = get_dataset_dicts()
    print("Finished loading dataset, checking for saved embeddings...")

    embeddings_exist = all(
        os.path.exists(os.path.join("/home/miria/direct-preference-optimization/embeddings", f"{name}_{i}.pkl"))
        for i in range(len(dataset))
        for name in ["chosen", "rejected"]
    )

    if embeddings_exist:
        print("Loading embeddings...")
        embeddings, labels = [], []
        for i in range(len(dataset)):
            with open(os.path.join(SAVE_DIR, f"chosen_{i}.pkl"), "rb") as f:
                embeddings.append(jnp.array(pickle.load(f)))
                labels.append(1)  # Chosen label
            with open(os.path.join(SAVE_DIR, f"rejected_{i}.pkl"), "rb") as f:
                embeddings.append(jnp.array(pickle.load(f)))
                labels.append(0)  # Rejected label

        embeddings = jnp.stack(embeddings)
        labels = jnp.array(labels)

    else:
        print("Generating a new batch of embeddings...")
        embeddings, labels = tokenize_data(dataset, tokenizer, pretrained_model, save_dir=SAVE_DIR)
        print("Generated and saved embeddings to disk.")

    print(f"Embeddings dtype: {embeddings.dtype}, shape: {embeddings.shape}")
    print(f"Labels dtype: {labels.dtype}, shape: {labels.shape}")

    v, w, metrics = train_with_cvxdpo(embeddings, labels)

    # # Save weights to disk for later use
    # os.makedirs("results", exist_ok=True)
    # with open("results/weights_v.safetensors", "wb") as f:
    #     pickle.dump(v, f)
    # with open("results/weights_w.safetensors", "wb") as f:
    #     pickle.dump(w, f)

    wandb.finish()


    # inference to gen for results, shown here to demo pipeline for experiments run inference.py
    test_prompt = "What is a Shakespearean sonnet?"
    result = inference(test_prompt, tokenizer, pretrained_model, v, w)
    print(f"Result for '{test_prompt}': {result}")


# ------------------------------------------------------------------------------
# convex dpo loss for completeness
def convex_dpo_loss(predictions_chosen, predictions_rejected, beta):
    """
    Computes the convexified DPO loss for evaluation.
    """
    logits = predictions_chosen - predictions_rejected
    return -jnp.mean(jnp.log(1 + jnp.exp(-beta * logits)))
