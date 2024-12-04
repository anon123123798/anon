import sys
import jax.numpy as jnp
from functools import partial
from jax import jit
import jax
from jax.nn import relu
from .pcg import pcg
from ..preconditioner.nystrom import Nys_Precond, rand_nys_appx
from ..utils.metric_utils import mse, compute_bin_acc
from ..utils.proximal_utils import proxl2_tensor
from time import perf_counter
from typing import NamedTuple, Tuple
import time
import wandb
import subprocess

class cvxdpo_State(NamedTuple):
    u: jnp.ndarray
    v: jnp.ndarray
    s: jnp.ndarray
    lam: jnp.ndarray
    nu: jnp.ndarray
    Gu: jnp.ndarray

# Helper to initialize cvxdpo state
def init_cvxdpo_state(d, n, P_S):
    u = jnp.zeros((2, d, P_S))
    v = jnp.zeros((2, d, P_S))
    s = jnp.zeros((2, n, P_S))
    lam = jnp.zeros((2, d, P_S))
    nu = jnp.zeros((2, n, P_S))
    return cvxdpo_State(u=u, v=v, s=s, lam=lam, nu=nu, Gu=jnp.zeros((2, n, P_S)))

# Helper for validation metrics
def compute_validation_metrics(metrics, u, model):
    y_hat = model.matvec_F(u)
    W1, w2 = model.get_ncvx_weights(u)
    y_hat_val = model.predict(model.Xval, W1, w2)

    metrics['train_loss'].append(mse(y_hat, model.y))
    metrics['val_loss'].append(mse(y_hat_val, model.yval))
    metrics['train_acc'].append(compute_bin_acc(y_hat, model.y))
    metrics['val_acc'].append(compute_bin_acc(y_hat_val, model.yval))

    return metrics

# Function that executes 1 step of cvxdpo
@partial(jit, static_argnames=['pcg_iters'])
def cvxdpo_step(state: cvxdpo_State, 
                model,
                Mnys, 
                beta: float, 
                gamma_ratio: float, 
                pcg_iters: int,
                pcg_tol: float):
    
    b_1 = model.rmatvec_F(model.y) / model.rho
    b = b_1 + state.v - state.lam + model.rmatvec_G(state.s - state.nu)

    # u update via PCG
    u, _, _ = pcg(b, model, Mnys, pcg_iters, pcg_tol)

    # v update using prox operator
    v = state.v.at[0].set(proxl2_tensor(u[0] + state.lam[0], beta=beta, gamma=1 / model.rho))
    v = v.at[1].set(proxl2_tensor(u[1] + state.lam[1], beta=beta, gamma=1 / model.rho))

    # s update using ReLU
    Gu = model.matvec_G(u)
    s = relu(Gu + state.nu)

    # dual updates
    lam = state.lam + (u - v) * gamma_ratio
    nu = state.nu + (Gu - s) * gamma_ratio

    return cvxdpo_State(u=u, v=v, s=s, lam=lam, nu=nu, Gu=Gu)

@partial(jit, static_argnames=['beta'])
def opt_conds(state: cvxdpo_State, model, beta: float)->Tuple[float,float,float]:
    y_hat = model.matvec_F(state.u)
    u_v_dist = jnp.linalg.norm(state.u - state.v) + jnp.linalg.norm(state.Gu - state.s)
    u_optimality = jnp.linalg.norm(model.rmatvec_F(y_hat - model.y.squeeze()) + model.rho * (state.lam + model.rmatvec_G(state.nu)))
    v_optimality = jnp.linalg.norm(beta * state.v / jnp.linalg.norm(state.v, axis=2, keepdims=True) - model.rho * state.lam)
    return u_v_dist, u_optimality, v_optimality


# log TFLOPS
def get_tflops(start_time, num_operations):
    elapsed_time = time.time() - start_time  # Time in seconds
    return num_operations / (elapsed_time * 1e12)  # TFLOPs

def get_vram_usage():
    """
    Returns current and maximum GPU VRAM usage using nvidia-smi.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            output = result.stdout.strip().split("\n")[0]
            used, total = map(int, output.split(", "))
            return used, total  # Return in MB
        else:
            raise RuntimeError(f"nvidia-smi error: {result.stderr}")
    except Exception as e:
        print(f"Could not retrieve VRAM usage: {e}")
        return 0, 0

# Function that runs cvxdpo optimizer
def run(model, admm_params: dict, model_type: str):
    """
    Executes the cvxdpo optimizer.

    Args:
        model: The cvxdpo model to optimize.
        admm_params: Dictionary containing ADMM parameters.

    Returns:
        A tuple containing the optimized weights (v, w) and the metrics dictionary.
    """
    # Extract parameters for cvxdpo
    rank = admm_params['rank']
    beta = admm_params['beta']
    gamma_ratio = admm_params['gamma_ratio']
    admm_iters = admm_params['admm_iters']
    pcg_iters = admm_params['pcg_iters']
    check_opt = admm_params['check_opt']
    verbose = admm_params['verbose']

    validate = model.Xval is not None
    n, d = model.X.shape

    state = init_cvxdpo_state(d, n, model.P_S)

    metrics = {'train_loss': [], 'train_acc': [], 'times': []}
    if validate:
        metrics.update({'val_loss': [], 'val_acc': []})
        metrics = compute_validation_metrics(metrics, state.u, model)
        best_model_dict = {'v': state.u[0], 'w': state.u[1], 'iteration': 0}

    U, S, model.seed = rand_nys_appx(model, rank, model_type, model.seed)
    Mnys = Nys_Precond(U, S, d, model.rho, model.P_S, model_type)

    for k in range(admm_iters):
        start = time.perf_counter()
        
        state = cvxdpo_step(state, model, Mnys, beta, gamma_ratio, pcg_iters, 1 / (1 + k) ** 1.2)

        if check_opt:
            u_v_dist, u_optimality, v_optimality = opt_conds(state, model, beta)
            if verbose:
                print(f"iter: {k}\n  u-v dist = {u_v_dist}, u resid = {u_optimality}, v resid = {v_optimality}\n")

        t_iter = time.perf_counter() - start
        metrics['times'].append(t_iter)

        y_hat = model.matvec_F(state.u)
        train_loss = mse(y_hat, model.y)
        train_acc = compute_bin_acc(y_hat, model.y)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)

        wandb.log({
            "iteration": k,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "time_per_iteration": t_iter
        })

        if validate:
            # Compute and log validation metrics
            metrics = compute_validation_metrics(metrics, state.u, model)
            val_acc = metrics['val_acc'][-1]
            val_loss = metrics['val_loss'][-1]

            wandb.log({
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            if val_acc > metrics['val_acc'][best_model_dict['iteration']] or \
               (val_acc == metrics['val_acc'][best_model_dict['iteration']] and val_loss < metrics['val_loss'][best_model_dict['iteration']]):
                best_model_dict.update({'v': state.u[0], 'w': state.u[1], 'iteration': k})

            # Early stopping condition
            if k >= 10 and k % 10 == 0 and val_acc <= metrics['val_acc'][best_model_dict['iteration']]:
                print("Validation accuracy is flat or decreasing. cvxdpo will now terminate and return the best model found.")
                return (best_model_dict['v'], best_model_dict['w']), metrics

        # Compute and log TFLOPS
        num_operations = 2 * d * n  
        tflops = get_tflops(start, num_operations)
        current_vram, max_vram = get_vram_usage()
        wandb.log({"TFLOPS": tflops})
        wandb.log({
            "VRAM_Usage_MB": current_vram,
            "Max_VRAM_Usage_MB": max_vram,
            "Timestamp": time.time(),
            "TFLOPS": tflops
        })

    return (state.u[0], state.u[1]), metrics




