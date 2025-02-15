import jax.numpy as jnp
from jax import jit

def proxl2_vec(z: jnp.ndarray,
               beta: float,
               gamma: float):
    return (1 - beta * gamma / jnp.max(beta*gamma,jnp.linalg.norm(z))) * z


@jit 
def proxl2_tensor(z: jnp.ndarray, 
           beta: float, 
           gamma: float):
    """
    Proximal l2 for ADMM update step on (v,w).
    """
    norms = jnp.linalg.norm(z, axis=0)
    return (1 - beta * gamma /jnp.maximum(beta*gamma,norms)) * z
 