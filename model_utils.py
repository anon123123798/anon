import jax.numpy as jnp
import jax.random as jrn

def get_hyperplane_cuts(X: jnp.ndarray, P: int, seed=None) -> jnp.ndarray:
    """
    Function to sample diagonals of D_i matrices given training data X using JAX.

    Parameters
    ----------
    X : jnp.ndarray
        Training data.
    P : int
        Number of unique hyperplane samples desired.
    seed : int, optional
        Randomized seed for repeatable results.

    Returns
    -------
    d_diags: jnp.ndarray
        n x P matrix, where each column i is the diagonal entries for D_i.
    """
    n, d = X.shape
    if seed is not None:
        key,subkey = jrn.split(seed)
    else:
        raise ValueError("Missing seed parameter")

    # Generate a random matrix and perform the matrix multiplication and comparison
    random_matrix = jrn.normal(subkey, (d, P))
    d_diags = (X @ random_matrix) >= 0

    # JAX does not have a direct equivalent to numpy's unique function along an axis,
    # we need to implement a workaround if we want to ensure the columns are unique.
    # we temp skip the uniqueness check for simplicity, as it's non-trivial and may not be needed.
    # if uniqueness is required, a custom function to filter unique columns should be implemented.

    return d_diags.astype(jnp.float32), key  # Ensure the default datatype is float32

def get_grelu_patterns(X: jnp.ndarray, P: int, seed=None) -> jnp.ndarray:
    n, d = X.shape
    if seed is not None:
        key, subkey = jrn.split(seed)
        random_tensor = jrn.normal(subkey, (2, d, P))
    else:
        raise ValueError("Missing seed parameter")
    d_diags = (X@random_tensor[0, :])>=0
    gates = random_tensor[1, :]
    return d_diags, gates, key 

def relu_optimal_weights_transform(v: jnp.ndarray, 
                              w: jnp.ndarray, 
                             ):
    """
    Given optimal v^*, w^* of convex problem (Eq (2.1)), derive the optimal weights u^*, alpha^* of the non-convex probllem (Eq (2.1))
    Applies Theorem 1 of Pilanci, Ergen 2020

    Parameters
    ----------
    v : ArrayType
        v weights in convex formulation
    w : ArrayType
        w weights in convex formulation

    verbose: boolean
        true to print weight transform information
   
    Returns
    -------
    (u, alpha) : Tuple[ArrayType, ArrayType]
        the transformed optimal weights
    """

    assert v is not None
    assert w is not None
    
    # Causes error due to boolean so is commented out for now
    # ensure shapes are correct
    #if v.shape == (P_S, d): v = v.T
    #if w.shape == (P_S, d): w = w.T
    #assert v.shape == (d, P_S), f"Expected weight v shape to be ({d},{P_S}), but got {v.shape}"
    #assert w.shape == (d, P_S), f"Expected weight w shape to be ({d},{P_S}), but got {w.shape}"

    # if verbose: 
    #     datatype_backend = get_backend_type(v)
    #     print(f"\tDoing weight transform: ")
    #     v_shp = v.cpu().numpy().shape if datatype_backend == "torch" else v.shape
    #     w_shp = w.cpu().numpy().shape if datatype_backend == "torch" else w.shape
    #     print(f"  starting v shape: {v_shp}")
    #     print(f"  starting w shape: {w_shp}")
    #     print(f"\t  (d, P_S): ({d}, {P_S})")

    alpha1 = jnp.sqrt(jnp.linalg.norm(v, 2, axis=0))
    mask1 = jnp.where(alpha1!=0)[0]
    u1 = v[:, mask1] / alpha1[mask1]
   
    alpha2 = -jnp.sqrt(jnp.linalg.norm(w, 2, axis=0))
    mask2 = jnp.where(alpha2!=0)[0]
    u2 = -w[:, mask2] / alpha2[mask2]

    u = jnp.append(u1, u2, axis=1)
    alpha = jnp.append(alpha1[mask1], alpha2[mask2])

    return u, alpha

def grelu_optimal_weights_transform(u: jnp.ndarray,  
                                    P_S: int, 
                                    d: int):
    W1 = jnp.zeros((d, P_S))
    w2 = jnp.sqrt(jnp.linalg.norm(u, 2, axis=0))
    non_zeros_1 = jnp.where(w2!=0)[0]
    W1 = W1.at[:, non_zeros_1].set(u[:, non_zeros_1] / w2[non_zeros_1])

    return W1, w2 