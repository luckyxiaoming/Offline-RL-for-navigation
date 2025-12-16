import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array



@jax.jit
def _solve_lstd_core(z: Array, next_z: Array, rewards: Array, done: Array, 
                     gamma: float, lamda: float):
 

    B, F = z.shape
    # 1. Add Bias Term
    ones = jnp.ones((B, 1))
    z_aug = jnp.concatenate([z, ones], axis=1)           # [B, F+1]
    next_z_aug = jnp.concatenate([next_z, ones], axis=1) # [B, F+1]
    
    # 2.  diff = phi(s) - gamma * phi(s')
    diff = z_aug - gamma * next_z_aug * (1 - done)
    
    # A_mat = (1/N) * sum( z * diff.T ) + lambda * I
    F_aug = F + 1
    A_mat = (z_aug.T @ diff) / B + lamda * jnp.eye(F_aug)
    
    # b_vec = (1/N) * sum( z * r )
    b_vec = (z_aug.T @ rewards) / B
    
    # 3. Solve Linear System
    w = jnp.linalg.solve(A_mat, b_vec)
    
    # 4. Loss 
    q_pred = z_aug @ w
    q_next = next_z_aug @ w
    q_target = rewards + gamma * q_next * (1 - done)
    loss = jnp.mean((q_pred - q_target) ** 2)
    
    return w, loss

class FQE():
    


    def __init__(self, args):
        self.F = args.feature_dim
        self.lamda = 1e-3
        self.gamma = args.gamma
        self.w = None
 
    def linear_regression(self, z: Array, next_z: Array, rewards: Array, done: Array):

        w_new, loss = _solve_lstd_core(
            z, next_z, rewards, done, 
            self.gamma, self.lamda
        )
        
        self.w = w_new

        return loss.item()
    
    @staticmethod
    @jax.jit
    def _predict_jit(z, w):
        B = z.shape[0]
        z_aug = jnp.concatenate([z, jnp.ones((B, 1))], axis=1)
        return z_aug @ w

    def evaluate(self, z: Array):
        if self.w is None:
            raise ValueError("please run linear_regression to train weights!")
        
        return self._predict_jit(z, self.w)
