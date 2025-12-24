import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array


@jax.jit
def _compute_batch_stats(z: Array, next_z: Array, rewards: Array, done: Array, 
                         gamma: float):
    """
    Raw Sums
    """
    rewards = rewards.reshape(-1, 1)
    done = done.reshape(-1, 1)
    B, F = z.shape
    
    # 1. 拼接 Bias (Augment with 1)
    ones = jnp.ones((B, 1))
    z_aug = jnp.concatenate([z, ones], axis=1)           # [B, F+1]
    next_z_aug = jnp.concatenate([next_z, ones], axis=1) # [B, F+1]
    
    # 2. 计算差分项
    # diff = phi(s) - gamma * phi(s')
    diff = z_aug - gamma * next_z_aug * (1 - done)
    
    # 3. 计算原始矩阵和 (Raw Matrix Sums)
    # A_batch = sum( z * diff.T )
    A_batch = z_aug.T @ diff
    
    # b_batch = sum( z * r )
    b_batch = z_aug.T @ rewards
    
    return A_batch, b_batch, B


class FQE():

    def __init__(self, args):
        self.F = args.feature_dim
        self.F_aug = self.F + 1
        self.lamda = 1e-3
        self.gamma = args.gamma
        self.reset_stats()
        self.w = None
    
    def reset_stats(self):
        """清空累加器，准备开始新一轮计算"""
        self.A_sum = jnp.zeros((self.F_aug, self.F_aug))
        self.b_sum = jnp.zeros((self.F_aug, 1))
        self.total_count = 0


    def add_batch(self, z: Array, next_z: Array, rewards: Array, done: Array):
        """
        处理一个 Batch，并将结果累加到 self.A_sum 和 self.b_sum 中。
        """
        # 调用 JIT 函数计算当前 batch
        A_batch, b_batch, count = _compute_batch_stats(
            z, next_z, rewards, done, self.gamma
        )
        
        # 累加 (Accumulate)
        self.A_sum += A_batch
        self.b_sum += b_batch
        self.total_count += count
        
        return count # 返回当前处理的数量，方便打印进度
 
    def solve(self):
        """
        所有 Batch 都跑完后，调用此函数进行求解。
        """
        if self.total_count == 0:
            raise ValueError("没有数据！请先调用 add_batch。")

        # 1. 归一化 (Normalize by Total N)
        # A_mean = (1/N) * Sum_A
        A_mean = self.A_sum / self.total_count
        b_mean = self.b_sum / self.total_count
        
        # 2. 正则化 (Regularization)
        # A_final = A_mean + lambda * I
        A_final = A_mean + self.lamda * jnp.eye(self.F_aug)
        
        # 3. 求解 (Solve)
        print(f"Solving Linear System with N={self.total_count} samples...")
        self.w = jnp.linalg.solve(A_final, b_mean)
        
        # 可选：计算一下 training loss (MSBE)
        # 注意：这里的 loss 是基于 summary statistics 的估计，或者你可以再跑一遍数据算精确 loss
        # 简单的验证：检查 w 的模长
        w_norm = jnp.linalg.norm(self.w)
        return w_norm

    @staticmethod
    @jax.jit
    def _compute_loss_jit(z, next_z, rewards, done, gamma, w):
        """
        专门用于计算 Loss 的静态 JIT 函数
        """
        rewards = rewards.reshape(-1, 1)
        done = done.reshape(-1, 1)
        B = z.shape[0]
        # 1. 构造特征
        ones = jnp.ones((B, 1))
        z_aug = jnp.concatenate([z, ones], axis=1)
        next_z_aug = jnp.concatenate([next_z, ones], axis=1)
        
        # 2. 预测 Q
        q_pred = z_aug @ w
        
        # 3. 计算 Target
        q_next = next_z_aug @ w
        q_target = rewards + gamma * q_next * (1 - done)
        
        # 4. 计算均方误差 (MSBE)
        loss = jnp.mean((q_pred - q_target) ** 2)
        return loss
    
    def calculate_loss(self, z, next_z, rewards, done):
        """
        外部调用的 Loss 计算接口
        """
        if self.w is None:
            raise ValueError("请先调用 solve() 算出 w 后再计算 Loss！")
            
        return self._compute_loss_jit(
            z, next_z, rewards, done, self.gamma, self.w
        )

    @staticmethod
    @jax.jit
    def _predict_jit(z, w):
        B = z.shape[0]
        z_aug = jnp.concatenate([z, jnp.ones((B, 1))], axis=1)
        return z_aug @ w

    def evaluate(self, z: Array):
        if self.w is None:
            raise ValueError("请先运行 add_batch 和 solve 来训练权重！")
        return self._predict_jit(z, self.w)
