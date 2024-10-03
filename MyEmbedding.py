"""
  @FileName：MyEmbedding.py
  @Author：Excelius
  @CreateTime：2024/10/3 16:36
  @Company: None
  @Description：
"""
import numpy as np
from torch import Tensor


class MyEmbedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 随机初始化嵌入矩阵
        self.embedding = np.random.randn(num_embeddings, embedding_dim)

    def forward(self, indices):
        # 根据索引获取嵌入向量
        return self.embedding[indices]

    def update(self, indices, gradients):
        # 更新嵌入矩阵
        learning_rate = 0.01
        for idx, grad in zip(indices, gradients):
            self.embedding[idx] -= learning_rate * grad
