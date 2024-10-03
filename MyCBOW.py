"""
  @FileName：MyCBOW.py
  @Author：Excelius
  @CreateTime：2024/9/25 14:42
  @Company: None
  @Description：
"""
from torch import nn
import torch.nn.functional as torch_functional

from MyEmbedding import MyEmbedding


class MyCBOW(nn.Module):
    def __init__(self, all_words_set_size, embedding_dimension):
        super(MyCBOW, self).__init__()
        # 神经网络层，使用的是 nn.Embedding 类，该类会根据输入的词汇表大小和指定的嵌入维度创建一个查找表
        # 每个词都会被映射到这个表中的一个唯一向量，从而将离散的词语转化为可以进行数学运算的连续向量表示
        self.embeddings = nn.Embedding(all_words_set_size, embedding_dimension)
        # 全连接层 (nn.Linear) 用于将词嵌入向量的维度从 embedding_dimension 转换为 128
        # 它的主要作用是对嵌入向量进行线性变换，以便后续层可以处理经过转换后的特征
        # 这有助于降低维度或调整向量空间, 适应进一步的计算需求
        self.proj = nn.Linear(embedding_dimension, 128)
        # 全连接层，输出层，用于将128维向量转换为词汇表大小的输出向量
        self.output = nn.Linear(128, all_words_set_size)

    def forward(self, inputs):
        # 对输入生成的词嵌入进行求和，并转置为行向量
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        # 先将求和后的向量通过全连接层进行线性变换，然后再应用ReLU激活函数
        out = torch_functional.relu(self.proj(embeds))
        # 将线性变换后的结果作为输入传递给输出层
        out = self.output(out)
        # 使用log_softmax计算输出的在最后一步概率分布
        nll_prob = torch_functional.log_softmax(out, dim=-1)
        return nll_prob
