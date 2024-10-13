"""
  @FileName：Skip_gram_TF.py
  @Author：Excelius
  @CreateTime：2024/10/6 11:37
  @Company: None
  @Description：
"""
import collections
import math
import os
import logging
import random

import numpy as np
import tensorflow.compat.v1 as tf
from importlib import reload
from sklearn.manifold import TSNE
import pickle

"""----------参数----------"""
colab_cwd = '/content'
if os.getcwd() == colab_cwd:
    # 处理好的文本存储在谷歌云盘
    file_path = os.path.join(os.getcwd(), "drive", "MyDrive", "data", "txt", "wiki_seg.txt")
else:
    # 本地的文件路径
    file_path = os.path.join(os.getcwd(), "data", "txt", "wiki_seg.txt")
# 每次迭代将使用 256 个样本进行参数更新
batch_size = 256
# 单词转为稠密向量的维度
embedding_size = 350
# 左右考虑多少个单词
skip_window = 4
# 重复使用输入以生成标签的次数
num_skips = 8

# 用来抽取的验证单词数
valid_size = 10
# 验证单词只从频数最高的 100 个单词中抽取
valid_window = 100
# 需要验证的单词
valid_has_word = ['父亲', '中国', '电脑', '手机', '书籍', '公里', '还', '部分', '年', '之后']
# valid_examples = np.array(random.sample(range(valid_window), valid_size))
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 负采样的噪声单词的数量
num_sampled = 64

"""----------logging配置----------"""
# 使用 logging.info 打印信息，colab 需要 reload() 函数，否则无法打印
reload(logging)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""----------数据处理----------"""


def read_data(filename):
    """
    从本地文件中读取数据，
    @param filename:
    @return:
    """
    logging.info("开始读取文件：" + filename)
    with open(filename, "r", encoding="utf-8") as f:
        # 一次性读取所有的行，返回一个列表
        seg_content = f.readlines()
        seg_words = []
        # 将单词都存放到seg_words列表中，不去重，方便下一步建立以词频为基础的词表
        for i in range(len(seg_content)):
            if i % 5000 == 0:
                logging.info("当前读取到第 " + str(i) + " 行，部分数据为: " + str(seg_content[i][:50]))
            # 获取当前行词的列表，使用strip()函数过滤空格包括换行符或者制表符
            seg_words.extend(seg_content[i].strip().split(" "))
    return seg_words


all_words_list = read_data(file_path)


def build_dataset(words, size):
    """
    创建词表，包括原始词表、转换后的索引词表、单词-索引表和索引-单词表
    @param words: 原始词表
    @param size: 高频词词表长度
    @return: 索引词表、原始词表、单词-索引词表、索引-单词词表
    """
    # 初始化 count 列表，其中 UNK 代表 unknown, 即高频词表以外的词
    _count = [['UNK', -1]]
    # 统计词频，只取前 vocabulary_size 个高频词, 格式为：('词', 词频)
    _word_collection = collections.Counter(words)
    logging.info('所有词的数量为：' + str(len(_word_collection)))
    _count.extend(_word_collection.most_common(size - 1))
    logging.info("高频词表前 20 个数据为数据为: " + str(_count[:20]))
    # 构建字典，将词转化为索引, 词典顺序为高频词顺序, 格式为('词', 索引), 其中索引从 0 开始
    _dictionary = dict()
    for _word, _ in _count:
        _dictionary[_word] = len(_dictionary)
    # 将词转化为索引存储到_data中, 词在高频词词典的话，索引为高频词词典的索引, 如果词不在字典中, 则转化为 UNK
    # 此时 _data 就是原来词的列表的索引列表
    _data = list()
    unk_count = 0
    for _word in words:
        if _word in _dictionary:
            index = _dictionary[_word]
        else:
            index = 0
            unk_count += 1
        _data.append(index)
    _count[0][1] = unk_count
    # 构建反向字典, 可以快速从词索引转化为词即 (索引-'词')
    _reverse_dictionary = dict(zip(_dictionary.values(), _dictionary.keys()))
    return _data, _count, _dictionary, _reverse_dictionary


# 高频词词表大小, 对于 4w 的切片， 10w 词表的结果较好
vocabulary_size = 800000
# 构建词表
data, count, dictionary, reverse_dictionary = build_dataset(all_words_list, vocabulary_size)
logging.info('所有词的数量为：' + str(len(count)))
logging.info('含 UNK 的前 20 个高频词' + str(count[: 20]))
logging.info('查看索引与词的映射：')
logging_str = ''.join(
    [f"{idx} : {word}, " for idx, word in zip(data[:10], [reverse_dictionary[i] for i in data[:10]])])
logging.info(logging_str)
# 完善验证单词的索引
valid_examples = []
# 如果待验证词在此表中，那么直接获得索引，否则指定为 unknown
for word in valid_has_word:
    if dictionary.get(word, None) is not None:
        valid_examples.append(dictionary[word])
    else:
        valid_examples.append(0)

# 删除原始词表，节省内存
del all_words_list
data_index = 0


def generate_batch(_batch_size, _num_skips, _skip_window):
    """
    生成训练用的 batch 数据以及标签数据，其中 batch 是 上下文， label 是 目标词
    @param _batch_size: batch 的大小
    @param _num_skips: 目标词的上下文窗口大小（只有一半）
    @param _skip_window: 目标词的上下文窗口大小（只有一半）
    @return: 生成的 batch 数据 和 label 数据
    """
    global data_index
    assert _batch_size % _num_skips == 0
    assert _num_skips <= 2 * _skip_window
    # 初始化二维数组, 行数为 _batch_size，列数为 2 * half_window_size, 数据类型为 int32
    _batch = np.ndarray(shape=(_batch_size), dtype=np.int32)
    _labels = np.ndarray(shape=(_batch_size, 1), dtype=np.int32)
    span = 2 * _skip_window + 1
    len_data = len(data)
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len_data
    for i in range(_batch_size // _num_skips):
        target = _skip_window  # target label at the center of the buffer
        targets_to_avoid = [_skip_window]
        for j in range(_num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            _batch[i * _num_skips + j] = buffer[_skip_window]
            _labels[i * _num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len_data
    return _batch, _labels


"""----------训练 & 验证----------"""
# 创建默认的 graph
graph = tf.Graph()
with graph.as_default():
    '''-----输入-----'''
    # 创建一个 TensorFlow 占位符的，类型为 32 位整数，大小为 batch_size 的一维张量
    # train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    # 创建一个 TensorFlow 占位符的，类型为 32 位整数，大小为 batch_size x 1 的二维张量
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # 创建常量张量, 值为 valid_examples 列表中的元素, 制定了数据类型, 用于后续作为固定输入
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    '''-----变量-----'''
    # 词嵌入矩阵, 形状为 [vocabulary_size, embedding_size], 通过训练，嵌入矩阵会学习到每个词的分布式表示（即词向量）
    # tf.random.uniform() 用于从均匀分布中随机初始化嵌入向量，范围是 [-1.0, 1.0]。给每个词的嵌入向量的初始值是在 -1 到 1 之间的随机数。
    embeddings = tf.Variable(tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # softmax 层的权重矩阵，形状为 [vocabulary_size, embedding_size], 用于在通过模型训练时，将词嵌入向量映射到词汇表中每个词的概率分布
    # tf.random.truncated_normal() 用于从截断的正态分布中随机初始化权重。
    # 标准差 stddev=1.0 / math.sqrt(embedding_size) 是一种常见的初始化方式，它有助于在训练开始时使网络中的权重分布更加合理，避免梯度消失或梯度爆炸问题。
    softmax_weights = tf.Variable(
        tf.random.truncated_normal([vocabulary_size, embedding_size],
                                   stddev=1.0 / math.sqrt(embedding_size)))
    # softmax_biases 是 softmax 层的偏置项，形状为 [vocabulary_size]，表示每个词汇都有一个对应的偏置
    # tf.zeros([vocabulary_size]) 使用全零初始化偏置项。这是常见的初始化方式，偏置值会在训练过程中更新
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    '''-----模型-----'''
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 转化变量输入，适配NCE
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         inputs=embed,
                                         labels=train_labels,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    # # 将嵌入矩阵每行元素求和得到一个向量
    # inputs = tf.reduce_sum(embed, 1)
    # # 计算 softmax 损失，每次使用负标签样本
    # loss = tf.reduce_mean(
    #     tf.nn.sampled_softmax_loss(
    #         softmax_weights, softmax_biases, train_labels, embed, num_sampled, vocabulary_size
    #     )
    # )
    '''-----Optimizer-----'''
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    embeddings_2 = (normalized_embeddings + softmax_weights) / 2.0
    norm_ = tf.sqrt(tf.reduce_sum(tf.square(embeddings_2), 1, keep_dims=True))
    normalized_embeddings_2 = embeddings_2 / norm_

num_steps = 1000001

with tf.Session(graph=graph) as session:
    if int(tf.version.VERSION.split('.')[1]) > 11:
        tf.compat.v1.global_variables_initializer().run()
    else:
        tf.initialize_all_variables().run()
    logging.info("初始化完成...")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            _batch_size=batch_size,
            _num_skips=num_skips,
            _skip_window=skip_window
        )
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            logging.info("在第 %d 次循环的平均损失为: %f" % (step, average_loss))
            average_loss = 0

        if step % 10000 == 0:
            logging.info("验证单词与最相似的 10 个单词：")
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 10  # 取最相似的 10 个单词
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "单词 %s 的最相似的 10 个单词为：" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                logging.info(log_str)
    final_embeddings = normalized_embeddings.eval()
    final_embeddings_2 = normalized_embeddings_2.eval()  # 更好的结果

"""----------结果----------"""
# 后续需要处理的点的数量
num_points = 10000
# 创建了 t-SNE 对象 tsne，n_components 指定降维后的维度为2，init 指定了初始化方法为 pca，n_iter 指定了最大迭代次数为 5000
tsne_2 = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# 对 final_embeddings 中的前 400 个嵌入（从索引 1 到 400）进行 t-SNE 降维，并将结果存储在 two_d_embeddings 中
two_d_embeddings = tsne_2.fit_transform(final_embeddings[1:num_points + 1, :])
two_d_embeddings_2 = tsne_2.fit_transform(final_embeddings_2[1:num_points + 1, :])
if os.getcwd() == colab_cwd:
    # 处理好的文本存储在谷歌云盘
    output_path = os.path.join(os.getcwd(), "drive", "MyDrive", "data", "embedding_skip_gram.pkl")
else:
    # 本地的文件路径
    output_path = os.path.join(os.getcwd(), "data", "embedding_skip_gram.pkl")
with open(output_path, 'wb') as f:
    pickle.dump([final_embeddings[:800000, :], final_embeddings_2[:800000, :], two_d_embeddings, two_d_embeddings_2,
                 reverse_dictionary], f)
