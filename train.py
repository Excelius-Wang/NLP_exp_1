"""
  @FileName：train.py
  @Author：Excelius
  @CreateTime：2024/9/22 16:12
  @Company: None
  @Description：
"""
import concurrent.futures
import logging
import os
import threading

import torch
from torch import optim, nn, unique
from tqdm import trange, tqdm

from MyCBOW import MyCBOW


def generate_context_vector(_context, word_to_idx):
    word_index_in_context = [word_to_idx[w] for w in _context]
    return torch.tensor(word_index_in_context, dtype=torch.long)


def add_word_to_all_word_set(content):
    # 本地集合，避免竞争
    unique_words = set()
    for count, line in enumerate(content):
        # 获取当前行词的列表，使用strip()函数过滤空格包括换行符或者制表符
        count_unique = 0
        seg_words = line.strip().split(" ")
        for word in seg_words:
            if word not in unique_words:
                unique_words.add(word)
                count_unique += 1
        if count % 5000 == 0:
            logging.info("当前线程 ID: " + str(threading.get_ident()) + " 处理第 " + str(count) + " 行处理完毕, " + "加入不重复的词数量为：" + str(count_unique))
    # 返回本地集合
    return unique_words


def run_threads(content, all_word_set):
    # 线程数，根据 CPU 核心数调整
    num_threads = os.cpu_count()
    logging.info("当前线程数：" + str(num_threads))
    # 将分词结果分成多个部分
    segments = [content[i::num_threads] for i in range(num_threads)]
    # 使用多线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(add_word_to_all_word_set, segment) for segment in segments]

        for future in concurrent.futures.as_completed(futures):
            # 合并结果
            all_word_set.update(future.result())


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Word2Vec
word_seg_cut_folder_path = os.path.join("data", "txt", "seg_cut")
# 读取所有分词文件，并按照创建时间排序（保持数据量递增的顺序）
seg_cut_files = os.listdir(word_seg_cut_folder_path)
seg_cut_files.sort(key=lambda file: os.path.getmtime(os.path.join(word_seg_cut_folder_path, file)))
# 遍历所有的分词文件，分别运行测试
for seg_cut_file in seg_cut_files:
    file_path = os.path.join(word_seg_cut_folder_path, seg_cut_file)
    # 集合存储所有的词
    all_word_set = set()
    logging.info("开始读取文件：" + file_path)
    with open(file_path, "r", encoding="utf-8") as seg_content_file:
        seg_content = seg_content_file.readlines()
        # 使用多线程，提速十分明显，从可能需要半小时提升到只需要十几秒
        run_threads(seg_content, all_word_set)
    logging.info("读取分词数据完毕，共计 " + str(len(all_word_set)) + " 个不重复词")

    all_words = list(all_word_set)

    # 超参数
    learning_rate = 0.001
    # 如果有GPU使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("当前设备的 cuda为：" + str(device))
    # 上下文信息，涉及文本的前n个及后n个即窗口大小
    context_size = 2
    # 词嵌入的维度，一个单词用多少个数据表示
    embedding_dimension = 100
    # 训练轮数
    epoch = 10

    all_words_set = set(all_words)
    all_words_set_size = len(all_words_set)

    word_to_index = {word: index for index, word in enumerate(all_words_set)}
    index_to_word = {index: word for index, word in enumerate(all_words_set)}

    # 直接构造 CBOW 的词表，如 [w1, w2, w4, w5], 若 w3 为目标词
    cbow_data = []
    for i in range(2, len(all_words) - 2):
        # 目标词
        target = all_words[i]
        # 上下文词
        context = [all_words[i - 2], all_words[i - 1], all_words[i + 1], all_words[i + 2]]
        cbow_data.append((context, target))
        if i % 50000 == 0:
            logging.info("正在构造第" + str(i) + "个目标词的 CBOW 词表：" + target + " 上下文词：" + str(context))
    logging.info("CBOW 数据构造完毕，共 " + str(len(cbow_data)) + " 条数据")

    # 模型训练，匹配 CPU 或 GPU
    model = MyCBOW(all_words_set_size, embedding_dimension).to(device)
    # 优化器, 随机梯度下降方法，每次参数更新时，根据梯度大小以0.001的比例调整参数值, 来最小化损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 存储损失函数的集合
    losses = []
    # 损失函数: 使用负对数似然损失
    loss_function = nn.NLLLoss()
    # 使用 trange 可以实时显示进度条
    for epoch in trange(epoch):
        total_loss = 0
        # tqdm 同理，可以显示实施进度条
        for context, target in tqdm(cbow_data):
            # 把训练集的上下文和标签都放到 GPU 或 CPU 中
            context_vector = generate_context_vector(context, word_to_index).to(device)
            target = torch.tensor([word_to_index[target]]).to(device)
            # 梯度清零
            model.zero_grad()
            # 开始前向传播
            train_predict = model(context_vector).to(device)
            loss = loss_function(train_predict, target)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print("losses-=", losses)
    break
