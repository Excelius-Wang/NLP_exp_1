"""
  @FileName：visualize.py
  @Author：Excelius
  @CreateTime：2024/10/10 09:27
  @Company: None
  @Description：
"""
import logging
from importlib import reload

# -*- coding: utf-8 -*-
from matplotlib import pylab as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages

"""----------logging配置----------"""
# 使用 logging.info 打印信息，colab 需要 reload() 函数，否则无法打印
reload(logging)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def plot(embeddings, labels, save_to_pdf='embed.pdf'):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pp = PdfPages(save_to_pdf)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['font.weight'] = 'light'  # 全局加粗字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.figure(figsize=(30, 30))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                     ha='right', va='bottom')
    plt.savefig(pp, format='pdf')
    plt.show()
    pp.close()


methods = ['cbow', 'skip_gram']
for method in methods:
    filename = 'data/embedding_%s_80w.pkl' % method
    with open(filename, 'rb') as f:
        [embeddings, embeddings_2, two_d_embeddings, two_d_embeddings_2, reverse_dictionary] = pickle.load(f)

    num_points = 1500
    words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
    logging.info(method + '保存文件中...')
    plot(two_d_embeddings, words, save_to_pdf='embeddings_%s_80w.pdf' % method)
    plot(two_d_embeddings_2, words, save_to_pdf='embeddings_2_%s_80w.pdf' % method)
