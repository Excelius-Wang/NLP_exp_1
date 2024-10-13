"""
  @FileName：verification.py
  @Author：Excelius
  @CreateTime：2024/10/9 14:32
  @Company: None
  @Description：
"""
import logging
import pickle
import numpy as np
from importlib import reload
from sklearn.metrics.pairwise import cosine_similarity

"""----------logging配置----------"""
# 使用 logging.info 打印信息，colab 需要 reload() 函数，否则无法打印
reload(logging)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 获取给定词的最相近的词
def find_most_similar(word_vector, _embeddings, top_n=5):
    similarities = []
    word_vector_reshaped = word_vector.reshape(1, -1)
    for i in range(len(_embeddings)):
        vector_2_reshaped = _embeddings[i].reshape(1, -1)
        sim = cosine_similarity(word_vector_reshaped, vector_2_reshaped)
        similarities.append((reverse_dictionary[i], sim))
        if i > 50000:
            break

        # 按相似度排序，取前n个
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


file_path_list = ['data/embedding_cbow_80w.pkl', 'data/embedding_skip_gram_80w.pkl']
# 加载 .pkl 文件
for file_path in file_path_list:
    with open(file_path, 'rb') as f:
        # embeddings 为列表，reverse_dictionary 为字典，格式为{索引：单词}
        embeddings, embeddings_2, two_d_embeddings, two_d_embeddings_2, reverse_dictionary = pickle.load(f)
        logging.info("读取本地模型" + file_path + "完成...")

    word_index_dict = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
    # 选择十个单词
    valid_word = ['电脑', '书籍', '公里', '还']
    # valid_word = ['父亲', '中国', '电脑', '手机', '书籍', '公里', '还', '部分', '年', '之后']
    # 完善验证单词的索引
    valid_word_index = []
    # 如果待验证词在此表中，那么直接获得索引，否则指定为 unknown
    for word in valid_word:
        if word_index_dict.get(word, None) is not None:
            valid_word_index.append(word_index_dict[word])
        else:
            valid_word_index.append(0)

    for word in valid_word_index:

        out_str = '当前词的索引为：' + str(word) + '，词为：' + str(reverse_dictionary[word]) + ', 词向量均值为：' + str(
            np.mean(embeddings[word])) + ', 最相近的前十个词为：\n'
        similar_words = find_most_similar(embeddings[word], embeddings, 11)

        for sim_word in similar_words:
            out_str += str(sim_word[0]) + ', 词向量均值为：' + str(
                np.mean(embeddings[word_index_dict[sim_word[0]]])) + ', 相似度为：' + str(
                sim_word[1][0][0]) + '\n'
        logging.info(out_str)
        # break

    # 类比实验
    logging.info('\n')
    word_vec_father = embeddings[word_index_dict['父亲']]
    logging.info('父亲 的词向量均值为：' + str(np.mean(word_vec_father)) + ', 索引为：' + str(word_index_dict['父亲']))
    word_vec_man = embeddings[word_index_dict['男人']]
    logging.info('男人 词向量均值为：' + str(np.mean(word_vec_man)) + ', 索引为：' + str(word_index_dict['男人']))
    word_vec_woman = embeddings[word_index_dict['女人']]
    logging.info('女人 词向量均值为：' + str(np.mean(word_vec_woman)) + ', 索引为：' + str(word_index_dict['女人']))
    word_vec_sub = word_vec_father - word_vec_man + word_vec_woman
    word_vec_res = find_most_similar(word_vec_sub, embeddings, 11)
    logging.info('父亲-男人+女人 之间的类比实验，最相近的前十个词为:')
    for word in word_vec_res:
        logging.info(
            word[0] + ', 词向量为：' + str(embeddings[word_index_dict[word[0]]][0]) + ', 相似度为：' + str(word[1][0][0]))

    logging.info('\n')
    word_vec_king = embeddings[word_index_dict['国王']]
    logging.info('国王 的词向量均值为：' + str(np.mean(word_vec_king)) + ', 索引为：' + str(word_index_dict['国王']))
    logging.info('男人 词向量均值为：' + str(np.mean(word_vec_man)) + ', 索引为：' + str(word_index_dict['男人']))
    logging.info('女人 词向量均值为：' + str(np.mean(word_vec_woman)) + ', 索引为：' + str(word_index_dict['女人']))
    word_vec_queen = embeddings[word_index_dict['女王']]
    logging.info('女王 词向量均值为：' + str(np.mean(word_vec_woman)) + ', 索引为：' + str(word_index_dict['女王']))
    word_vec_sub = word_vec_king - word_vec_man + word_vec_woman
    word_vec_res = find_most_similar(word_vec_sub, embeddings, 11)
    logging.info('国王-男人+女人 之间的类比实验，最相近的前十个词为:')
    for word in word_vec_res:
        logging.info(
            word[0] + ', 词向量均值为：' + str(np.mean(embeddings[word_index_dict[word[0]]])) + ', 相似度为：' + str(
                word[1][0][0]))

    logging.info('\n')
    word_vec_pin_pai = embeddings[word_index_dict['乒乓球拍']]
    logging.info(
        '乒乓球拍 的词向量均值为：' + str(np.mean(word_vec_pin_pai)) + ', 索引为：' + str(word_index_dict['乒乓球拍']))
    word_vec_pin = embeddings[word_index_dict['乒乓球']]
    logging.info('乒乓球 词向量均值为：' + str(np.mean(word_vec_pin)) + ', 索引为：' + str(word_index_dict['乒乓球']))
    word_vec_yu = embeddings[word_index_dict['羽毛球拍']]
    logging.info('羽毛球拍 词向量均值为：' + str(np.mean(word_vec_yu)) + ', 索引为：' + str(word_index_dict['羽毛球拍']))
    word_vec_yu_qiu = embeddings[word_index_dict['羽毛球']]
    logging.info(
        '羽毛球拍 词向量均值为：' + str(np.mean(word_vec_yu_qiu)) + ', 索引为：' + str(word_index_dict['羽毛球']))
    word_vec_sub = word_vec_pin_pai - word_vec_pin + word_vec_yu_qiu
    word_vec_res = find_most_similar(word_vec_sub, embeddings, 11)
    logging.info('乒乓球拍-乒乓球+羽毛球拍 之间的类比实验，最相近的前十个词为:')
    for word in word_vec_res:
        logging.info(
            word[0] + ', 词向量为：' + str(np.mean(embeddings[word_index_dict[word[0]]])) + ', 相似度为：' + str(
                word[1][0][0]))

    logging.info('\n')
    word_vec_uk = embeddings[word_index_dict['英国']]
    logging.info('英国 的词向量均值为：' + str(np.mean(word_vec_uk)) + ', 索引为：' + str(word_index_dict['英国']))
    word_vec_lon = embeddings[word_index_dict['伦敦']]
    logging.info('伦敦 词向量均值为：' + str(np.mean(word_vec_lon)) + ', 索引为：' + str(word_index_dict['伦敦']))
    word_vec_usa = embeddings[word_index_dict['美国']]
    logging.info('美国 词向量均值为：' + str(np.mean(word_vec_usa)) + ', 索引为：' + str(word_index_dict['美国']))
    word_vec_hua = embeddings[word_index_dict['华盛顿']]
    logging.info('华盛顿 词向量均值为：' + str(np.mean(word_vec_hua)) + ', 索引为：' + str(word_index_dict['华盛顿']))
    word_vec_sub = word_vec_uk - word_vec_lon + word_vec_hua
    word_vec_res = find_most_similar(word_vec_sub, embeddings, 11)
    logging.info('英国-伦敦+美国 之间的类比实验，最相近的前十个词为:')
    for word in word_vec_res:
        logging.info(
            word[0] + ', 词向量为：' + str(np.mean(embeddings[word_index_dict[word[0]]])) + ', 相似度为：' + str(
                word[1][0][0]))
