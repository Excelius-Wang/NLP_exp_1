"""
  @FileName：word_segment.py
  @Author：Excelius
  @CreateTime：2024/9/22 09:56
  @Company: None
  @Description：利用 jieba 进行分词
"""
import logging
import os

import jieba


# 加载停用词词表，使用的是哈工大停用词词表
def load_stop_words(path):
    """
    加载停用词词表，使用停用词可以提高效率
    @ stop_words_path: 停用词路径
    @return: 停用词词表的列表
    """
    with open(path, 'r', encoding='utf-8') as stop_words_file:
        logging.info("加载停用词词表...")
        return stop_words_file.read().split('\n')


# 设置日志格式
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 设置输出缓冲区
wiki_seg_file = os.path.join('data', 'txt', 'wiki_seg.txt')
out_put = open(wiki_seg_file, 'w', encoding='utf-8')

wiki_text_file = os.path.join('data', 'txt', 'pre_pro_wiki_text.txt')
stop_words_path = os.path.join('data', 'stopwords', 'hit_stopwords.txt')
stop_words = load_stop_words(stop_words_path)
logging.info("开始分词...")
with open(wiki_text_file, 'r', encoding='utf-8') as wiki_text:
    for count, line in enumerate(wiki_text):
        # 启动paddle模式, 延迟加载。 0.40版之后开始支持
        # jieba.enable_paddle()
        seg_list = []
        # 开始分词，使用的是精确模式
        seg_list = jieba.cut(line, cut_all=False)
        # 去掉空格和空字符串
        strip_seg_list = [word for word in seg_list if word.strip()]
        # 去掉停用词
        strip_seg_list = [word for word in strip_seg_list if word not in stop_words]
        # 把生成的数据保存到 out_put 中
        for word in strip_seg_list:
            out_put.write(word + ' ')
        out_put.write('\n')
        if count % 100 == 0 and count > 0:
            logging.info("已经处理的分词数据：" + str(count))
    logging.info("分词结束, 处理的数据总量为：" + str(count))
