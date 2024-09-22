"""
  @FileName：text_pre_progress.py
  @Author：Excelius
  @CreateTime：2024/9/22 10:27
  @Company: None
  @Description：需要对文本进行预处理，如剔除英文、简繁转换、剔除多余的空格和换行符
"""
import re
import logging

import opencc


def filter_text(text):
    """
    过滤掉非中文字符及多个连续的空格，保留中文字符、单个空格、换行符
    :param text: 输入文本
    :return: 过滤后的文本
    """
    # 保留：简体中文、空格、换行符，过滤掉其他内容
    pattern = r'[^\u4e00-\u9fff\s\n]'
    # 使用正则表达式将匹配到的非指定字符替换为空字符
    filtered_text = re.sub(pattern, '', text)
    # 使用正则表达式替换多个空格为单个空格
    filtered_text = re.sub(r'\s+', ' ', filtered_text)
    return filtered_text


# 设置日志格式
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
text_file = 'data/txt/wiki_text.txt'
pre_pro_wiki_text_file = 'data/txt/pre_pro_wiki_text.txt'
out_put = open(pre_pro_wiki_text_file, 'w', encoding='utf-8')
with open(text_file, 'r', encoding='utf-8') as wiki_text:
    # 遍历 wiki_text 里的每行数据，分别对其进行处理：
    # 1. 先将繁体中文转换为简体中文
    # 2. 将过滤后的简体中文文本除了简体中文字符、空格、换行符以外的所有字符，
    #    需要注意去掉了多个连续的空格只保留了一个空格
    count = 0
    logging.info('==>开始处理文本...')
    for line in wiki_text:
        # 先把繁体中文转换为简体中文，创建 OpenCC 对象，指定繁体转换为简体
        converter = opencc.OpenCC('t2s')  # 't2s' 表示繁体转简体
        # 将文本中的简体中文转换为简体中文
        simplified_text = converter.convert(line)
        # 过滤非中文字符，只保留简体中文字符、单个空格符以及换行符
        filter_simplified_text = filter_text(simplified_text)
        # 把处理后的文本写入到 out_put 中
        out_put.write(filter_simplified_text + '\n')
        count += 1
        if count % 100 == 0:
            logging.info('==>已经处理了 ' + str(count) + ' 数据...')
    logging.info('==>处理完成，共计处理 ' + str(count) + ' 条数据。')