"""
  @FileName：text_pre_progress.py
  @Author：Excelius
  @CreateTime：2024/9/22 10:27
  @Company: None
  @Description：需要对文本进行预处理，如剔除英文、简繁转换、剔除多余的空格和换行符
"""
import os
import re
import logging

import opencc


def filter_quote(text):
    """
    过滤掉正文中的 注释、参考、参考资料、参考文献、参考书目、脚注、扩展阅读、参见、延伸阅读、概述、研究书目、引用
    :param text: 输入文本
    :return: 过滤后的文本
    """
    # 定义要过滤的关键词
    keywords = [
        "注释", "参考", "参考资料", "参考文献", "参考书目",
        "脚注", "扩展阅读", "参见", "延伸阅读", "概述",
        "研究书目", "引用"
    ]
    # 创建正则表达式模式
    pattern = r'(' + '|'.join(keywords) + r').*'

    # 去除包含关键词及其后的内容
    cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)

    return cleaned_text.strip()


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


def main():
    # 设置日志格式
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    text_file = os.path.join(os.getcwd(), 'data', 'txt', 'wiki_text.txt')
    pre_pro_wiki_text_file = os.path.join(os.getcwd(), 'data', 'txt', 'pre_simp_wiki_text.txt')
    out_put = open(pre_pro_wiki_text_file, 'w', encoding='utf-8')
    with open(text_file, 'r', encoding='utf-8') as wiki_text:
        # 遍历 wiki_text 里的每行数据，分别对其进行处理：
        # 1. 先将繁体中文转换为简体中文
        # 2. 去除杂乱的引用内容
        # 3. 将过滤后的简体中文文本除了简体中文字符、空格、换行符以外的所有字符，
        #    需要注意去掉了多个连续的空格只保留了一个空格
        count = 0
        logging.info('==>开始处理文本...')
        for line in wiki_text:
            # 先把繁体中文转换为简体中文，创建 OpenCC 对象，指定繁体转换为简体
            converter = opencc.OpenCC('t2s')  # 't2s' 表示繁体转简体
            # 将文本中的简体中文转换为简体中文
            simplified_text = converter.convert(line)
            # 过滤引用内容
            # 过滤非中文字符，只保留简体中文字符、单个空格符以及换行符
            simplified_text = filter_text(filter_quote(simplified_text))
            # 把处理后的文本写入到 out_put 中
            out_put.write(simplified_text + '\n')
            count += 1
            if count % 500 == 0:
                logging.info('==>已经处理了 ' + str(count) + ' 数据...')
        logging.info('==>处理完成，共计处理 ' + str(count) + ' 条数据。')


if __name__ == '__main__':
    main()
