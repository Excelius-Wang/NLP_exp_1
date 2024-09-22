"""
  @FileName：word_segment.py
  @Author：Excelius
  @CreateTime：2024/9/22 09:56
  @Company: None
  @Description：利用 jieba 进行分词
"""
import logging

# 设置日志格式
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 设置输出缓冲区
wiki_seg_file = 'data/txt/wiki_seg.txt'
out_put = open(wiki_seg_file, 'w', encoding='utf-8')

wiki_text_file = 'data/txt/wiki_text.txt'
with open(wiki_text_file, 'r', encoding='utf-8') as wiki_text:
    logging.info(wiki_text)
    count = 0
    for line in wiki_text:
        if count < 10:
            count = count + 1
            print(line)
