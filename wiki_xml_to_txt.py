"""
  @FileName：wiki_xml_to_txt.py
  @Author：Excelius
  @CreateTime：2024/9/22 10:27
  @Company: None
  @Description：
"""
import sys
import logging  # 打印日志包
from gensim.corpora import WikiCorpus  # 导入gensim库中用于处理维基百科语料库的WikiCorpus类


def main():
    """
    该函数主要在命令行中运行，需要读取命令行参数，接收的参数为：执行的py文件名 原始的bz2语料
    @return: Nones
    """
    # 如果命令参数有误，给予提示
    if len(sys.argv) != 2:
        print("命令输入有误！命令格式: python 执行的python文件名 wiki预料库路径")
        exit()

    # 设置日志格式
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 读取只读、流式、内存高效的wiki语料
    wiki_corpus = WikiCorpus(sys.argv[1], dictionary={})
    # 语料库统计
    text_num = 0

    # 创建获取完毕的语料库文件 wiki_text.txt, 读取数据并保存到该文件中
    file_path = 'data/txt/wiki_text.txt'
    logging.info('==>开始处理文件...')
    with open(file_path, 'w', encoding='utf-8') as out_put:
        # WikiCorpus.get_texts() 方法遍历语料库获取 token(模型输入单元) 列表
        for text in wiki_corpus.get_texts():
            # 将分词后的文本写入到文件中，每行一个文本
            out_put.write(' '.join(text) + '\n')
            text_num += 1
            if text_num % 10000 == 0:
                logging.info('已处理 ' + str(text_num) + ' 文本')

        logging.info('处理的文本总数为：' + str(text_num))


if __name__ == '__main__':
    main()
