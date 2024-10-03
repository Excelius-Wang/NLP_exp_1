"""
  @FileName：word_seg_cut_by_50000.py
  @Author：Excelius
  @CreateTime：2024/9/24 20:40
  @Company: None
  @Description：把处理好的数据，按照每 4W行 进行切片，方便进行测试
"""
import logging
import os.path


def write_wiki_seg_cut_by_4w(folder_path, line_count, wiki_seg_txt_list):
    """
    将切分的数据写入文件
    @param folder_path: 文件夹路径
    @param line_count: 行数
    @param wiki_seg_txt_list: wiki 语料库的数据
    @return: None
    """
    # 切分后的文件路径
    wiki_seg_cut_path = os.path.join(folder_path, 'wiki_seg_cut_' + str(line_count) + '.txt')
    out_put = open(wiki_seg_cut_path, 'w', encoding='utf-8')
    logging.info('已写入文件：' + wiki_seg_cut_path + '， 当前写入的数据量为：' + str(line_count))
    for text_line in wiki_seg_txt_list:
        out_put.write(text_line)
    out_put.close()


# 设置日志格式
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word_seg_txt_path = os.path.join('data', 'txt', 'wiki_seg.txt')

# 设置输出文件夹
wiki_seg_folder = os.path.join('data', 'txt', 'seg_cut')
# 文件夹不存在则创建文件夹
if not os.path.exists(wiki_seg_folder):
    os.makedirs(wiki_seg_folder)
    logging.info('已创建文件夹：' + wiki_seg_folder)
else:
    logging.info('已存在文件夹：' + wiki_seg_folder)

# 读取语料库
wiki_seg_file = os.path.join('data', 'txt', 'wiki_seg.txt')
wiki_seg_cut_list = []
# 读取原segment txt 文件，进行切分
with open(wiki_seg_file, 'r', encoding='utf-8') as wiki_seg_contents:
    for count, line in enumerate(wiki_seg_contents):
        # 每满 4W 切一次
        if count % 40000 == 0 and count > 0:
            write_wiki_seg_cut_by_4w(wiki_seg_folder, count, wiki_seg_cut_list)
        wiki_seg_cut_list.append(line)
    # 切分最后不满 4W 的情况
    write_wiki_seg_cut_by_4w(wiki_seg_folder, count, wiki_seg_cut_list)
logging.info('语料库切分完毕')