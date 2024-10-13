

# 北京交通大学NLP课程词向量实验

北京交通大学自然语言处理课程词向量实验，不调包实现 Word2Vec，实验所需的语料库为[维基百科中文语料库](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)。

本次实验基于 TensorFlow V1 版本编码实现，参考了[《TensorFlow 实战》](https://book.douban.com/subject/26974266/)书本以及其他相关内容教程和公开代码。具体可见我的个人博客：[NLP 第一次实验 - Ther's World (excelius.xyz)](https://www.excelius.xyz/nlp第一次实验/)（待完善！）

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


## 目录

- [代码运行环境](#代码运行环境)
  * [Python 版本](#Python版本：)
  * [所需相关库](#所需相关库：)
  * [安装步骤](#安装步骤)
- [bz2 文件转 txt 文件](#bz2文件转txt文件[)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [语料库预处理](#语料库预处理)
- [分词](#分词)
- [模型训练](#模型训练)
- [模型验证](#模型验证)
- [词向量二维展示](#词向量二维展示)
- [作者](#作者)

### 代码运行环境

#### Python版本：

​	Python 3.9.19

#### 所需相关库：

- gensim==4.3.3
- jieba==0.42.1
- matplotlib==3.9.2
- numpy==2.1.2
- OpenCC==1.1.9
- scikit_learn==1.5.2
- tensorflow==2.17.0
- tensorflow_intel==2.17.0

​	已导出`requirement.txt`文件，可以使用`pip install -r requirements.txt`或`conda install --yes --file requirements.txt`命令一键安装相关的库；或安装相关库。

​	建议使用 anaconda 管理 Python 环境，创建本项目的环境安装好相应的库再运行以下步骤。

#### 安装步骤

1. 配置实验运行环境
2. 克隆项目

```sh
git clone https://github.com/Excelius-Wang/NLP_exp_1.git
```

### bz2文件转txt文件

​	在本项目的路径下打开命令行，激活已配置好的 Python 环境，使用命令：

```sh
python wiki_xml_to_txt.py zhwiki_xml.bz2
```

​	其中 zhwiki_xml.bz2  为本地下载完成的 wiki 语料库文件。

​	生成 wiki_text.txt 文件位于`项目根目录/data/txt/wiki_text.txt`。

### 语料库预处理

​	当 bz2 文件转换为 txt 文件后，运行下一步的语料库预处理文件，使用以下命令：

```sh
python text_pre_progress.py
```

​	该命令会自动读取上一步中的`data/txt/wiki_text.txt`文件，生成`data/txt/pre_simp_wiki_text.txt`文件。

### 分词

​	完成文件预处理操作后，使用以下命令完成分词操作：

```bash
python word_segment.py
```

​	该命令会读取上一步生成的`data/txt/pre_simp_wiki_text.txt`文件，进一步生生成`data/txt/wiki_seg.txt`。

### 模型训练

​	模型训练可以尝试在本机也可以尝试在 Google Colab 上运行，若在 Colab 上运行建议使用 TPU，否则可能爆显存。

​	这里建议使用 Pycharm 打开项目（需要配置 Python 解释器即先前创建的 Python 环境），若在本机完成，打开`CBOW_TF.py`或`Skip-Gram.py`文件，直接 右键文件空白处 -> 运行 即可。若本机训练完成，对应的`pkl`文件位于`data`文件夹中。

​	如果使用 Colab，需要挂在自己的 Google 云盘，在 Google 云盘里创建`data/txt/`文件夹，将`wiki_seg.txt`上传至内，或者修改代码里的文件路径。代码运行可以直接上传相应的 .ipynb 文件并运行，也可以复制相应的代码，复制到一个空的代码行，再运行即可。Colab 中使用 TPU V2-8 的运行时间在 40min（CBOW）与 1.5h 左右（Skip-Gram），消耗大概在 15-30 之间。对应的`pkl`文件保存在 Google 云盘的 `data`文件夹中。

### 模型验证

​	首先需要把模型训练完成后的`embedding_cbow_80w.pkl`与`embedding_skip_gram_80w.pkl`文件放在`data`文件夹中。然后在 Pycharm 中打开`verification.py`，右键 -> 运行即可。

### 词向量二维展示

​	同样需要把模型训练完成后的`embedding_cbow_80w.pkl`与`embedding_skip_gram_80w.pkl`文件放在`data`文件夹中。然后在 Pycharm 中打开`visualize.py`，右键 -> 运行即可。会生成 4 个本地 pdf 文件供更加方便浏览，也可以在对应的 plot 绘图窗口查看。

### 作者

[Ther's World (excelius.xyz)](https://www.excelius.xyz/)

邮箱：excelius@qq.com

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/Excelius-Wang/NLP_exp_1 /blob/master/LICENSE.txt)

<!-- links -->

[your-project-path]:Excelius-Wang/NLP_exp_1
[contributors-shield]: https://img.shields.io/github/contributors/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[contributors-url]: https://github.com/Excelius-Wang/NLP_exp_1/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[forks-url]: https://github.com/Excelius-Wang/NLP_exp_1/network/members
[stars-shield]: https://img.shields.io/github/stars/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[stars-url]: https://github.com/Excelius-Wang/NLP_exp_1/stargazers
[issues-shield]: https://img.shields.io/github/issues/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/Excelius-Wang/NLP_exp_1.svg
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/Excelius-Wang/NLP_exp_1/blob/master/LICENSE.txt
