# spider
# data1:与主题相关的文档
# data2：与主题不相关的文档
# data3：待判别相关性的文档
# 分段提取.py：爬虫程序，可爬取新浪新闻，爬取的文章以txt格式存储在news_articles中
# 复制.py：将news_articles中的文件复制到data3中，待判别
# 改进判别器v1.py：通过识别data1的文档归纳出主题，并判别data3的文档与该主题的相关性
# 判别器原理：循环神经网络，合并了tf-idf和skipgram的特征
