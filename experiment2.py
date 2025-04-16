import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Step 1: 从文件夹中读取文本内容
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # 确保只加载 .txt 文件
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read().strip())
    return documents

# 文件夹路径设置
data1_directory = "./data1"  # 主题相关文本文件夹路径
data2_directory = "./data2"  # 非主题相关文本文件夹路径
data3_directory = "./data3"  # 待判别文本文件夹路径

# 加载数据
positive_documents = load_documents_from_directory(data1_directory)  # 主题相关
negative_documents = load_documents_from_directory(data2_directory)  # 非主题相关
new_documents = load_documents_from_directory(data3_directory)       # 待判别文本

# 合并数据并生成标签
documents = positive_documents + negative_documents
labels = [1] * len(positive_documents) + [0] * len(negative_documents)  # 1: 主题相关, 0: 非主题相关

# Step 2: TF-IDF特征提取
tfidf = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf.fit_transform(documents).toarray()

# Step 3: Word2Vec模型训练
tokenized_docs = [doc.split() for doc in documents]  # 分词
word2vec = Word2Vec(tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = word2vec.wv

# 将文档转换为Word2Vec特征
def get_word2vec_features(docs, word_vectors):
    features = []
    for doc in docs:
        tokens = doc.split()
        vecs = [word_vectors[token] for token in tokens if token in word_vectors]
        if vecs:
            features.append(np.mean(vecs, axis=0))  # 平均词向量
        else:
            features.append(np.zeros(word_vectors.vector_size))  # 用零向量填充
    return np.array(features)

word2vec_features = get_word2vec_features(documents, word_vectors)

# 合并TF-IDF和Word2Vec特征
features = np.hstack((tfidf_features, word2vec_features))

# Step 4: 构建神经网络判别器
model = Sequential([
    Dense(128, activation='relu', input_shape=(features.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 二分类问题，输出概率
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: 模型训练
model.fit(features, np.array(labels), epochs=10, batch_size=2, verbose=1)

# Step 6: 对新文本进行判别
new_tfidf_features = tfidf.transform(new_documents).toarray()
new_word2vec_features = get_word2vec_features(new_documents, word_vectors)
new_features = np.hstack((new_tfidf_features, new_word2vec_features))

predictions = model.predict(new_features)
for i, doc in enumerate(new_documents):
    print(f"'{doc[:30]}...' 的主题相关性: {'相关' if predictions[i] > 0.5 else '不相关'}")
