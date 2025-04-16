import os
import jieba
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import classification_report
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
import pickle
from collections import Counter


# 1. 增强的预处理函数
def preprocess_text(text, language='chinese'):
    """文本预处理流程"""
    # 中文处理
    if language == 'chinese':
        text = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff' or char.isalnum()])
        words = jieba.lcut(text)
    # 英文处理
    else:
        words = simple_preprocess(text)
    return ' '.join(words)


# 2. 改进的数据加载
def load_and_preprocess_documents(directory, language='chinese'):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read().strip()
                processed = preprocess_text(text, language)
                if processed:  # 过滤空文档
                    documents.append(processed)
    return documents


# 3. 智能数据分割
def safe_data_split(features, labels, test_size=0.2):
    """自动适应不同数据量的分割方法"""
    n_samples = len(labels)
    n_classes = len(np.unique(labels))

    # 极小数据集使用留一法
    if n_samples < 10 or n_samples * test_size < n_classes:
        print(f"警告: 数据量不足(总样本={n_samples}, 类别={n_classes})，使用留一法验证")
        loo = LeaveOneOut()
        for train_idx, val_idx in loo.split(features):
            return features[train_idx], features[val_idx], labels[train_idx], labels[val_idx]
    else:
        return train_test_split(
            features, labels,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )


# 4. 改进的Word2Vec特征提取
class EnhancedTextVectorizer:
    def __init__(self, w2v_size=300, w2v_window=5, w2v_min_count=3):
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.w2v_min_count = w2v_min_count
        self.word_vectors = None
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=2000)

    def tokenize(self, docs):
        return [doc.split() for doc in docs]

    def train_w2v(self, tokenized_docs):
        print("训练Word2Vec模型...")
        self.word_vectors = Word2Vec(
            tokenized_docs,
            vector_size=self.w2v_size,
            window=self.w2v_window,
            min_count=self.w2v_min_count,
            workers=4,
            epochs=10
        ).wv

    def get_w2v_features(self, docs):
        tokenized = self.tokenize(docs)
        features = []
        for tokens in tokenized:
            vecs = [self.word_vectors[token] for token in tokens if token in self.word_vectors]
            features.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.w2v_size))
        return np.array(features)

    def fit(self, docs):
        # TF-IDF训练
        tfidf_features = self.tfidf.fit_transform(docs).toarray()

        # Word2Vec训练
        tokenized_docs = self.tokenize(docs)
        self.train_w2v(tokenized_docs)
        w2v_features = self.get_w2v_features(docs)

        # 合并特征并标准化
        combined = np.hstack((tfidf_features, w2v_features))
        self.scaler.fit(combined)
        return combined

    def transform(self, docs):
        tfidf_features = self.tfidf.transform(docs).toarray()
        w2v_features = self.get_w2v_features(docs)
        combined = np.hstack((tfidf_features, w2v_features))
        return self.scaler.transform(combined)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.word_vectors.save(os.path.join(path, 'word_vectors.kv'))
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(path, 'tfidf.pkl'), 'wb') as f:
            pickle.dump(self.tfidf, f)

    def load(self, path):
        self.word_vectors = KeyedVectors.load(os.path.join(path, 'word_vectors.kv'))
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(path, 'tfidf.pkl'), 'rb') as f:
            self.tfidf = pickle.load(f)


# 5. 改进的神经网络模型
def build_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.6),

        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dropout(0.4),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


# 6. 可视化训练过程
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练集准确率')
    plt.plot(history.history['val_accuracy'], label='验证集准确率')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练集损失')
    plt.plot(history.history['val_loss'], label='验证集损失')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# 主流程
def main():
    # 配置参数
    DATA1_PATH = "./data1"  # 主题相关文本
    DATA2_PATH = "./data2"  # 非主题相关文本
    DATA3_PATH = "./data3"  # 待判别文本
    MODEL_PATH = "./model.keras"  # 注意使用.keras扩展名
    VECTORIZER_PATH = "./vectorizer"

    # 1. 数据加载与预处理
    print("加载和预处理数据...")
    pos_docs = load_and_preprocess_documents(DATA1_PATH)
    neg_docs = load_and_preprocess_documents(DATA2_PATH)
    new_docs = load_and_preprocess_documents(DATA3_PATH)

    documents = pos_docs + neg_docs
    labels = np.array([1] * len(pos_docs) + [0] * len(neg_docs))

    # 检查数据分布
    print("类别分布:", dict(zip(*np.unique(labels, return_counts=True))))

    # 2. 特征工程
    vectorizer = EnhancedTextVectorizer()

    if os.path.exists(VECTORIZER_PATH):
        print("加载已有特征提取器...")
        vectorizer.load(VECTORIZER_PATH)
        features = vectorizer.transform(documents)
    else:
        print("训练新特征提取器...")
        features = vectorizer.fit(documents)
        vectorizer.save(VECTORIZER_PATH)

    # 3. 智能数据分割
    X_train, X_val, y_train, y_val = safe_data_split(features, labels)

    # 4. 模型训练
    if os.path.exists(MODEL_PATH):
        print("加载已有模型...")
        model = load_model(MODEL_PATH)
    else:
        print("训练新模型...")
        model = build_model(features.shape[1])

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, save_best_only=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            class_weight={1: len(neg_docs) / len(pos_docs), 0: 1.0}  # 处理类别不平衡
        )

        plot_history(history)

        # 评估模型
        print("\n验证集评估:")
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        print(classification_report(y_val, y_pred))

    # 5. 对新数据进行预测
    if new_docs:
        print("\n预测新文档...")
        new_features = vectorizer.transform(new_docs)
        predictions = model.predict(new_features)

        print("\n预测结果:")
        for i, (doc, pred) in enumerate(zip(new_docs, predictions)):
            prob = pred[0]
            label = "相关" if prob > 0.5 else "不相关"
            print(f"文档{i + 1}: {label} (置信度: {prob:.2%})")
            print(f"内容预览: {doc[:50]}...\n")


if __name__ == "__main__":
    # 初始化jieba分词器
    jieba.initialize()
    main()