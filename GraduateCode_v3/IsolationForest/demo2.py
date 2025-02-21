from sklearn.preprocessing import LabelEncoder
import numpy as np
import random

class CustomIsolationForest:
    def __init__(self, n_trees, max_depth):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, feature_weights):
        n_samples, n_features = X.shape
        self.feature_weights = feature_weights
        for _ in range(self.n_trees):
            tree = self._build_tree(X, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X, depth):
        if depth >= self.max_depth or len(X) <= 1:
            return LeafNode(X)

        # 在选择分裂特征时考虑特征权重
        feature_idx = self._choose_split_feature(X)
        split_value = np.random.uniform(X[:, feature_idx].min(), X[:, feature_idx].max())
        left_mask = X[:, feature_idx] < split_value
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], depth + 1)

        return InternalNode(feature_idx, split_value, left_subtree, right_subtree)

    def _choose_split_feature(self, X):
        # 基于特征权重随机选择分裂特征
        weighted_features = random.choices(range(X.shape[1]), k=X.shape[1], weights=self.feature_weights)
        return weighted_features[0]

    def anomaly_score(self, X):
        scores = np.zeros(X.shape[0])
        for tree in self.trees:
            scores += tree.anomaly_score(X)
        scores /= self.n_trees
        return scores

# 其余的类和方法保持不变

if __name__ == "__main__":
    # 从文件读取数据
    data = []
    with open("/root/commandDetect/kg/command_data.test", "r") as file:
        for line in file:
            parts = line.strip().split('\"')
            if len(parts) >= 4:
                timestamp, pcomm, ppcomm, ppid = parts[1], parts[3], parts[7], parts[9]
                data.append([timestamp, ppcomm, pcomm])

    data_tmp = data
    data = np.array(data)

    # 提取时间并编码
    timestamps = [np.datetime64(item[0]) for item in data]
    le_timestamp = LabelEncoder()
    data[:, 0] = le_timestamp.fit_transform(timestamps)

    # 将进程名编码为整数
    le_process = LabelEncoder()
    data[:, 1] = le_process.fit_transform(data[:, 1])
    data[:, 2] = le_process.fit_transform(data[:, 2])

    # 定义特征权重，这里简单地设置了三个特征的不同权重
    feature_weights = [0.5, 0.3, 0.2]

    # 初始化并训练自定义 IF 模型
    custom_if = CustomIsolationForest(n_trees=10, max_depth=5)
    custom_if.fit(data, feature_weights)

    # 计算异常得分
    anomaly_scores = custom_if.anomaly_score(data)

    # 打印异常得分
    for i, score in enumerate(anomaly_scores):
        if score < 0:
            print(f"进程链 {i+1} 可能是入侵")
            print(data_tmp[i])
        else:
            print(f"进程链 {i+1} 正常")
