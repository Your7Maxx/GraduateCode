from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

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
#data[:, 3] = le_process.fit_transform(data[:, 3])

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(data)

# joblib.dump(clf, 'if_model.pkl')

predictions = clf.predict(data)
for i, pred in enumerate(predictions):
    if pred == -1:
        print(f"进程链 {i+1} 可能是入侵")
        print(data_tmp[i])
    else:
        print(f"进程链 {i+1} 正常")

anomaly_scores = clf.decision_function(data)  # [-1,1]

for i, score in enumerate(anomaly_scores):
    print(f"进程链 {i+1} 的异常度分数为: {score:.4f}")
