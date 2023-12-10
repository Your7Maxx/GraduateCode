from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# 初始化时间编码器
le_timestamp = LabelEncoder()
# 初始化进程名编码器
le_process = LabelEncoder()
clf = None

# 准备重新加载数据的函数
def load_data():
    global data, clf, le_timestamp, le_process
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
    le_timestamp = LabelEncoder()
    timestamps = [item[0] for item in data]
    data[:, 0] = le_timestamp.fit_transform(timestamps)

    # 将进程名编码为整数
    le_process = LabelEncoder()
    le_process.classes_ = np.unique(data[:, 1])
    data[:, 1] = le_process.transform(data[:, 1])
    le_process.classes_ = np.unique(data[:, 2])
    data[:, 2] = le_process.transform(data[:, 2])

    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(data)



# 创建定时任务调度器
scheduler = BackgroundScheduler()
scheduler.add_job(load_data, 'interval', seconds=5)  # 每5秒重新加载数据
scheduler.start()

# 第一次加载数据
load_data()

@app.route('/anomaly_score', methods=['POST'])
def get_anomaly_score():
    try:
        data = request.json
        timestamp = data['timestamp']
        ppcomm = data['ppcomm']
        pcomm = data['pcomm']

        input_data = [timestamp, ppcomm, pcomm]
        input_data[0] = le_timestamp.transform([input_data[0]])[0]
        input_data[1] = le_process.transform([input_data[1]])[0]
        input_data[2] = le_process.transform([input_data[2]])[0]

        anomaly_score = clf.decision_function([input_data])[0]
        return jsonify({'anomaly_score': anomaly_score})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
