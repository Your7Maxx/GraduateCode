from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

app = Flask(__name__)

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

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(data)

# joblib.dump(clf, 'if_model.pkl')

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
