from flask import Flask, request, jsonify
import pickle
from xgboost import XGBClassifier
from slpp import ShellTokenizer
from sklearn.feature_extraction.text import HashingVectorizer

app = Flask(__name__)

# 加载已保存的XGBoost模型
model_path = 'hashPlus_xgb_model.pkl'
with open(model_path, 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# 创建ShellTokenizer
shell_tokenizer = ShellTokenizer(verbose=False, debug=False)

@app.route('/predict', methods=['POST'])
def predict():
    # 获取请求中的参数 new_command
    new_command = request.json.get('new_command', '')

    # 使用HashingVectorizer获取Hashing特征向量
    hv = HashingVectorizer(
        lowercase=False,
        tokenizer=shell_tokenizer.tokenize_command,
        token_pattern=None,
        n_features=500
    )
    X_hashing = hv.transform([new_command])

    # 获取类别概率
    class_probabilities = xgb_model.predict_proba(X_hashing)
    # 对特征向量进行预测
    predictions = xgb_model.predict(X_hashing)

    return jsonify({
        'Predictions': predictions.tolist(),
        'Class Probabilities': class_probabilities.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
