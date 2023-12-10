import openai
import requests

# 代理服务器的地址和端口
proxy_url = "http://192.168.1.100:7890"


# 设置代理服务器
proxies = {
    "http": proxy_url,
    "https": proxy_url,
}

# 设置OpenAI API密钥
api_key = "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"

# 请求数据
data = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello ChatGPT!"}]
}

# 使用代理服务器发送请求，并设置OpenAI API密钥
response = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=data, proxies=proxies)

if response.status_code == 200:
    completion = response.json()
    print(completion['choices'][0]['message']['content'])
else:
    print(f"请求失败：{response.status_code} - {response.text}")

