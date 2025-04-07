# coding=utf-8
import requests
import json

# 定义 API 的基本 URL
API_URL = "http://localhost:8080"

# 1. 上传本地文档
# 注意：将 "path/to/your/document.txt" 替换为你的实际文件路径
# files = {
#     "files": [
#         ("document.txt", open("path/to/your/document.txt", "rb")),
#         ("document2.pdf", open("path/to/your/document2.pdf", "rb")),  # 可以上传多个文件
#     ]
# }
# response = requests.post(f"{API_URL}/uploadfiles/", files=files)
# print(response.json())
# print()

# 2. 查看历史对话
response = requests.get(f"{API_URL}/history/")
print(response.json())
print()

# 3. 发送简单对话
#headers = {"Content-Type": "application/json"}
#data = {"query": "你好"}
#response = requests.post(f"{API_URL}/chat/", headers=headers, data=json.dumps(data))
#print(response.json())
#print()

# 4. 发送复杂对话
# headers = {"Content-Type": "application/json"}
# data = {"query": "请总结一下上传的文档内容"}
# response = requests.post(f"{API_URL}/chat/", headers=headers, data=json.dumps(data))
# print(response.json())
# print()
