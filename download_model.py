# coding=utf-8
from huggingface_hub import snapshot_download

model_id = "deepseek-ai/deepseek-llm-7b-chat"  # 模型 ID
local_dir = "./deepseek-llm-7b-chat"  # 本地保存目录 (可选)
local_dir_use_symlinks = False # 是否使用符号链接 (可选)

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks,
        cache_dir="./.cache" # 设置缓存目录 (可选)
    )
    print(f"model {model_id} success download to {local_dir}")
except Exception as e:
    print(f"Download Error !!!: {e}")
    print("Check the internet connection and HF_ENDPOINT")
    print("If error still exsit ,please choose other proxy")
