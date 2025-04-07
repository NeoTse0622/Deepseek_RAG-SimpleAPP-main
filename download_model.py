# coding=utf-8
from huggingface_hub import snapshot_download

model_id = "deepseek-ai/deepseek-llm-7b-chat"  # ģ�� ID
local_dir = "./deepseek-llm-7b-chat"  # ���ر���Ŀ¼ (��ѡ)
local_dir_use_symlinks = False # �Ƿ�ʹ�÷������� (��ѡ)

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks,
        cache_dir="./.cache" # ���û���Ŀ¼ (��ѡ)
    )
    print(f"model {model_id} success download to {local_dir}")
except Exception as e:
    print(f"Download Error !!!: {e}")
    print("Check the internet connection and HF_ENDPOINT")
    print("If error still exsit ,please choose other proxy")
