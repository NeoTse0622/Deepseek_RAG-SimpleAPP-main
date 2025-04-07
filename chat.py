# coding=utf-8
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./deepseek-llm-7b-chat"  # 修改为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.7) # 添加了采样参数
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("欢迎使用 DeepSeek 7B Chat！")
    print("输入 'exit' 退出对话。")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("感谢使用，再见！")
            break

        try:
            response = generate_response(user_input)
            print("DeepSeek: " + response)
        except Exception as e:
            print(f"发生错误: {e}")
            print("请检查你的输入和模型是否正确加载。")

'''
1. **导入必要的库:**  `torch` 和 `transformers`。
2. **加载模型和 tokenizer:**  使用你提供的模型路径加载 DeepSeek 7B 模型和 tokenizer。  **请确保 `model_path` 指向正确的模型目录。**
3. **`generate_response(prompt)` 函数:**
   - 接收用户输入的 `prompt`。
   - 使用 tokenizer 将 prompt 转换为模型可以理解的输入格式。
   - 使用 `model.generate()` 生成回复。
     - `max_new_tokens`:  限制生成回复的最大长度。
     - `do_sample=True`: 启用采样，使回复更具创造性。
     - `top_p=0.9`:  控制采样的范围，只考虑概率最高的 90% 的 token。
     - `temperature=0.7`:  调整生成文本的随机性。  较低的值使回复更保守，较高的值使回复更随机。  你可以根据需要调整这些参数。
   - 使用 tokenizer 将模型生成的 token 转换为文本。
   - 返回生成的回复。
4. **`if __name__ == "__main__":` 块:**
   - 这是 Python 程序的入口点。
   - 打印欢迎信息和退出说明。
   - 进入一个无限循环，等待用户输入。
   - 获取用户输入，并将其转换为小写。
   - 如果用户输入 "exit"，则退出循环。
   - 调用 `generate_response()` 函数生成回复。
   - 打印模型的回复。
   - 使用 `try...except` 块捕获可能发生的错误，并打印错误信息。  这可以帮助你调试问题。

**使用方法:**

1. **保存代码:** 将代码保存为 `chat.py` (或其他你喜欢的名字)。
2. **确保模型已正确加载:**  确保 `model_path` 指向你的 DeepSeek 7B 模型目录，并且模型已正确加载。
3. **运行脚本:** 在终端中运行 `python chat.py`。
4. **开始对话:**  在终端中输入你的问题，然后按 Enter 键。  模型会生成回复。
5. **退出对话:**  输入 "exit" 并按 Enter 键退出对话。

**注意事项:**

* **模型路径:** 确保 `model_path` 变量指向你的 DeepSeek 7B 模型目录。
* **硬件要求:**  运行 DeepSeek 7B 模型需要一定的硬件资源 (GPU 和内存)。  如果你的硬件不足，可能会遇到性能问题或内存错误。
* **采样参数:**  你可以根据需要调整 `do_sample`, `top_p`, 和 `temperature` 参数，以控制生成文本的质量和多样性。
* **错误处理:**  代码包含基本的错误处理，但你可能需要根据实际情况添加更详细的错误处理。
* **提示词工程:**  模型的回复质量很大程度上取决于你提供的提示词。  尝试使用不同的提示词，以获得更好的结果。
* **编码问题:** 确保你的终端和脚本文件都使用 UTF-8 编码。

'''