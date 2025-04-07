# A simple local Deepseek_RAG application test with no use of Ollama architecture


## Preparation 准备工作
```bash
# 1. 安装 huggingface_hub (如果尚未安装)
pip install huggingface_hub

# 2. 登录 Hugging Face
huggingface-cli login

# 3. 安装 Git LFS (如果尚未安装)
sudo apt-get update
sudo apt-get install git-lfs
git lfs install

# 4. 克隆模型仓库
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat

# 5. 下载模型文件
cd deepseek-llm-7b-chat
git lfs pull
```

## RAG本地知识库挂载
```bash
pip install langchain transformers sentence-transformers faiss-cpu unstructured pdf2image python-docx docx2txt
```

*   `langchain`: 用于构建 RAG 流程。
*   `transformers`: 用于加载 DeepSeek 7B 模型。
*   `sentence-transformers`: 用于生成文本嵌入 (embeddings)。
*   `faiss-cpu`: 用于存储和检索文本嵌入 (向量数据库)。
*   `unstructured`: 用于加载各种文档 (text, pdf, word)。
*   `pdf2image`:  PDF 处理依赖。
*   `python-docx`: Word 文档处理依赖。
* `docx2txt`: txt文档处理依赖。


## 加载文档并创建向量数据库
 `Please run vector.py`

 *   **`DATA_PATH`**:  替换为你的本地知识库文档所在的目录。
*   **`DB_FAISS_PATH`**:  指定向量数据库的保存路径。
*   **文档加载**: 使用 `TextLoader`, `PDFLoader`, `Docx2txtLoader`, `UnstructuredFileLoader` 加载不同类型的文档。  `UnstructuredFileLoader` 可以处理更多类型的文档，但可能需要安装额外的依赖。
*   **文本切分**: 使用 `RecursiveCharacterTextSplitter` 将文档切分成小的文本块 (chunks)。  `chunk_size` 和 `chunk_overlap` 可以根据你的需求调整。
*   **Embeddings**: 使用 `HuggingFaceEmbeddings` 生成文本块的 embeddings。  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 是一个常用的多语言 embedding 模型。  也可以选择其他模型。
*   **向量数据库**: 使用 `FAISS` 存储 embeddings。  `FAISS` 是一个高效的向量相似度搜索库。

## 创建本地对话脚本
`Please run chat.py`
*   **`MODEL_PATH`**:  替换为你的 DeepSeek 7B 模型路径。
*   **`DB_FAISS_PATH`**:  替换为你的向量数据库路径。
*   **`load_model()`**: 加载 DeepSeek 7B 模型和 tokenizer。
*   **`load_vector_db()`**: 加载 FAISS 向量数据库。
*   **`create_rag_chain()`**: 创建 RAG 链。
    *   `RetrievalQA.from_chain_type()`:  使用 LangChain 的 `RetrievalQA` 类创建 RAG 链。
    *   `chain_type="stuff"`:  指定 chain type。  "stuff" 是最简单的 chain type，它将所有检索到的文档都塞到 prompt 中。  适合小文档。对于大文档，可以考虑使用 "map_reduce" 或 "refine" 等 chain type。
    *   `retriever`:  使用 `db.as_retriever()` 创建 retriever。  `search_kwargs={'k': 3}` 表示从向量数据库中检索最相关的 3 个文档。
    *   `return_source_documents=True`:  返回源文档。
*   **`generate_response()`**:  生成回复。
*   **自定义 Prompt**: 使用 `PromptTemplate` 自定义 Prompt，让模型更好地利用上下文信息。
*   **适配 langchain 的 llm 接口**:  由于 `RetrievalQA` 需要 langchain 风格的 llm 接口，所以需要用 `HuggingFacePipeline` 包装一下 `transformers` 的 `pipeline`。
*   **打印源文档**:  在回复中打印源文档，方便验证信息的来源。

## 准备知识库文档
将你的 text, pdf, word 文档放到 `data` 目录中
