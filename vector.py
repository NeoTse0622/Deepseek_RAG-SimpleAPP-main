# coding=utf-8
import os
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

DATA_PATH = "data/"  # 你的知识库文档所在的目录
DB_FAISS_PATH = "vectorstore/db_faiss"  # 向量数据库保存路径

def create_vector_db():
    """
    加载文档，切分文本，生成 embeddings 并存储到 FAISS 向量数据库中。
    """

    document_paths = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            document_paths.append(file_path)

    documents = []
    for doc_path in tqdm(document_paths, desc="加载文档"):
        try:
            if doc_path.endswith(".txt"):
                loader = TextLoader(doc_path, encoding="utf-8")
            elif doc_path.endswith(".pdf"):
                loader = PyPDFLoader(doc_path)  # 使用 PyPDFLoader
            elif doc_path.endswith(".docx") or doc_path.endswith(".doc"):
                try:
                    loader = Docx2txtLoader(doc_path)
                except ImportError:
                    print(f"docx2txt 未安装，尝试使用 UnstructuredFileLoader 加载 {doc_path}")
                    loader = UnstructuredFileLoader(doc_path)
            else:
                loader = UnstructuredFileLoader(doc_path)  # 尝试使用通用加载器
            documents.extend(loader.load())
        except Exception as e:
            print(f"加载文档 {doc_path} 失败: {e}")

    print(f"加载的文档数量: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"分割后的文本块数量: {len(texts)}")

    # 过滤掉空文本块
    texts = [text for text in texts if text.page_content.strip()]

    # 打印文本块内容
    # for i, text in enumerate(texts):
    #     print(f"文本块 {i}: {text.page_content}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                       model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    import torch
    create_vector_db()
    print("向量数据库创建完成！")


