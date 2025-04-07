import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

MODEL_PATH = "./deepseek-llm-7b-chat"  # 你的模型路径
DB_FAISS_PATH = "vectorstore/db_faiss"  # 向量数据库路径

def load_model():
    """加载模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

def load_vector_db():
    """加载 FAISS 向量数据库"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                       model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def create_rag_chain(model, tokenizer, db):
    """创建 RAG 链"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,  #  这里的 llm 需要适配 langchain 的接口，需要做转换
        chain_type="stuff",  #  "stuff" 是最简单的 chain_type, 适合小文档
        retriever=db.as_retriever(search_kwargs={'k': 3}),  #  从向量数据库中检索最相关的 3 个文档
        return_source_documents=True,  #  返回源文档
        chain_type_kwargs={"prompt": prompt} # 使用自定义 prompt
    )
    return qa_chain

def generate_response(qa_chain, query):
    """生成回复"""
    response = qa_chain({"query": query})
    return response["result"], response["source_documents"]

# 自定义 Prompt
from langchain.prompts import PromptTemplate
prompt_template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
上下文：{context}
问题：{question}
有用的回答："""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

if __name__ == "__main__":
    model, tokenizer = load_model()
    db = load_vector_db()

    #  适配 langchain 的 llm 接口
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    num_return_sequences=1)

    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = create_rag_chain(llm, tokenizer, db)

    print("欢迎使用 DeepSeek 7B Chat (RAG)！")
    print("输入 'exit' 退出对话。")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("感谢使用，再见！")
            break

        try:
            response, source_documents = generate_response(qa_chain, user_input)
            print("DeepSeek: " + response)
            print("\n来源文档:")
            for doc in source_documents:
                print(f"  - {doc.metadata['source']}")
        except Exception as e:
            print(f"发生错误: {e}")
            print("请检查你的输入和模型是否正确加载。")
