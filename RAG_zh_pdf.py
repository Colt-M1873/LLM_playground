from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# 创建目录加载器,加载data目录下所有PDF文件
loader = DirectoryLoader(
    "./data/aibox",  # 指定PDF文件所在目录
    glob="**/*.pdf",  # 匹配所有PDF文件,包括子目录
    loader_cls=PyPDFLoader  # 使用PDF加载器
)

# 加载所有PDF文件
data = loader.load()
print(f"成功加载 {len(data)} 页PDF文档")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 英文embedding模型
# local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# 中文embedding模型 670MB
local_embeddings = OllamaEmbeddings(model="bge-large")
# 中文embedding模型 408MB
# local_embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh")

# 创建vectorstore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

from langchain_ollama import ChatOllama

model = ChatOllama(
    base_url='http://localhost:11434',
    model="llama3.1",
)

# 构造langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 简单qa
from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. You need to give more detailed information according to the context.请尽量使用中文回答问题，并使用Markdown格式。
<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

retriever = vectorstore.as_retriever()

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# question = "aibox是什么?"
# print(qa_chain.invoke(question))

# 创建一个交互式问答循环
while True:
    # 获取用户输入的问题
    question = input("\n请输入您的问题(输入'exit'退出): ")
    
    # 检查是否退出
    if question.lower() == 'exit':
        print("感谢使用,再见!")
        break
        
    # 调用qa_chain获取答案并流式输出
    try:
        print("\n回答:", end=" ", flush=True)
        for chunk in qa_chain.stream(question):
            print(chunk, end="", flush=True)
        print()  # 换行
    except Exception as e:
        print("\n抱歉,处理您的问题时出现错误:", str(e))
