from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 创建目录加载器,加载data目录下所有PDF文件
loader = DirectoryLoader( "./data/aibox",  glob="**/*.pdf", loader_cls=PyPDFLoader  )
data = loader.load()
print(f"成功加载 {len(data)} 页PDF文档")

# 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 英文embedding模型
# local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# 中文embedding模型 BAAI 670MB
local_embeddings = OllamaEmbeddings(model="bge-large")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
output_parser = StrOutputParser()

# 连接模型
model = ChatOllama(
    base_url='http://localhost:11434',
    model="llama3.1",
)
# 从prompt.txt文件中读取RAG模板
with open('./src/prompt.txt', 'r', encoding='utf-8') as file:
    RAG_TEMPLATE = file.read().strip()
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构造langchain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)