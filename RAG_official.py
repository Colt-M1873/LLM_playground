# https://python.langchain.com/docs/tutorials/local_rag/

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()


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


vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)

print(len(docs))

# print(docs[0])


from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

model = Ollama(base_url='http://localhost:11434',model="llama3.1")

# 构造langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | model | StrOutputParser()

question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)
# print(docs)
# print(chain.invoke(docs))



# 简单qa
from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)

# Run
# print(chain.invoke({"context": docs, "question": question}))    


# RAG

retriever = vectorstore.as_retriever()

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "What are the approaches to Task Decomposition?"

print(qa_chain.invoke(question))
