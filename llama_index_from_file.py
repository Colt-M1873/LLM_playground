# 本代码演示如何通过llama_index来对文件（如txt或pdf）进行向量索引，然后传给ChatGPT来辅助阅读。
# llama_index可以将文字转换为向量，不仅能有效利用token，文字之间的联系也会更加紧密。
# llama_index的项目地址：https://github.com/jerryjliu/llama_index
#
# 使用：
# 1. pip install llama-index（写这个例子时所用的版本为0.5.1）
# 2. export OPENAI_API_KEY="sk-xxx" 设置环境变量
# 3. 将txt或pdf文件放在./data目录下
# 4. 命令行下运行python llama_index_example.py；或在ipython下，运行%run llama_index_example.py（更为推荐）
#
# 参考：
# - https://github.com/jerryjliu/llama_index/tree/main/examples/paul_graham_essay
# - https://github.com/jerryjliu/llama_index/blob/main/examples/vector_indices/SimpleIndexDemo-ChatGPT.ipynb
# - https://twitter.com/madawei2699/status/1632270319724691457 (注，该推中的代码已经过时）

# Loging for debug
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # logging.DEBUG for more if needed
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from langchain.chat_models import ChatOpenAI
from llama_index import LLMPredictor, ServiceContext
# Use LLM gpt-3.5-turbo
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=3072)

from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex
# Load files under ./data
documents = SimpleDirectoryReader("./data").load_data()
# Index file to embedding vector with gpt-3.5-turbo by calling ChatGPT embedding API
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
# Save index into file for later use
index.save_to_disk("index.json")
# Load index from file
new_index = GPTSimpleVectorIndex.load_from_disk("index.json", service_context=service_context)

# Query by calling ChatGPT API
response = new_index.query("请用中文回答。这段文字在讲什么？")
# response = new_index.query("请用中文回答。这段文字在讲什么？", response_mode="tree_summarize")
print(response)
