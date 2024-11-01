Tokenizer专题：从意义到数据


简要讲解GPT2的结构以及Bert的结构。

Word embedding 包括 Token embedding和Position embedding

专注讲Token Embedding

目前你的理解：Token embedding是意义的向量化，意义相近的词汇在转换成向量后也是相近的。

真正令你感到奇妙且迷惑的是从词汇到意义到向量的映射过程，后续的学习过程是向量之间的计算与优化过程。

Word2Vec是什么

GPT2的Tokenizer详细看代码原理

统计机器翻译的原理？




llama_index_example.ipynb
RAG retrieval-augmented-generation设置

参考 
https://gist.github.com/fjchen7/89d129354bca42792a0ed949d5421ba8

<!-- 
Token Embedding的结果，通过GPT2或者llama的token-embedding结果来查看网络热梗的相关性，梗与年代时间人物的相关性。例如airplane-911-world trade center这些词之间的向量相似性
吉吉国大模型入门：ylg也能看懂的硅胶模型，换电棍的音源准备材料做简单的教程，以3b1b的教程结合原论文做ppt
以电棍的相关文章报道梗百科内容做RAG， 大模型RAG做一期视频，做抽象名人搜索的大模型RAG，整理孙笑川电棍山泥若之类的语料去先做RAG，后做训练
推公式，一步一步推QKV公式结合ipynb代码调试理解做视频。


RAG优化，embedding optimization

RAG和微调训练的区别有哪些，拿mac做做训练，两边都尝试一下，macM3的性能如果不拿来练练模型只是写业务代码的话，就像开是超跑买菜，太浪费了。


LLM+RAG达到 输入 欧内的手，后面大模型自动补全 好汉
 可视化显示模型的预测结果，说明模型是因为预料中出现过欧内的手，好汉这个组合，并且出现概率比较高，才会选择好汉。

调整模型的tempreature，从 欧内的手好汉，到欧内的手sdacxs乱码，体现tempreature超参数的作用。 温度越高，熵越大，越混乱无序。

找意大利语的123456789和英雄联盟之间的关系，空间向量可视化。
是否otto和LOL的空间距离最近，是的话那就可能因为是电棍。


空间向量可视化做出来之后能帮你看到许多东西，看3b1b视频有没有代码，是怎么实现的。




https://learn.deeplearning.ai/ 有很多课程，特别是LLM的课程，学一学


https://learn.deeplearning.ai/courses/multimodal-rag-chat-with-videos/lesson/7/multimodal-rag-with-multimodal-langchain



Samkim三鑫集团
看看三星官方号是啥样的，模仿一下

千里马OS，bgm配千里马在奔驰

搜索手机一键切换红色模式的视频是怎么做的

阿里巴巴-阿里郎郎
Alibaba-Alirangrang

-->


# RAG

Prompt Engineering < RAG < Finetuning < Train from scratch

Finetuning 太吃性能，即使是Mistral 7B也需要3090以上的显卡和约64GB的现存

RAG论文
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks  https://arxiv.org/pdf/2005.11401


先把这个课程操作一遍
https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/1/introduction

RAG项目
https://github.com/chatchat-space/Langchain-Chatchat 

llama_index用于对文件创建索引，以便进行RAG
https://github.com/run-llama/llama_index


huggingface找RAG项目

本地跑性能跟得上的话最好直接docker来做，不要自己配环境


从RAG到KimGPT

爬虫爬取朝中社2024年以来的所有文章，其实十来篇应该就已经足够对文风进行模仿了。实质上只需要运行一段RAG代码就能完成。




Langchain课程
https://learn.deeplearning.ai/courses/langchain/lesson/2/models,-prompts-and-parsers






Huggingface Inference API 免费对话

以
获取API token，权限选Read


非流式传输：
以 meta-llama/Llama-3.2-1B 为例
https://huggingface.co/meta-llama/Llama-3.2-1B


流式传输；
以https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct?inference_api=true 
为例


Huggingface + Llamaindex 做RAG ？ 能否实现用endpoint 做RAG？

https://huggingface.co/learn/cookbook/rag_llamaindex_librarian




# Langchain+Ollama本地部署

https://blog.csdn.net/lovechris00/article/details/136869964

# Langchain + Ollama 本地RAG

https://github.com/datawhalechina/handy-ollama/blob/main/docs/C7/3.%20%E4%BD%BF%E7%94%A8%20LangChain%20%E6%90%AD%E5%BB%BA%E6%9C%AC%E5%9C%B0%20RAG%20%E5%BA%94%E7%94%A8.md

需要有ollama 3.1模型
ollama runxxx

需要在本地有nomic-embed-text模型
ollama pull nomic-embed-text


中文的词嵌入模型 https://ollama.com/library/bge-large

https://ollama.com/znbang/bge 注意这个bge是英文模型，bge-large才是中文
https://ollama.com/shaw/dmeta-embedding-zh

包装成一个shell脚本，包括创建venv，python install用清华源，ollama拉模型啦embed模型，跑openwebui




GPT key实现远程RAG
https://gist.github.com/fjchen7/89d129354bca42792a0ed949d5421ba8




# 本地部署ollama文档，常用命令默认端口整理.




# 命令行实现流式显示，python代码







# 极简的本地学城文档查询助手
目录文件中提取所有关联网址，webbaseloader解析所有网址并向量化
输入到本地跑的ollama来做RAG

安全性保障：因为全在本地，访问学城网址也是在本机，所以没有被外人得知文档内容的风险。






# similarity_search()原理？




langchain 官方教程，有很多关于RAG的文章
https://python.langchain.com/docs/tutorials/



# 对比nomic-embed-text和中文的文本嵌入模型



# olllama + langchain + openwebui
本地RAG对话返回





# openwebui接入langchain

通过pipeline进行
https://openwebui.com/f/colbysawyer/langchain_pipe

# llamaindex https://huggingface.co/learn/cookbook/rag_llamaindex_librarian





# 放低条件，不要优先搞openwebui，先把命令行对话搞定






