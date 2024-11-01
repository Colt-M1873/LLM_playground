import RAG

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
        print("\n回答:\n", end=" ", flush=True)
        for chunk in RAG.qa_chain.stream(question):
            print(chunk, end="", flush=True)
        print()  # 换行
    except Exception as e:
        print("\n抱歉,处理您的问题时出现错误:", str(e))
