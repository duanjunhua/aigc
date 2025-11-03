"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-15 13:56
@Version: v1.0
@Description: 实用LangChain开发应用程序
    包括处理用户评论、基于文档问答、寻求外部知识等。
"""
from chatgpt.phase01.load_env import get_openai_api_key
# 从langchain 0.2.0 版本开始，OpenAI 需通过 langchain-community 包进行导入
from langchain_community.chat_models import ChatOpenAI

get_openai_api_key()

# 我们将参数temperature设置为0.0，从而减少生成答案的随机性
chat = ChatOpenAI(temperature = 0.0)
print(chat)


