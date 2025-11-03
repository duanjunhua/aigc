"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-31 17:07
@Version: v1.0
@Description: 聊天记忆
    检索增强生成 (retrieval augmented generation，RAG) 的整体工作流程：
        文档加载 、 切分 、 存储、检索和输出

    添加聊天历史的功能
"""
# 导入OpenAI参数
from chatgpt.tool import get_completion
# 导入OpenAI
from langchain_openai import ChatOpenAI
# 导入向量库
from langchain_chroma.vectorstores import Chroma
# 导入向量化模型
from langchain_openai.embeddings import OpenAIEmbeddings

question = "这节课的主要话题是什么？"

# 初始化llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
# 初始化模型
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载向量数据库
persist_directory = "chroma"
vectorDb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

from langchain.chains import RetrievalQA
# 导入提示词模板
from langchain.prompts import PromptTemplate

# 定义提示模板，包含一些关于如何使用下面的上下文片段的说明，然后有一个上下文变量的占位符
template = """使用以下上下文片段来回答最后的问题。如果你不知道答案，只需说不知道，不要试图编造答案。答案最多使用三个句子。
尽量简明扼要地回答。在回答的最后一定要说"感谢您的提问！"
{context}
问题：{question}
有用的回答：
"""
prompt = PromptTemplate.from_template(template)

# 基于该模板构建检索式问答链
template_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorDb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
# 提问
# res = template_chain({"query": question})

# 其中包含了检索的原文档：source_documents以及结果result
# print(res["result"])

# ------------------------ 记忆（Memory） ----------------------
"""
    使用 ConversationBufferMemory 。它保存聊天消息历史记录的列表，这些历史记录将在回答问题时与问题一起传递给聊天机器人，从而将它们添加到上下文中
"""
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    # 与 prompt 的输入变量保持一致
    memory_key="chat_history",
    # 以消息列表的形式返回聊天记录，而不是单个字符串
    return_messages=True
)

# ------------------------ 对话检索链（ConversationalRetrievalChain） ----------------------
"""
    对话检索链（ConversationalRetrievalChain）在检索 QA 链的基础上，增加了处理对话历史的能力。它的工作流程是:
        1. 将之前的对话与新问题合并生成一个完整的查询语句。
        2. 在向量数据库中搜索该查询的相关文档。
        3. 获取结果后,存储所有答案到对话记忆区。
        4. 用户可在 UI 中查看完整的对话流程。
这种链式方式将新问题放在之前对话的语境中进行检索，可以处理依赖历史信息的查询。并保留所有信息在对话记忆中，方便追踪。
"""
from langchain.chains import ConversationalRetrievalChain

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorDb.as_retriever(),
    memory=memory
)
first_answer = conversation_chain({"question": question})
print(f"{question} \n", first_answer['answer'])

next_question = "为什么要学习这些？"
next_answer = conversation_chain({"question": next_question})
print(f"{next_question} \n", next_answer['answer'])


