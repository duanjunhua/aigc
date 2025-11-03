"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-31 16:21
@Version: v1.0
@Description: 问答
    Langchain 在实现与外部数据对话的功能时需要经历下面的5个阶段，它们分别是：Document Loading->Splitting->Storage->Retrieval->Output
        1、完成数据的存储和获取
        2、获取相关的切分文档
        3、将文档传递给语言模型，获得答案

    一般流程如下：
        1、问题提出，
        2、查找相关的文档
        3、将查找的切分文档和系统提示一起传递给语言模型，获得答案

    默认情况下，我们将所有的文档切片都传递到同一个上下文窗口中，即同一次语言模型调用中。MapReduce、Refine 和 MapRerank 是三种方法，用于解决短上下文窗口的问题
"""
# 导入OpenAI参数
from chatgpt.tool import get_completion
# 导入OpenAI
from langchain_openai import ChatOpenAI
# 导入向量库
from langchain_chroma.vectorstores import Chroma
# 导入向量化模型
from langchain_openai.embeddings import OpenAIEmbeddings

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
# 查看加载向量数据库长度
print(vectorDb._collection.count())
# 根据相似性进行检索
question = "这节课的主要话题是什么？"
# docs = vectorDb.similarity_search(query=question, k=2)
# print(len(docs))

# ------------------------ 构造检索式问答链 ----------------------
"""
    基于 LangChain，可以构造一个使用 GPT4 进行问答的检索式问答链，这是一种通过检索步骤进行问答的方法。可以通过传入一个语言模型和一个向量数据
库来创建它作为检索器。然后，可以用问题作为查询调用它，得到一个答案。
"""
from langchain.chains import RetrievalQA

# 声明一个检索式问答链
qa_chain = RetrievalQA.from_chain_type(
    # 语言模型
    llm,
    # 检索器
    retriever=vectorDb.as_retriever()
)
# 可以以该方式进行检索问答
# res = qa_chain({"query": question})
# print(res["result"])

# ------------------------ 深入探究检索式问答链 ----------------------
"""
    在获取与问题相关的文档后，需要将文档和原始问题一起输入语言模型，生成回答。默认是合并所有文档，一次性输入模型。但存在【上下文长度限制】
的问题，若相关文档量大，难以一次将全部输入模型。针对这一问题，有 MapReduce 、Refine 和 MapRerank 三种策略。
    MapReduce：通过多轮检索与问答实现长文档处理
    Refine：让模型主动请求信息
    MapRerank：则通过问答质量调整文档顺序

    三种策略各有优劣 MapReduce 分批处理长文档，Refine 实现可交互问答，MapRerank 优化信息顺序。通过这些可以应对语言模型的上下文限制，
解决长文档问答困难，提升问答覆盖面。
"""

# ============= 基于模板的检索式问答链 =============
"""
它只涉及对语言模型的一次调用。也有局限性，即如果文档太多，可能无法将它们全部适配到上下文窗口中
"""
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

# ============= 基于 MapReduce 的检索式问答链 =============
"""
    在MapReduce技术中，首先将每个独立的文档单独发送到语言模型以获取原始答案。然后，这些答案通过最终对语言模型的一次调用组合成最终的答案。
    它的优势在于可以处理任意数量的文档，劣势在于涉及了更多对语言模型的调用
    
    这种方法有两个问题（如果信息分布在两个文档之间，它并没有在同一上下文中获取到所有的信息。）：
        1、速度慢
        2、结果实际上更差
"""
mapreduce_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorDb.as_retriever(),
    chain_type="map_reduce"
)
# res = mapreduce_chain({"query": question})
# print(res["result"])

# ============= 基于 Refine 的检索式问答链 =============
"""
    Refine 文档链类似于 MapReduce ，对于每一个文档，会调用一次 LLM。但改进之处在于，最终输入语言模型的 Prompt 是一个序列，将之前的回复与新文档组合在一起，并请求得到改进后的响应。
这是一种类似于 RNN 的概念，增强了上下文信息，从而解决信息分布在不同文档的问题。
    如：
        第一次调用，Prompt包含问题与文档A，语言模型生成初始回答。
        第二次调用，Prompt包含第一次回复、文档B，请求模型更新回答。
        以此类推，...
"""
refine_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorDb.as_retriever(),
    chain_type="refine"
)
# res = refine_chain({"query": question})

"""
时间更久，但这个结果比MapReduce链的结果要好，因为使用 Refine 文档链通过累积上下文，使语言模型能渐进地完善答案，而不是孤立处理每个文档。
该策略可以有效解决信息分散带来的语义不完整问题。
"""
# print(res["result"])




