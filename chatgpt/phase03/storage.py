"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-21 09:48
@Version: v1.0
@Description: 存储
"""
from chatgpt.tool import get_completion

result = "AI未运行"

"""
在 LangChain 中，储存指的是大语言模型（LLM）的短期记忆。LangChain中的储存模块，可将先前的对话嵌入到语言模型中的，使其具有连续对话的能力。
使用 LangChain 中的储存(Memory)模块时，它旨在保存、组织和跟踪整个对话的历史，从而为用户和模型之间的交互提供连续的上下文。
    LangChain 提供了多种储存类型：
        缓冲区储存允许保留最近的聊天消息
        摘要储存则提供了对整个对话的摘要
        实体储存则允许在多轮对话中保留有关特定实体的信息
储存模块可以通过简单的 API 调用来访问和更新，允许开发人员更轻松地实现对话历史记录的管理和维护
    例子主要介绍其中四种储存模块：
        对话缓存储存 (ConversationBufferMemory）：随着对话变得越来越长，所需的内存量也变得非常长，将大量的tokens发送到LLM的成本，也会变得更加昂贵，这也就是为什么API的调用费用是基于它需要处理的tokens数量而收费的
        对话缓存窗口储存 (ConversationBufferWindowMemory）：对话缓存窗口储存只保留一个窗口大小的对话。可以用于保持最近交互的滑动窗口，以便缓冲区不会过大
        对话令牌缓存储存 (ConversationTokenBufferMemory）
        对话摘要缓存储存 (ConversationSummaryBufferMemory）
"""
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 初始化llm
llm = ChatOpenAI(temperature = 0.0)

# ------------------------ 缓存存储 ----------------------
# 初始化内存
memory = ConversationBufferMemory()

# 新建一个 ConversationChain Class 实例
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# print("第一轮对话：")
# result = conversation.predict(input="你好，我叫迈克")
# print("第二轮对话：")
# result = conversation.predict(input="1+1等于多少？")
# print("第三轮对话：")
# result = conversation.predict(input="我叫什么名字？")
# print(result)

"""
当我们在使用大型语言模型进行聊天对话时，大型语言模型本身实际上是无状态的。语言模型本身并不记得到目前为止的历史对话
"""

# ------------------------ 缓存窗口存储 ----------------------
from langchain.memory import ConversationBufferWindowMemory


# k=1表明只保留一个对话记忆
memory = ConversationBufferWindowMemory(k=1)

# memory.save_context({"input": "你好，我叫皮皮鲁"}, {"output": "你好啊，我叫鲁西西"})
# memory.save_context({"input": "很高兴和你成为朋友！"}, {"output": "是的，让我们一起去冒险吧！"})
# print(memory.load_memory_variables({}))

# 在对话链中应用窗口储存
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# print("第一轮对话：")
# print(conversation.predict(input="你好, 我叫皮皮鲁"))
# print("第二轮对话：")
# print(conversation.predict(input="我在看三国演义？"))
# # 注意此处：由于这里用的是一个窗口的记忆，因此只能保存一轮的历史消息，因此AI并不能知道你第一轮对话中提到的名字，他最多只能记住上一轮（即第二轮）的对话信息
# print("第三轮对话：")
# print(conversation.predict(input="直接精简回答我叫什么名字以及我在看什么"))


# ------------------------ 对话字符缓存储存 ----------------------
"""
使用对话字符缓存记忆，内存将限制保存的token数量。如果字符数量超出指定数目，它会切掉这个对话的早期部分，以保留与最近的交流相对应的字符数量，但不超过字符限制
"""
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)

memory.save_context({"input": "朝辞白帝彩云间，"}, {"output": "千里江陵一日还。"})
memory.save_context({"input": "两岸猿声啼不住，"}, {"output": "轻舟已过万重山。"})
"""
ChatGPT使用一种基于字节对编码（Byte Pair Encoding，BPE）的方法来进行 tokenization （将输入文本拆分为token）。BPE 是一种常见的 tokenization
技术，它将输入文本分割成较小的子词单元。
注意：下面只会打印最后一句AI的回复："轻舟已过万重山。"
"""
# print(memory.load_memory_variables({}))

# ------------------------ 对话摘要缓存储存 ----------------------
"""
对话摘要缓存储存，使用 LLM 对到目前为止历史对话自动总结摘要，并将其保存下来
"""
from langchain.memory import ConversationSummaryBufferMemory

# 创建一个长字符串
schedule = "在八点你和你的产品团队有一个会议。你需要做一个PPT。上午9点到12点你需要忙于LangChain。\
Langchain是一个有用的工具，因此你的项目进展的非常快。中午，在意大利餐厅与一位开车来的顾客共进午餐 \
走了一个多小时的路程与你见面，只为了解最新的 AI。确保你带了笔记本电脑可以展示最新的 LLM 样例."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

memory.save_context({"input": "你好，我叫皮皮鲁"}, {"output": "你好啊，我叫鲁西西"})
memory.save_context({"input": "很高兴和你成为朋友！"}, {"output": "是的，让我们一起去冒险吧！"})
memory.save_context({"input": "今天的日程安排是什么？"}, {"output": f"{schedule}"})
print(memory.load_memory_variables({})['history'])

# ------------------------ 基于对话摘要缓存储存的对话链 ----------------------
# 对话摘要缓存memory，新建一个对话链
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
# result = conversation.predict(input="展示什么样的样例最好呢？")
# print(result)
# # 摘要记录更新了，添加了最新一次对话的内容总结
# print(memory.load_memory_variables({}))





