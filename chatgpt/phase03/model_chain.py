"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-21 15:49
@Version: v1.0
@Description: 模型链
链（Chains）通常将大语言模型（LLM）与提示（Prompt）结合在一起，基于此，我们可以对文本或数据进行一系列操作。链（Chains）可以一次性接受多个输入。
    例如，我们可以创建一个链，该链接受用户输入，使用提示模板对其进行格式化，然后将格式化的响应传递给 LLM 。
我们可以通过将多个链组合在一起，或者通过将链与其他组件组合在一起来构建更复杂的链
"""
from chatgpt.tool import get_completion
from langchain_community.chat_models import ChatOpenAI

# 初始化llm
llm = ChatOpenAI(model="gpt-4o-mini",temperature = 0.0)

result = "AI未运行"

product = "笔记本电脑"
# ------------------------ 大语言模型链 ----------------------
# 大语言模型链（LLMChain）是一个简单但非常强大的链
import warnings
warnings.filterwarnings("ignore")

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# 初始化template
prompt = ChatPromptTemplate.from_template("以中文描述制造{product}的一个公司的最佳名称是什么?")

# 构建大语言模型链：以让我们以一种顺序的方式去通过运行提示并且结合到大语言模型中
chain = LLMChain(llm=llm, prompt=prompt)

# 运行
# result = chain.run(product)

# ------------------------ 简单顺序链 ----------------------
# 顺序链（SequentialChains）是按预定义顺序执行其链接的链。其中每个步骤都有一个输入/输出，一个步骤的输出是下一个步骤的输入。
from langchain.chains import SimpleSequentialChain

# 创建两个子链
# 提示模板 1 ：这个提示将接受产品并返回最佳名称来描述该公司
first_prompt = ChatPromptTemplate.from_template(
    "以中文精简描述制造{product}的一个公司的最好的名称是什么"
)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

# 提示模板 2 ：接受公司名称，然后输出该公司的长为20个单词的描述
second_prompt = ChatPromptTemplate.from_template(
    "以中文写一个15字的描述对于下面这个公司：{company_name}的"
)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

# 构建顺序链
simple_chain = SimpleSequentialChain(chains = [first_chain, second_chain], verbose=True)

# 运行
# result = simple_chain.run(product)


# ------------------------ 顺序链 ----------------------
"""
当只有一个输入和一个输出时，简单顺序链（SimpleSequentialChain）即可实现。当有多个输入或多个输出时，我们则需要使用顺序链（SequentialChain）来实现
"""
import pandas as pd
from langchain.chains import SequentialChain

# 创建子链
# prompt模板1
first_prompt = ChatPromptTemplate.from_template(
    "把下面的评论review翻译成中文:\n\n{Review}"
)
# chain 1: 输入：Review、输出：中文的Review
first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="Chinese_Review")

#子链2
# prompt模板2
second_prompt = ChatPromptTemplate.from_template(
    "请你用一句话来总结下面的评论review:\n\n{Chinese_Review}"
)
# chain 2: 输入：中文的Review 输出：总结
second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

#子链3
# prompt模板3
third_prompt = ChatPromptTemplate.from_template(
    "下面的评论review使用的什么语言:\n\n{Review}"
)
# chain 3: 输入：Review、输出：语言
third_chain = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

# 子链4
# prompt模板 4: 使用特定的语言对下面的总结写一个后续回复
fourth_prompt = ChatPromptTemplate.from_template(
    "使用特定的语言对下面的总结写一个后续回复:\n\n总结: {summary}\n\n语言: {language}"
)
# chain 4: 输入： 总结, 语言 输出： 后续回复
fourth_chain = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

# 子链组合
# 输入：review，输出：英文review，总结，后续回复
overall_chain = SequentialChain(
    chains=[first_chain, second_chain, third_chain, fourth_chain],
    input_variables=["Review"],
    output_variables=["Chinese_Review", "summary","followup_message"],
    verbose=True
)

# df = pd.read_csv('Data.csv')
# review = df.Review[0]
# result = overall_chain(review)


# ------------------------ 路由链 ----------------------
"""
根据输入将其路由到一条链，具体取决于该输入到底是什么。每个子链都专门用于特定类型的输入，那么可以组成一个路由链，它首先决定将它传递给哪个子链，然后将它传递给那个链
    路由器由两个组件组成：
        路由链（Router Chain）：路由器链本身，负责选择要调用的下一个链
        目的链（destination_chains）：路由器链可以路由到的链
"""
from langchain.chains.router import MultiPromptChain    # 导入多提示链
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# --------- 定义提示模板 ---------
# 第一个提示适合回答物理问题
physics_template = """你是一个非常聪明的物理专家。你擅长用一种简洁并且易于理解的方式去回答问题。当你不知道问题的答案时，你承认你不知道.
这是一个问题:{input}"""
# 第二个提示适合回答数学问题
math_template = """你是一个非常优秀的数学家。你擅长回答数学问题。你之所以如此优秀，是因为你能够将棘手的问题分解为组成部分，回答组成部分，然后将它们组合在一起，回答更广泛的问题。
这是一个问题：{input}"""
# 第三个适合回答历史问题
history_template = """你是以为非常优秀的历史学家。 你对一系列历史时期的人物、事件和背景有着极好的学识和理解你有能力思考、反思、辩证、讨论和评估过去。你尊重历史证据，并有能力利用它来支持你的解释和判断。
这是一个问题:{input}"""
# 第四个适合回答计算机问题
computerscience_template = """ 你是一个成功的计算机科学专家。你有创造力、协作精神、前瞻性思维、自信、解决问题的能力、对理论和算法的理解以及出色的沟通技巧。\
你非常擅长回答编程问题。你之所以如此优秀，是因为你知道如何通过以机器可以轻松解释的命令式步骤描述解决方案来解决问题，并且你知道如何选择在时间复杂性和空间复杂性之间取得良好平衡的解决方案。
这还是一个输入：{input}"""

# --------- 对提示模版进行命名和描述 ---------
# 可以为每个模板命名，并给出具体描述
# 中文
prompt_infos = [
    {
        "名字": "物理学",
        "描述": "擅长回答关于物理学的问题",
        "提示模板": physics_template
    },
    {
        "名字": "数学",
        "描述": "擅长回答数学问题",
        "提示模板": math_template
    },
    {
        "名字": "历史",
        "描述": "擅长回答历史问题",
        "提示模板": history_template
    },
    {
        "名字": "计算机科学",
        "描述": "擅长回答计算机科学问题",
        "提示模板": computerscience_template
    }
]

"""
LLMRouterChain（此链使用 LLM 来确定如何路由事物）
多提示链：一种特定类型的链，用于在多个不同的提示模板之间进行路由。
"""
# 基于提示模版信息创建相应目标链，目标链是由路由链调用的链，每个目标链都是一个语言模型链
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["名字"]
    prompt_template = p_info["提示模板"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
destinations = [f"{p['名字']}: {p['描述']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# 创建默认目标链：一个当路由器无法决定使用哪个子链时调用的链
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# 定义不同链之间的路由模板
# 多提示路由模板
MULTI_PROMPT_ROUTER_TEMPLATE = """给语言模型一个原始文本输入，让其选择最适合输入的模型提示。系统将为您提供可用提示的名称以及最适合改提示的描述。\
如果你认为修改原始输入最终会导致语言模型做出更好的响应，你也可以修改原始输入。

<< 格式 >>
返回一个带有JSON对象的markdown代码片段，该JSON对象的格式如下：
```json
{{{{
    "destination": 字符串 \ 使用的提示名字或者使用 "DEFAULT"
    "next_inputs": 字符串 \ 原始输入的改进版本
}}}}

记住：“destination”必须是下面指定的候选提示名称之一，或者如果输入不太适合任何候选提示，则可以是 “DEFAULT” 。
记住：如果您认为不需要任何修改，则 “next_inputs” 可以只是原始输入。

<< 候选提示 >>
{destinations}
<< 输入 >>
{{input}}

<< 输出 (记得要包含 ```json)>>

样例:
<< 输入 >>
"什么是黑体辐射?"
<< 输出 >>
```json
{{{{
"destination": 字符串 \ 使用的提示名字或者使用 "DEFAULT"
"next_inputs": 字符串 \ 原始输入的改进版本
}}}}
"""

# 构建路由链：通过传入llm和整个路由提示来创建路由链
"""
注意的是这里有路由输出解析，这很重要，因为它将帮助这个链路决定在哪些子链路之间进行路由
"""
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# 创建整体链路
# 多提示链（router_chain：路由链路、destination_chains：目标链路、default_chain：默认链路）
chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)

# 提问
# 物理
result = chain.run("什么是波的散射？")
print(result)
# 数学
result = chain.run("999*999=？")
print(result)
# 历史
result = chain.run("秦国统一天下时间？")
print(result)
# 计算机
result = chain.run("学习AI需要掌握哪些技术？")
print(result)
