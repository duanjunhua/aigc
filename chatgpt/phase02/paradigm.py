"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-15 16:04
@Version: v1.0
@Description: 语言模型，提问范式与 Token
    大型语言模型（LLM）的工作原理、训练方式以及分词器（tokenizer）等细节对 LLM 输出的影响
    LLM 的提问范式（chat format）：是一种指定系统消息（system message）和用户消息（user message）的方式

    大语言模型（LLM）是通过预测下一个词的监督学习方式进行训练的。以预测下一个词为训练目标的方法使得语言模型获得强大的语言生成能力

    大型语言模型主要可以分为两类:基础语言模型和指令调优语言模型
        基础语言模型（Base LLM）通过反复预测下一个词来训练的方式进行训练，没有明确的目标导向
        指令微调的语言模型（Instruction Tuned LLM）则进行了专门的训练，以便更好地理解问题并给出符合指令的回答。指令微调使语言模型更加适合任务导向的对话应用
"""
# 熟练掌握指令微调的工作机制，是开发者实现语言模型应用的重要一步
from chatgpt.tool import get_completion, get_completion_from_messages, get_completion_from_messages_tokens, get_completion_from_messages_tokens_count

result = "AI未运行"

# ------------------------ 语言模型 ----------------------------------
'''
如何将基础语言模型转变为指令微调语言模型，就是训练一个指令微调语言模型（例如ChatGPT）的过程
1. 在大规模文本数据集上进行无监督预训练，获得基础语言模型（需要使用数千亿词甚至更多的数据，在大型超级计算系统上可能需要数月时间）
2. 使用包含指令及对应回复示例的小数据集对基础模型进行有监督 fine-tune，让模型逐步学会遵循指令生成输出，可以通过雇佣承包商构造
适合的训练示例（从基础语言模型到指令微调语言模型的转变过程可能只需要数天时间，使用较小规模的数据集和计算资源）
'''
# result = get_completion("中国的首都是哪里？")

# ------------------------ Tokens ----------------------------------
# Tokens：LLM 实际上并不是重复预测下一个单词，而是重复预测下一个 token
# 对于一个句子，语言模型会先使用分词器将其拆分为一个个 token ，而不是原始的单词。对于生僻词，可能会拆分为多个 token 。这样可以大幅降低字典规模，提高模型训练和推断的效率
# 注意：对于英文输入，一个 token 一般对应 4 个字符或者四分之三个单词；对于中文输入，一个token 一般对应一个或半个词

# token 限制是输入的 Prompt 和输出的 completion 的 token 数之和，因此输入的 Prompt 越长，能输出的completion 的上限就越低
# result = get_completion("Take the letters in lollipop and reverse them")

# ------------------------  辅助函数 (提问范式) ----------------------------------
'''
语言模型提供了专门的“提问格式”，可以更好地发挥其理解和回答问题的能力。
System、User、Assistant 消息（区分了“系统消息”和“用户消息”两个部分）：
    system → assistant
    user → assistant  (assistant → user)
    例如：
        系统消息:你是一个能够回答各类问题的助手。
        用户消息:太阳系有哪些行星?
'''
messages = [
    {'role': 'system', 'content': '你是一个助理，并以李白的风格作出回答。'},
    {'role': 'user','content': '就快乐的上班为主题给我写一首短诗'},
]
# result = get_completion_from_messages_tokens(messages,temperature=1)

messages_merge = [
    {'role': 'system', 'content': '你是一个助理，并以李白的风格作出回答，你的所有答复只能是一句话。'},
    {'role': 'user','content': '就快乐的上班为主题给我写一首短诗'},
]
# result = get_completion_from_messages_tokens(messages_merge,temperature=1)

# result, tokens = get_completion_from_messages_tokens_count(messages, temperature=1)
# print(tokens)

# ------------------------  评估输入 ----------------------------------
'''
使用系统消息（system_message）作为整个系统的全局指导，并选择使用 “#” 作为分隔符。
分隔符是用来区分指令或输出中不同部分的工具，可以帮助模型更好地识别各个部分，从而提高系统在执行特定任务时的准确性和效率
'''

print(result)



