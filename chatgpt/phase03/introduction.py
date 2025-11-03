"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-20 11:21
@Version: v1.0
@Description: LangChain介绍
"""

from chatgpt.tool import get_completion

result = "AI未运行"

# 计算
# result = get_completion("1+1是什么")

#  用普通话表达海盗邮件
customer_email = """
嗯呐，我现在可是火冒三丈，我那个搅拌机盖子竟然飞了出去，把我厨房的墙壁都溅上了果汁！
更糟糕的是，保修条款可不包括清理我厨房的费用。
伙计，赶紧给我过来！
"""
# 普通话 + 平静、尊敬的语调
style = "正式普通话，用一个平静、尊敬、有礼貌的语调"
prompt = """
把由三个反引号分隔的文本翻译成一种{style}风格。
文本: ```{customer_email}```
"""
# print(f"提示：{prompt}")
# result = get_completion(prompt)

# ==================================== LangChain ====================================
# LangChain构造用普通话表达海盗邮件
from langchain.prompts import ChatPromptTemplate
# 从langchain 0.2.0 版本开始，OpenAI 需通过 langchain-community 包进行导入
from langchain_community.chat_models import ChatOpenAI
# 我们将参数temperature设置为0.0，从而减少生成答案的随机性
chat = ChatOpenAI(temperature = 0.0)

# ------------------------ 提示模板 ----------------------
template = """
把由三个反引号分隔的文本翻译成一种{style}风格。文本: ```{custom_text}```
"""
prompt_template = ChatPromptTemplate.from_template(template)
# print(prompt_template.messages[0].prompt)

# 使用提示模版
customer_msg = prompt_template.format_messages(style=style, custom_text=customer_email)
# 打印客户消息类型：list
# print(f"客户消息类型：{type(customer_msg)}")
# 打印第一个客户消息类型：langchain_core.messages.human.HumanMessage
# print(f"第一个客户消息类型：{type(customer_msg[0])}")
# 打印第一个元素：
# print(f"第一个客户客户消息类型类型: {customer_msg[0]}")
# response = chat(customer_msg)
# result = response.content

service_reply = "嘿，顾客，保修不包括厨房的清洁费用， 因为您在启动搅拌机之前忘记盖上盖子而误用搅拌机, 这是您的错。倒霉！ 再见！"
service_style_pirate = "一个有礼貌的语气使用海盗风格"
service_reply_message = prompt_template.format_messages(style=service_style_pirate, custom_text = service_reply)
# response = chat(service_reply_message)
# result = response.content

# 使用提示模版，可以让我们更为方便地重复使用设计好的提示
'''
LangChain还提供了提示模版用于一些常用场景。比如自动摘要、问答、连接到SQL数据库、连接到不同的API。
通过使用LangChain内置的提示模版，可以快速建立自己的大模型应用，而不需要花时间去设计和构造提示。
'''

# ------------------------ 输出解析 ----------------------
customer_review = """\
这款吹叶机非常神奇。 它有四个设置：\
吹蜡烛、微风、风城、龙卷风。 \
两天后就到了，正好赶上我妻子的\
周年纪念礼物。 \
我想我的妻子会喜欢它到说不出话来。 \
到目前为止，我是唯一一个使用它的人，而且我一直\
每隔一天早上用它来清理草坪上的叶子。 \
它比其他吹叶机稍微贵一点，\
但我认为它的额外功能是值得的。
"""
review_template = """\
对于以下文本，请从中提取以下信息：
礼物：该商品是作为礼物送给别人的吗？ \
如果是，则回答 是的；如果否或未知，则回答 不是。
交货天数：产品需要多少天到达？ 如果没有找到该信息，则输出-1。
价钱：提取有关价值或价格的任何句子，并将它们输出为逗号分隔的 Python 列表。

使用以下键将输出格式化为 JSON：
礼物
交货天数
价钱
文本: {text}
"""
# parser_template = ChatPromptTemplate.from_template(review_template)
# print("提示模版：", prompt_template)
#
# parser_messages = parser_template.format_messages(text=customer_review)
# response = chat(parser_messages)
# result = response.content
# # str
# print("结果类型:", type(response.content))


review_template_2 = """\
对于以下文本，请从中提取以下信息：
礼物：该商品是作为礼物送给别人的吗？
如果是，则回答 是的；如果否或未知，则回答 不是。
交货天数：产品到达需要多少天？ 如果没有找到该信息，则输出-1。
价钱：提取有关价值或价格的任何句子，并将它们输出为逗号分隔的 Python 列表。
文本: {text}

{format_instructions}
"""
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="礼物", description="这件物品是作为礼物送给别人的吗？如果是，则回答 是的，如果否或未知，则回答 不是。")
delivery_days_schema = ResponseSchema(name="交货天数", description="产品需要多少天才能到达？如果没有找到该信息，则输出-1。")
price_value_schema = ResponseSchema(name="价钱", description="提取有关价值或价格的任何句子，并将它们输出为逗号分隔的 Python 列表")

response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
# print("输出格式规定：", format_instructions)

json_prompt = ChatPromptTemplate.from_template(review_template_2)
json_messages = json_prompt.format_messages(text=customer_review, format_instructions=format_instructions)
# messages = json_messages[0].content
# print(f"客户消息：{messages}")
response = chat(json_messages)
# str
print("结果类型:", type(response.content))
output_dict = output_parser.parse(response.content)
# output_dict 类型为字典( dict ), 可直接使用 get 方法
print("解析后的结果类型:", type(output_dict))
print("解析后的结果:", output_dict)
result = response.content

print(result)

