"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-14 17:15
@Version: v1.0
@Description: 聊天机器人
"""
# ChatGPT 这样的聊天模型实际上是组装成以一系列消息作为输入，并返回一个模型生成的消息作为输出的

from chatgpt.tool import get_completion, get_completion_from_messages
import panel as pn  # GUI

result = "未运行AI"

# 讲笑话
joke_messages = [
    {'role': 'system', 'content': '你是一个像莎士比亚一样说话的助手。'},  # 向助手描述它应该如何表现的方式
    {'role': 'user', 'content': '给我讲个笑话'},  # 用户提问
    {'role': 'assistant', 'content': '鸡为什么过马路'},  # 助手回复
    {'role': 'user', 'content': '我不知道'}  # 用户回答
]

# 友好的聊天机器人
friend_message = [
    {'role': 'system', 'content': '你是个友好的聊天机器人。'},
    {'role': 'user', 'content': '你好, 我是Michael。'}
]

# 构建上下文
no_context_message = [
    {'role': 'system', 'content': '你是个友好的聊天机器人。'},
    {'role': 'user', 'content': '好，你能提醒我，我的名字是什么吗？'}
]

# 每次与语言模型的交互都互相独立，味着我们必须提供所有相关的消息，以便模型在当前对
# 话中进行引用。如果想让模型引用或 “记住” 对话的早期部分，则必须在模型的输入中提供早期的交流。
# 我们将其称为上下文 (context)
has_context_messages = [
    {'role': 'system', 'content': '你是个友好的聊天机器人。'},
    {'role': 'user', 'content': 'Hi, 我是Michael'},
    {'role': 'assistant', 'content': "Hi Michael! 很高兴认识你。今天有什么可以帮到你的吗?"},
    {'role': 'user', 'content': '是的，你可以提醒我, 我的名字是什么?'}]


# result = get_completion_from_messages(has_context_messages, temperature=1)
# print(result)

# 点餐机器人
def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role': 'user', 'content': f"{prompt}"})
    response = get_completion_from_messages(context)
    context.append({'role': 'assistant', 'content': f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600)))
    return pn.Column(*panels)


pn.extension()
panels = []  # collect display
context = [
    {
        'role': 'system',
        'content': '''
            你是订餐机器人，为披萨餐厅自动收集订单信息。
            你要首先问候顾客。然后等待用户回复收集订单信息。收集完信息需确认顾客是否还需要添加其他内容。
            最后需要询问是否自取或外送，如果是外送，你要询问地址。
            最后告诉顾客订单总金额，并送上祝福。
            请确保明确所有选项、附加项和尺寸，以便从菜单中识别出该项唯一的内容。
            你的回应应该以简短、非常随意和友好的风格呈现。
            菜单包括：
            
            菜品：
                意式辣香肠披萨（大、中、小） 12.95、10.00、7.00
                芝士披萨（大、中、小） 10.95、9.25、6.50
                茄子披萨（大、中、小） 11.95、9.75、6.75
                薯条（大、小） 4.50、3.50
                希腊沙拉 7.25
            配料：
                奶酪 2.00
                蘑菇 1.50
                香肠 3.00
                加拿大熏肉 3.50
                AI酱 1.50
                辣椒 1.00
            饮料：
                可乐（大、中、小） 3.00、2.00、1.00
                雪碧（大、中、小） 3.00、2.00、1.00
                瓶装水 5.00
        '''
    }
]  # accumulate messages

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")
interactive_conversation = pn.bind(collect_messages, button_conversation)
dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard.show()

# 提问：
# 1. 帮我来一份芝士披萨，大的，配料家蘑菇，再来一份大可乐。自取
# 2. 能告诉我金额怎么计算的吗
# 3. 创建上一个食品订单的 json 摘要。逐项列出每件商品的价格，字段应该是 1) 披萨，包括大小 2) 配料列表 3) 饮料列表，包括大小 4) 配菜列表包括大小 5) 总价，你应该给我返回一个可解析的Json对象，包括上述字段