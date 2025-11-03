"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-02-11 16:
@Version: v1.0
@Description: openai 工具
"""
from openai import OpenAI
from chatgpt.phase01.load_env import get_openai_api_key

get_openai_api_key()

client = OpenAI()


def get_completion(prompt, model="gpt-4o-mini", temperature=0):
    # 调用 OpenAI 的ChatCompletion 端点
    response = client.chat.completions.create(
        # 所选模型的ID，如：gpt-4、gpt-3.5-turbo等。注意：不是所有可用的模型都与openai.ChatCompletion兼容
        model=model,
        # 表示对话的消息对象数组。包含两个属性role（可能的值有：system、user、assistant）和content（包含对话消息的字符串）
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature= temperature,
    )
    return response.choices[0].message.content


def get_completion_from_messages(messages, model="gpt-4o-mini", temperature=0):
    # 调用 OpenAI 的ChatCompletion 端点
    response = client.chat.completions.create(
        # 所选模型的ID，如：gpt-4、gpt-3.5-turbo等。注意：不是所有可用的模型都与openai.ChatCompletion兼容
        model=model,
        # 表示对话的消息对象数组。包含两个属性role（可能的值有：system、user、assistant）和content（包含对话消息的字符串）
        messages=messages,
        temperature= temperature,
    )
    return response.choices[0].message.content


def get_completion_from_messages_tokens(messages, model="gpt-4o-mini", temperature=0, max_tokens = 500):
    """
        封装一个支持更多参数的自定义访问 OpenAI GPT4 的函数
        参数:
        messages: 这是一个消息列表，每个消息都是一个字典，包含 role(角色）和 content(内容)。角色可以是 'system'、'user' 或 'assistant’，内容是角色的消息。
        model: 调用的模型，默认为 gpt-4o-mini(ChatGPT)
        temperature: 这决定模型输出的随机程度，默认为0，表示输出将非常确定。增加温度会使输出更随机。
        max_tokens: 这决定模型输出的最大的 token 数。
    """
    # 调用 OpenAI 的ChatCompletion 端点
    response = client.chat.completions.create(
        # 所选模型的ID，如：gpt-4、gpt-3.5-turbo等。注意：不是所有可用的模型都与openai.ChatCompletion兼容
        model=model,
        # 表示对话的消息对象数组。包含两个属性role（可能的值有：system、user、assistant）和content（包含对话消息的字符串）
        messages=messages,
        temperature= temperature,
        max_tokens= max_tokens,
    )
    return response.choices[0].message.content


def get_completion_from_messages_tokens_count(messages, model="gpt-4o-mini", temperature=0, max_tokens = 500):
    """
        封装一个支持更多参数的自定义访问 OpenAI GPT4 的函数
        参数:
            messages: 这是一个消息列表，每个消息都是一个字典，包含 role(角色）和 content(内容)。角色可以是 'system'、'user' 或 'assistant’，内容是角色的消息。
            model: 调用的模型，默认为 gpt-4o-mini(ChatGPT)
            temperature: 这决定模型输出的随机程度，默认为0，表示输出将非常确定。增加温度会使输出更随机。
            max_tokens: 这决定模型输出的最大的 token 数。
        返回:
            content: 生成的回复内容。
            token_dict: 包含'prompt_tokens'、'completion_tokens'和'total_tokens'的字典，分别
            表示提示的 token 数量、生成的回复的 token 数量和总的 token 数量。
    """
    # 调用 OpenAI 的ChatCompletion 端点
    response = client.chat.completions.create(
        # 所选模型的ID，如：gpt-4、gpt-3.5-turbo等。注意：不是所有可用的模型都与openai.ChatCompletion兼容
        model=model,
        # 表示对话的消息对象数组。包含两个属性role（可能的值有：system、user、assistant）和content（包含对话消息的字符串）
        messages=messages,
        temperature= temperature,
        max_tokens= max_tokens,
    )
    content = response.choices[0].message.content
    token_dict = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
    }
    return content, token_dict
