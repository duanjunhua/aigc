"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-13 17:08
@Version: v1.0
@Description: 文本转换：大语言模型具有强大的文本转换能力，可以实现多语言翻译、拼写纠正、语法调整、格式转换等不同类型的文本转换任务
    此处将介绍如何通过编程调用API接口，使用语言模型实现文本转换功能
"""
from IPython.display import display, Markdown, Latex, HTML, JSON
import time
from chatgpt.tool import get_completion

# 引入 Redlines 包，详细显示并对比纠错过程
from redlines import Redlines
from IPython.display import display, Markdown

result = "未运行AI"

# ------------------------ 一、文本翻译 ----------------------------------
# 通过在大规模高质量平行语料上进行 Fine-Tune，大语言模型可以深入学习不同语言间的词汇、语法、语义等层面的对应关系，
# 模拟双语者的转换思维，进行意义传递的精准转换，而非简单的逐词替换

# 语言翻译
prompt = f'''
将以下中文翻译成英语：
```你好，欢迎学习大模型```
'''

# 识别语言
prompt_check_lang = f'''
请告诉我以下文本是什么语种:
```Combien coûte le lampadaire?```
'''

# 语气转换
prompt_trans_tone = f"""
请将以下文本翻译成中文，分别展示成正式与非正式两种语气:
```Would you like to order a pillow?```
"""

# 语气与写作风格调整：比如工作邮件需要使用正式、礼貌的语气和书面词汇；而与朋友的聊天可以使用更轻松、口语化的语气
prompt_emoj = f'''
将以下文本翻译成商务信函的格式:
```小老弟，我小羊，上回你说咱部门要采购的显示器是多少寸来着？```
'''

# 文件格式转换：大语言模型如 ChatGPT 可在不同数据格式之间转换，可以轻松实现 JSON 到 HTML、XML、Markdown 等格式的相互转化
data_json = {"resturant employees": [
    {"name": "Shyam", "email": "shyamjaiswal@gmail.com"},
    {"name": "Bob", "email": "bob32@gmail.com"},
    {"name": "Jai", "email": "jai87@gmail.com"}
]}

prompt_format_data = f'''
将以下Python字典从JSON转换为HTML表格，保留表格标题和列名：{data_json}
'''
# print(HTML(result))

# 利用大语言模型进行自动校对可以极大地降低人工校对的工作量，如使用大语言模型检查句子的拼写和语法错误
text = [
    "我们去踢足球一起.",   # The girl has a ball.
    "我看书正在.",    # ok
    "一起去逛商场怎么样?",  # Homonyms
    "我很高兴今天",   # Homonyms
]
for i in range(len(text)):
    prompt = f"""请校对并更正以下文本，注意纠正文本保持原始语种，无需输出原始文本。如果您没有发现任何错误，请说“未发现错误”。
    ```{text[i]}```"""
    # response = get_completion(prompt)
    # print(i, response)

# 大语言模型进行语法纠错
error_grammar_txt = f'''
事实证语法明错误并不影响会阅读效果
'''
prompt_grammar = f"校对并更正以下商品评论：```{error_grammar_txt}```"


# diff = Redlines(error_grammar_txt, result)
# print(Markdown(diff.output_markdown))

# 大语言模型中的 “温度”(temperature) 参数可以控制生成文本的随机性和多样性。temperature 的值越大，
# 语言模型输出的多样性越大；temperature 的值越小，输出越倾向高概率的文本。
# 一般来说，如果需要可预测、可靠的输出，则将 temperature 设置为0

# 温度（temperature）参数可以控制语言模型生成文本的随机性。温度为0时，每次使用同样的Prompt，得到的结果总是一致的
result = get_completion(prompt_emoj, temperature=0)
print(result)

