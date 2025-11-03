"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-27 14:32
@Version: v1.0
@Description: 代理
    大型语言模型（LLMs）非常强大，但它们缺乏“最笨”的计算机程序可以轻松处理的特定能力。LLM 对逻辑推理、计算和检索外部信息的能力较弱，这与最简单的计算机程序形成对比。
    如：语言模型无法准确回答简单的计算问题，还有当询问最近发生的事件时，其回答也可能过时或错误，因为无法主动获取最新信息。这是由于当前语言模型仅依赖预训练数据，与外界“断开。

基于上述问题，LangChain 框架提出了 “代理”(Agent) 的解决方案。代理作为语言模型的外部模块，可提供计算、逻辑、检索等功能的支持，使语言模型获得异常强大的推理和获取信息的超能力。
"""
from chatgpt.tool import get_completion
# openai模型
from langchain_community.chat_models import ChatOpenAI

# 初始化llm，参数temperature设置为0.0，从而减少生成答案的随机性
llm = ChatOpenAI(model="gpt-4o-mini",temperature = 0.0)

result = "AI未运行"

# ------------------------ 使用LangChain内置工具llm-math和wikipedia ----------------------
"""
要使用代理 (Agents) ，我们需要三样东西：
    1、一个基本的LLM
    2、将要进行交互的工具Tools
    3、一个控制交互的代理 (Agents) 。
"""
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

# 初始化 工具Tool，工具Tool 包含一个给定工具“名称 name”和 “描述 description” 的实用链
tools = load_tools(
    # llm-math工具结合语言模型和计算器用以进行数学计算
    # wikipedia工具通过API连接到wikipedia进行搜索查询
["llm-math","wikipedia"],
    # 第一步初始化的模型
    llm=llm
)

# 初始化一个简单的代理 (Agents)
agent = initialize_agent(
    # 第二步加载的工具
    tools,
    # 第一步初始化的模型
    llm,
    # 代理类型:
    #   CHAT：代表代理模型为针对对话优化的模型
    #   Zero-shot：意味着代理 (Agents) 仅在当前操作上起作用，即它没有记忆
    #   REACT：代表针对REACT设计的提示模版
    #   DESCRIPTION：根据工具的描述description来决定使用哪个工具）
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # 是否处理解析错误。当发生解析错误时，将错误信息返回给大模型，让其进行纠正
    handle_parsing_errors=True,
    # 是否输出中间步骤结果
    verbose=True
)

"""
Thought: I need to calculate 25% of 300. This can be done by multiplying 300 by 0.25 (which is the decimal equivalent of 25%).
Action:
```
{
    "action": "Calculator",
    "action_input": "300 * 0.25"
}
```
Observation: Answer: 75.0
Thought:I now know the final answer.  
Final Answer: 75.0

过程：
    1、模型思考：我可以使用计算工具来计算300的25%
    2、模型基于思考采取行动：使用计算器（calculator），输入（action_input）300*0.25
    3、模型得到观察：答案: 75.0
    4、基于观察，模型对于接下来需要做什么，给出思考：计算工具返回了300的25%，答案为75
    5、给出最终答案: 300的25%等于75。
    6、以字典的形式给出最终答案
"""
# result = agent("计算300的25%")

"""
Thought: 我需要查找关于秦始皇嬴政的功绩的信息，以便提供一个全面的回答。
Action:
```
{
    "action": "wikipedia",
    "action_input": "秦始皇"
}
```

Observation: Page: Qin Shi Huang
Summary: Qin Shi Huang (Chinese: 秦始皇, Qín Shǐ Huáng, ; February 259 – 12 July 210 BC) was the founder of the Qin dynasty and the first emperor of China. Rather than maintain the title of "king" (wáng 王) borne by the previous Shang and Zhou rulers, he assumed the invented title of "emperor" (huángdì 皇帝), which would see continuous use by monarchs in China for the next two millennia. 
Born in Handan, the capital of Zhao, as Ying Zheng (嬴政) or Zhao Zheng (趙政), his parents were King Zhuangxiang of Qin and Lady Zhao. The wealthy merchant Lü Buwei assisted him in succeeding his father as the king of Qin, after which he became King Zheng of Qin (秦王政). By 221 BC, he had conquered all the other warring states and unified all of China, and he ascended the throne as China's first emperor. During his reign, his generals greatly expanded the size of the Chinese state: campaigns south of Chu permanently added the Yue lands of Hunan and Guangdong to the Sinosphere, and campaigns in Inner Asia conquered the Ordos Plateau from the nomadic Xiongnu, although the Xiongnu later rallied under Modu Chanyu.
Qin Shi Huang also worked with his minister Li Si to enact major economic and political reforms aimed at the standardization of the diverse practices among earlier Chinese states. He is traditionally said to have banned and burned many books and executed scholars. His public works projects included the incorporation of diverse state walls into a single Great Wall of China and a massive new national road system, as well as his city-sized mausoleum guarded by a life-sized Terracotta Army. He ruled until his death in 210 BC, during his fifth tour of eastern China.
Qin Shi Huang has often been portrayed as a tyrant and strict Legalist—characterizations that stem partly from the scathing assessments made during the Han dynasty that succeeded the Qin. Since the mid-20th century, scholars have begun questioning this evaluation, inciting considerable discussion on the actual nature of his policies and reforms. According to the sinologist Michael Loewe "few would contest the view that the achievements of his reign have exercised a paramount influence on the whole of China's subsequent history, marking the start of an epoch that closed in 1911."



Page: Mausoleum of Qin Shi Huang
Summary: The Mausoleum of Qin Shi Huang (Chinese: 秦始皇陵; pinyin: Qínshǐhuáng Líng) is a tomb complex constructed for Qin Shi Huang, the first emperor of the Chinese Qin dynasty. It is located in modern-day Lintong District in Xi'an, Shaanxi. It was constructed over 38 years from 246 to 208 BC, and is situated underneath a 76-metre-tall (249 ft) tomb mound shaped like a truncated pyramid. The layout of the mausoleum is modeled on the layout of Xianyang, the capital of the Qin dynasty, which was divided into inner and outer cities. The circumference of the inner city is 2.5 km (1.6 mi) and the outer is 6.3 km (3.9 mi). The tomb is located in the southwest of the inner city and faces east. The main tomb chamber housing the coffin and burial artifacts is the core of the architectural complex of the mausoleum.
The tomb itself has not yet been excavated.  Archaeological explorations currently concentrate on various sites of the extensive necropolis surrounding the tomb, including the Terracotta Army to the east of the tomb mound. The Terracotta Army served as a garrison to the mausoleum and has yet to be completely excavated.

Page: History of China
Summary: The history of China spans several millennia across a wide geographical area. Each region now considered part of the Chinese world has experienced periods of unity, fracture, prosperity, and strife. Chinese civilization first emerged in the Yellow River valley, which along with the Yangtze basin constitutes the geographic core of the Chinese cultural sphere. China maintains a rich diversity of ethnic and linguistic people groups. The traditional lens for viewing Chinese history is the dynastic cycle: imperial dyna
Thought:我已经找到关于秦始皇嬴政的功绩的信息。他是中国历史上第一位皇帝，统一了中国，并进行了许多重要的改革和建设。接下来，我将总结他的主要功绩。

1. **统一中国**：秦始皇在公元前221年完成了对六国的征服，结束了长期的战国时期，建立了中央集权的秦朝，成为中国历史上第一位皇帝。

2. **标准化改革**：他与李斯合作，实施了经济和政治的重大改革，标准化了度量衡、货币和书写系统，促进了全国的经济交流和文化统一。

3. **修建长城**：秦始皇将各个国家的防御工事整合成了统一的长城，以抵御北方游牧民族的侵扰。

4. **公共工程**：他还修建了广泛的道路系统，促进了交通和贸易的发展。

5. **兵马俑**：秦始皇的陵墓中有著名的兵马俑，这些陶俑是为了保护他在来世的安全，展示了当时的工艺水平。

6. **法律与行政**：他推行法家思想，建立了严格的法律体系，强化了中央集权。

7. **文化政策**：虽然他被指责焚书坑儒，但他的统治也促进了文化的统一和传播。

秦始皇的统治对后世产生了深远的影响，标志着中国历史的新纪元。

Final Answer: 秦始皇嬴政的主要功绩包括统一中国、实施标准化改革、修建长城、进行公共工程、建立兵马俑、强化法律与行政以及推动文化统一。

过程：
    1、模型思考：我应该使用维基百科去搜索
    2、模型基于思考采取行动：使用维基百科，输入秦始皇、History of China
    3、模型得到观察： 页面：Qin Shi Huang、Mausoleum of Qin Shi Huang
    4、基于观察，模型对于接下来需要做什么，给出思考：他是中国历史上第一位皇帝，统一了中国，并进行了许多重要的改革和建设
    5、给出最终答案: 秦始皇嬴政的主要功绩包括统一中国、实施标准化改革、修建长城、进行公共工程、建立兵马俑、强化法律与行政以及推动文化统一。
    6、以字典的形式给出最终答案
"""
question = "秦始皇嬴政是一位大一统开国皇帝，他都有哪些功绩？"
# result = agent(question)

# ------------------------ 使用LangChain内置工具PythonREPLTool ----------------------
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool

# 创建一个能将顾客名字转换为拼音的 python 代理
pinyin_agent = create_python_agent(
    # 已经加载的大语言模型
    llm,
    # 使用Python交互式环境工具 REPLTool
    tool=PythonREPLTool(),
    # 输出中间步骤
    verbose=True
)
customer_list = ["小明","小黄","小红","小蓝","小橘","小绿",]
"""
需要提前安装pinyin库： conda install pinyin

To convert the given Chinese names to Pinyin, I will use the `pinyin` library in Python. I need to install the library first if it's not already available. Then, I will convert the list of names to Pinyin and print the output.

Action: Python_REPL  
Action Input: `!pip install pinyin`  
Observation: SyntaxError('invalid syntax', ('<string>', 1, 1, '!pip install pinyin\n', 1, 2))
Thought:I cannot use pip commands directly in this environment. Instead, I will assume the `pinyin` library is available and proceed to convert the names to Pinyin.

Action: Python_REPL  
Action Input: `import pinyin; names = ['小明', '小黄', '小红', '小蓝', '小橘', '小绿']; pinyin_names = [pinyin.get(name) for name in names]; print(pinyin_names)`  
Observation: ['xiǎomíng', 'xiǎohuáng', 'xiǎohóng', 'xiǎolán', 'xiǎojú', 'xiǎolv̀']

Thought:I now know the final answer
Final Answer: ['xiǎomíng', 'xiǎohuáng', 'xiǎohóng', 'xiǎolán', 'xiǎojú', 'xiǎolv']

过程：
    1、模型思考：我应该使用维基百科去搜索
        1.1、[chain/start] AgentExecutor：Entering Chain run with input
        1.2、[chain/start] AgentExecutor > LLMChain：Entering Chain run with input
        1.3、[llm/start] AgentExecutor > LLMChain > ChatOpenAI：Entering LLM run with input
        1.4、[llm/end] AgentExecutor > LLMChain > ChatOpenAI：Entering LLM run with output
        1.5、[chain/end] AgentExecutor > LLMChain：Entering Chain run with output
    2、模型基于思考采取行动：这里输出的为python代码 import pinyin
        2.1、[toll/start] AgentExecutor > Python_REPL：Entering Tool run with input
        2.2、[tool/end] AgentExecutor > Python_REPL：Exiting Tool run with output
    3、模型得到观察： [chain/start] AgentExecutor > LLMChain : Exiting LLM run with input
    4、基于观察，模型对于接下来需要做什么，给出思考：
        4.1、[llm/start] AgentExecutor > LLMChain > ChatOpenAI : Exiting LLM run with input
        4.2、[llm/end] AgentExecutor > LLMChain > ChatOpenAI : Exiting LLM run with output
    5、给出最终答案：[chain/end] AgentExecutor > LLMChain：Exiting Chain run with output
    6、返回最终答案: [chain/end] AgentExecutor：Exiting Chain run with output
    
"""
# result = pinyin_agent.run(f"将使用pinyin拼音库这些客户名字转换为拼音，并打印输出列表: {customer_list}。")

# ------------------------ 定义自己的工具并在代理中使用 ----------------------
"""
创建和使用自定义时间工具：
    LangChian tool 函数装饰器可以应用用于任何函数，将函数转化为LangChain 工具，使其成为代理可调用的工具。
需要给函数加上非常详细的文档字符串使，得代理知道在什么情况下、如何使用该函数/工具
"""
# 导入tool函数装饰器
from langchain.agents import tool
from datetime import date


@tool
def time(text: str) -> str:
    """
        自定义时间工具：
        返回今天的日期，用于任何需要知道今天日期的问题。输入应该总是一个空字符串，这个函数将总是返回今天的日期，任何日期计算应该在这个函数之外进行。
    """
    return str(date.today())


# 初始化代理
customized_agent = initialize_agent(
    # 使用自定义的时间工具加入代理
    tools=[time],
    # 初始化的模型
    llm=llm,
    # 代理类型
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # 处理解析错误
    handle_parsing_errors=True,
    # 输出中间步骤
    verbose = True
)

"""
使用代理询问今天的日期

Thought: 我需要获取今天的日期。
Action:
```
{
    "action": "time",
    "action_input": ""
}
```

Observation: 2025-10-27
Thought:我现在知道今天的日期是2025年10月27日。
Final Answer: 今天的日期是2025年10月27日。

过程：
    1、模型思考：我需要使用 time 工具来获取今天的日期
    2、模型基于思考采取行动：Action的输出因使用工具的不同而有所不同，使用time工具，输入为空字符串
    3、模型得到观察： 2025-10-27
    4、基于观察，模型对于接下来需要做什么，给出思考：我现在知道今天的日期是2025年10月27日。
    5、给出最终答案: 今天的日期是2025年10月27日。
    6、以字典的形式给出最终答案：{‘input': xxx, 'output': xxx}

"""
# result = customized_agent("今天的日期是？")
print(result)


