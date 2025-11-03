"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-23 17:28
@Version: v1.0
@Description: 评估
    评估是检验语言模型问答质量的关键环节。评估可以检验语言模型在不同文档上的问答效果，找出其弱点。还可以通过比较不同模型，选择最佳系统。
    此外，定期评估也可以检查模型质量的衰减。评估通常有两个目的：
        1. 检验LLM应用是否达到了验收标准
        2. 分析改动对于LLM应用性能的影响
    思路就是利用语言模型本身和链本身，来辅助评估其他的语言模型、链和应用程序
"""
from chatgpt.tool import get_completion
# openai模型
from langchain_community.chat_models import ChatOpenAI

# 初始化llm
llm = ChatOpenAI(model="gpt-4o-mini",temperature = 0.0)

result = "AI未运行"

# ------------------------ 创建LLM应用 ----------------------
# 检索QA链，在文档上进行检索
from langchain.chains import RetrievalQA
# 文档加载器，采用csv格式存储
from langchain_community.document_loaders import CSVLoader
# 导入向量存储索引创建器
from langchain.indexes import VectorstoreIndexCreator
# 向量存储
from langchain.vectorstores import DocArrayInMemorySearch

# 加载中文数据
file = 'ProductData.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')
data = loader.load()

# 查看数据
import pandas as pd
test_data = pd.read_csv(file,skiprows=0)
# print(test_data)

from langchain_community.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 将指定向量存储类,创建完成后，我们将从加载器中调用,通过文档记载器列表加载
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch, embedding=embeddings).from_loaders([loader])

# 通过指定语言模型、链类型、检索器和我们要打印的详细程度来创建检索QA链
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

# 设置测试的数据
examples = [
    {
        "query": "高清电视机怎么进行护理？",
        "answer": "使用干布清洁。"
    },
    {
        "query": "旅行背包有内外袋吗？",
        "answer": "有。"
    }
]

# 通过LLM生成测试用例
"""
Langchain提供的方法： QAGenerateChain，可以通过QAGenerateChain来为我们的文档自动创建问答集
"""
# 导入QA生成链，它将接收文档，并从每个文档中创建一个问题答案对
from langchain.evaluation.qa import QAGenerateChain

# 由于 QAGenerateChain 类中使用的 PROMPT 是英文，故我们继承 QAGenerateChain 类，将PROMPT加上“请使用中文输出”
from langchain.output_parsers.regex import RegexParser
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from typing import Any

template = """You are a teacher coming up with questions to ask on a quiz. Given the following document, please generate a question and answer based on that document.

Example Format:
<Begin Document>
...
<End Document>
QUESTION: question here
ANSWER: answer here

These questions should be detailed and be based explicitly on information in the document. Begin!

<Begin Document>
{doc}
<End Document>
请使用中文输出
"""

output_parser = RegexParser(
    regex=r"QUESTION: (.*?)\nANSWER: (.*)", output_keys=["query", "answer"]
)
PROMPT = PromptTemplate(
    input_variables=["doc"], template=template, output_parser=output_parser
)


# 继承QAGenerateChain
class ChineseQAGenerateChain(QAGenerateChain):
    """LLM Chain specifically for generating examples for question answering."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> QAGenerateChain:
        """Load QA Generate Chain from LLM."""
        return cls(llm=llm, prompt=PROMPT, **kwargs)


# #通过传递chat open AI语言模型来创建这个链
example_gen_chain = ChineseQAGenerateChain.from_llm(ChatOpenAI())
# 应用了 QAGenerateChain 的 apply 方法对 data 中的前5条数据创建了2个“问答对”
new_examples = example_gen_chain.apply([{"doc": t} for t in data[:2]])

# [{'qa_pairs': {'query': '这个产品的名称是什么？', 'answer': '全自动咖啡机'}}, {'qa_pairs': {'query': '这个产品的名称是什么？描述中提到的规格是什么？', 'answer': '产品的名称是"电动牙刷"。描述中提到的规格是一般大小，高度为9.5英寸，宽度为1英寸。'}}]
# print(new_examples)

# 将之前手动创建的问答集合并到QAGenerateChain 创建的问答集中，这样在答集中既有手动创建的例子又有 llm 自动创建的例子，这会使我们的测试集更加完善
examples += [ v for item in new_examples for k,v in item.items()]
# result = qa.run(examples[0]["query"])
# print(result)

# ------------------------ 通过LLM进行评估实例 ----------------------
# 为所有不同的示例创建预测
predictions = qa.apply(examples)
# 对预测的结果进行评估，导入QA问题回答，评估链，通过语言模型创建此链
from langchain.evaluation.qa import QAEvalChain #导入QA问题回答，评估链
# 通过调用chatGPT进行评估
eval_chain = QAEvalChain.from_llm(llm)
# 在此链上调用evaluate，进行评估
graded_outputs = eval_chain.evaluate(examples, predictions)

# 传入示例和预测，得到一堆分级输出，循环遍历它们打印答案
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['results'])
    print()

"""
LangChain 使整个评估流程自动化。它可以自动构建包含问答样本的测试集，然后使用语言模型对测试集自动产生回复，最后通过另一个模型链自动判断每个回答的准确性。
这种全自动的评估方式极大地简化了问答系统的评估和优化过程，开发者无需手动准备测试用例，也无需逐一判断正确性，大大提升了工作效率。

借助LangChain的自动评估功能，可以快速评估语言模型在不同文档集上的问答效果，并可以持续地进行模型调优，无需人工干预。这种自动化的评估方法解放了双手，
使我们可以更高效地迭代优化问答系统的性能。
"""


