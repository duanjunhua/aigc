"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-28 17:16
@Version: v1.0
@Description: 向量数据库与词向量

    检索增强生成整体流程：
        Document Loading[PDF、Database、URLs]    →   Splitting[Splits]   →   Storage[Vectorstore]     →       Retrieval[Question & Relevant Splits]   →   Output[Prompt -> LLM]     →    Answer
"""
# 导入OpenAI参数
from chatgpt.tool import get_completion

# ------------------------ 文档读取 ----------------------
from langchain_community.document_loaders import PyPDFLoader

# 加载 PDF
loaders_chinese = [
    PyPDFLoader("车路云一体化发展：架构、技术与产业全景.pdf"),
    PyPDFLoader("matplotlib/第一回：Matplotlib初相识.pdf"),
    PyPDFLoader("matplotlib/第二回：艺术画笔见乾坤.pdf"),
    PyPDFLoader("matplotlib/第三回：布局格式定方圆.pdf")
]
docs = []
for loader in loaders_chinese:
    docs.extend(loader.load())
# print(docs)

# 分割文本，使用 RecursiveCharacterTextSplitter (递归字符文本拆分器)来创建块
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # 每个文本块的大小。这意味着每次切分文本时，会尽量使每个块包含 1500个字符。
    chunk_size=1500,
    # 每个文本块之间的重叠部分。
    chunk_overlap=150
)
splits = text_splitter.split_documents(docs)
print("文本块长度：", len(splits))

# ------------------------ Embeddings ----------------------
"""
    在机器学习和自然语言处理（NLP）中， Embeddings （嵌入）是一种将类别数据，如单词、句子或者整个文档，转化为实数向量的技术。这些实数向量可以
被计算机更好地理解和处理。嵌入背后的主要想法是，相似或相关的对象在嵌入空间中的距离应该很近。
    例如：使用词嵌入（word embeddings）来表示文本数据，在词嵌入中，每个单词被转换为一个向量，这个向量捕获了这个单词的语义信息
        1、"king" 和 "queen" 这两个单词在嵌入空间中的位置将会非常接近，因为它们的含义相似。
        2、"apple" 和 "orange" 也会很接近，因为它们都是水果
"""
import numpy as np

# 对切块进行 Embedding 处理
from langchain_openai.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# 测试句子相似案例
def test_embedd_relations(embedding):
    sentence1_chinese = "我喜欢狗"
    sentence2_chinese = "我喜欢犬科动物"
    sentence3_chinese = "外面的天气很糟糕"
    embedding1_chinese = embedding.embed_query(sentence1_chinese)
    embedding2_chinese = embedding.embed_query(sentence2_chinese)
    embedding3_chinese = embedding.embed_query(sentence3_chinese)
    # 使用点积来比较两个嵌入，分数越高句子越相似
    result12 = np.dot(embedding1_chinese, embedding2_chinese)
    result23 = np.dot(embedding2_chinese, embedding3_chinese)
    print(f"【我喜欢狗】与【我喜欢犬科动物】的相似分数：{result12}，【我喜欢犬科动物】与【外面的天气很糟糕】的相似分数：{result23}")

# 测试向量化结果
# test_embedd_relations(embedding=embeddings)


# ------------------------ Vectorstores ----------------------
# ============= 初始化Chroma =============
"""
Langchain集成了超过30个不同的向量存储库。此处选择Chroma是因为它轻量级且数据存储在内存中，这使得它非常容易启动和开始使用。
"""
import os
import shutil

# 导入Chroma向量库，chromadb版本要求0.5.4
from langchain_community.vectorstores import Chroma

# 指定一个持久化路径
persist_directory_chinese = "chroma"

# 文件若已存在则删除
if os.path.exists(os.path.abspath(persist_directory_chinese)):
    shutil.rmtree(os.path.abspath(persist_directory_chinese))

# 从已加载的文档中创建一个向量数据库
vectordb = Chroma.from_documents(
    # 加载的切块
    documents=splits,
    # 向量化模型
    embedding=embeddings,
    # 允许将persist_directory目录保存到磁盘上
    persist_directory=persist_directory_chinese
)

# 数据库长度
print("数据库长度：", vectordb.__len__())

# 持久化向量数据库
vectordb.persist()

# ============= 相似性搜索(Similarity Search) =============
# 定义一个需要检索答案的问题
question = "车路云一体化系统是什么？"

# 调用已加载的向量数据库根据相似性检索答案
docs_chinese = vectordb.similarity_search(
    # 需要检索的问题
    question,
    # 返回的结果集数量，默认为4个
    k=2
)

print(docs_chinese[0].page_content)



