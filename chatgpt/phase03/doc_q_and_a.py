"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-22 16:40
@Version: v1.0
@Description: 基于文档的问答
        使用大语言模型构建一个能够回答关于给定文档和文档集合的问答系统是一种非常实用和有效的应用场景。与仅依赖模型预训练知识不
    同，这种方法可以进一步整合用户自有数据，实现更加个性化和专业的问答服务。语言模型不仅利用了自己的通用知识，还可以充分运用
    外部输入文档的专业信息来回答用户问题，显著提升答案的质量和适用性。

    基于文档问答的这个过程，我们会涉及 LangChain 中的其他组件，比如：嵌入模型（EmbeddingModels)和向量储存(Vector Stores)
"""
from chatgpt.tool import get_completion
# openai模型
from langchain_community.chat_models import ChatOpenAI

result = "AI未运行"

# 初始化llm
llm = ChatOpenAI(model="gpt-4o-mini",temperature = 0.0)

# ------------------------ 直接使用向量储存查询 ----------------------
# 检索QA链，在文档上进行检索
from langchain.chains import RetrievalQA
# 文档加载器，采用csv格式存储
from langchain_community.document_loaders import CSVLoader
# 向量存储
from langchain_community.vectorstores import DocArrayInMemorySearch
# 在jupyter显示信息的工具
from IPython.display import display, Markdown
import pandas as pd

# ===== 数据导入 =====
# 数据是字段为 name 和 description 的文本数据
file = 'DataQA.csv'
# 使用langchain文档加载器对数据进行导入
loader = CSVLoader(file_path=file, encoding='utf-8')
# 使用pandas导入数据，用以查看
data = pd.read_csv(file, usecols=[1, 2])
# print(data)
# print(data.head())

# ===== 基本文档加载器创建向量存储 =====
# 导入向量存储索引创建器
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建指定向量存储类, 创建完成后，从加载器中调用, 通过文档加载器列表加载
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch, embedding=embeddings).from_loaders([loader])

query="请用markdown表格的方式列出所有具有防晒功能的衬衫，对每件衬衫描述进行总结"

# 使用索引查询创建一个响应，并传入这个查询
# response = index.query(query, llm=llm)
# display(Markdown(response))

# ------------------------ 结合表征模型和向量存储 ----------------------
"""
    由于语言模型的上下文长度限制，直接处理长文档具有困难。为实现对长文档的问答，我们可以引入向量嵌入(Embeddings)和向量存储(Vector Store)等技术：
        一、使用文本嵌入(Embeddings)算法对文档进行向量化，使语义相似的文本片段具有接近的向量表示
        二、将向量化的文档切分为小块，存入向量数据库。这个流程正是创建索引(index)的过程
    向量数据库对各文档片段进行索引，支持快速检索。当用户提出问题时，可以先将问题转换为向量，在数据库中快速找到语义最相关的文档片段。
然后将这些文档片段与问题一起传递给语言模型，生成回答。
    通过嵌入向量化和索引技术，实现对长文档的切片检索和问答。这种流程克服了语言模型的上下文限制，可以构建处理大规模文档的问答系统。
"""

# ===== 数据导入 =====
loader = CSVLoader(file_path=file, encoding='utf-8')
docs = loader.load()

# 查看单个文档，每个文档对应于CSV中的一行数据
# print(docs[0])

# ===== 文本向量表征模型 =====
# 因为文档比较短，此处不进行任何分块, 可以直接进行向量表征。使用初始化OpenAIEmbedding实例上的查询方法embed_query为文本创建向量表征
embed = embeddings.embed_query("你好呀，我的名字叫小可爱")
# 查看得到向量表征的长度
# print("\n\033[32m向量表征的长度: \033[0m \n", len(embed))

# 每个元素都是不同的数字值，组合起来就是文本的向量表征
# print("\n\033[32m向量表征前5个元素: \033[0m \n", embed[:5])

# ===== 基于向量表征创建并查询向量存储 =====
# 将创建文本向量表征(embeddings)存储在向量存储(vector store)中，使用DocArrayInMemorySearch类的from_documents方法来实现，
# 该方法接受文档列表以及向量表征模型作为输入
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

query = "请推荐一件具有防晒功能的衬衫"
# 使用上面的向量存储来查找与传入查询类似的文本，得到一个相似文档列表
docs = db.similarity_search(query)
# print("\n\033[32m返回文档的个数: \033[0m \n", len(docs))
# print("\n\033[32m第一个文档: \033[0m \n", docs[0])

# ===== 使用查询结果构造提示来回答问题 =====
# 合并获得的相似文档内容
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
# 将合并的相似文档内容后加上问题（question）输入到 `llm.call_as_llm`中
# result = llm.invoke(f"{qdocs}问题：请用markdown表格的方式列出所有具有防晒功能的衬衫，对每件衬衫描述进行总结")
# print(result.content)

# ===== 使用检索问答链来回答问题 =====
"""
通过LangChain创建一个检索问答链，对检索到的文档进行问题回答。检索问答链的输入包含以下：
    llm：语言模型，进行文本生成
    chain_type：传入链类型，此处使用stuff，将所有查询得到的文档组合成一个文档传入下一步。其他的方式包括：
        Map Reduce： 将所有块与问题一起传递给语言模型，获取回复，使用另一个语言模型调用将所有单独的回复总结成最终答案，它可以在任意数量的文档上运行。可以并行处理单个问题，同时也需要更多的调用。它将所有文档视为独立的
        Refine： 用于循环许多文档，际上是迭代的，建立在先前文档的答案之上，非常适合前后因果信息并随时间逐步构建答案，依赖于先前调用的结果。它通常需要更长的时间，并且基本上需要与Map Reduce一样多的调用
        Map Re-rank： 对每个文档进行单个语言模型调用，要求它返回一个分数，选择最高分，这依赖于语言模型知道分数应该是什么，需要告诉它，如果它与文档相关，则应该是高分，并在那里精细调整说明，可以批量处理它们相对较快，但是更加昂贵
    retriever：检索器
"""
# 基于向量储存，创建检索器
retriever = db.as_retriever()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
# 创建一个查询并在此查询上运行链
query = "请用markdown表格的方式列出所有具有防晒功能的衬衫，对每件衬衫描述进行总结"
result = qa_stuff.run(query)
print(result)

