"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-30 16:51
@Version: v1.0
@Description: 检索(Retrieval)
    在构建检索增强生成 (RAG) 系统时，信息检索是核心环节。检索模块负责对用户查询进行分析，从知识库中快速定位相关文档或段落，为后续的语言生成
提供信息支持。检索是指根据用户的问题去向量数据库中搜索与问题相关的文档内容，当我们访问和查询向量数据库时可能会运用到如下几种技术：
    1、基本语义相似度(Basic semantic similarity)
    2、最大边际相关性(Maximum marginal relevance，MMR)
    3、过滤元数据
    4、LLM辅助检索
    使用基本的相似性搜索大概能解决80%的相关检索工作

    检索技术：
        Question Query  ->     Storage[Vector Store]    ->      Retrieval[Relevant Splits]      ->      Output[Prompt & LLM]    -> Answer
"""
# 导入OpenAI参数
from chatgpt.tool import get_completion

result = "向量库未运行"

# ------------------------ 向量数据库检索 ----------------------
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

# 指定一个持久化路径
persist_directory_chinese = "chroma"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载向量数据库( VectorDB )
vectordb = Chroma(
    # 向量库路径
    persist_directory=persist_directory_chinese,
    # 向量化函数
    embedding_function=embeddings,
)
# print(vectordb._collection.count())

# ============= 相似性检索（Similarity Search） =============
# 将句子存入向量库
sentence_txt = [
    "毒鹅膏菌（Amanita phalloides）具有大型且引人注目的地上（epigeous）子实体（basidiocarp）",
    "一种具有大型子实体的蘑菇是毒鹅膏菌（Amanita phalloides）。某些品种全白。",
    "A. phalloides，又名死亡帽，是已知所有蘑菇中最有毒的一种。"
]

# sentence_db = Chroma.from_texts(texts=sentence_txt, embedding=embeddings)

question = "告诉我关于具有大型子实体的全白色蘑菇的信息"

"""
进行相似性搜索，similarity_search（相似性搜索） 方法可以根据问题的语义去数据库中搜索与之相关性最高的文档
    query：检索问题
    k：设置 k=2 ，只返回两个最相关的文档
"""
# result = sentence_db.similarity_search(query=question, k=2)

# ============= 解决多样性：最大边际相关性(MMR) =============
"""
    最大边际相关模型 (MMR，Maximal Marginal Relevance) 是实现多样性检索的常用算法：
        1、Query the Vector Store
        2、Choose the 'fetch_k' most similar responses
        3、Within those responses the 'k' most diverse

    MMR 的基本思想是同时考量查询与文档的相关度，以及文档之间的相似度。它计算每个候选文档与查询的相关度，并减去与已经选入结果集的文档的相似度。这样更不相似的文档会有更高的得分
        相关度：确保返回结果对查询高度相关
        相似度：鼓励不同语义的文档被包含进结果集
MMR 是解决检索冗余问题、提供多样性结果的一种简单高效的算法。它平衡了相关性和多样性，适用于对多样信息需求较强的应用场景。
"""
"""
设置 fetch_k 参数，用来告诉向量数据库我们最终需要 k 个结果返回，fetch_k=3 ，也就是我们最初获取 3 个文档
k=1 表示返回最不同的1个文档
"""
# result = sentence_db.max_marginal_relevance_search(query=question, k=2, fetch_k=3)
"""
当向量数据库中存在相同的文档时，而用户的问题又与这些重复的文档高度相关时，向量数据库会出现返回重复文档的情况，可以运用Langchain的 max_marginal_relevance_search 来解决这个问题
"""

# ============= 解决特殊性：使用元数据 =============
"""
当我们向向量数据库提出问题时，数据库并没有很好的理解问题的语义，返回的结果就会不如预期。要解决这个问题，我们可以通过过滤元数据的方式来实现精准搜索，当前很多向量数据库都支持对 元数据（metadata） 的操作。
metadata 为每个嵌入的块(embedded chunk)提供上下文
"""
question = "他们在第二讲中对Figure说了些什么？"
# docs = vectordb.similarity_search(query=question, k=2, filter={"source": "matplotlib/第二回：艺术画笔见乾坤.pdf"})
# for doc in docs:
#     print(doc.metadata)

# ============= 解决特殊性：在元数据中使用自查询检索器（LLM辅助检索） =============
"""
    LangChain提供了SelfQueryRetriever模块，它可以通过语言模型从问题语句中分析出：
        1、向量搜索的查询字符串(search term)
        2、过滤文档的元数据条件(Filter)
    它使用语言模型自动解析语句语义,提取过滤信息,无需手动设置。这种基于理解的元数据过滤更加智能方便,可以自动处理更复杂的过滤逻辑。
    
利用语言模型实现自动化过滤的技巧,可以大幅降低构建针对性问答系统的难度。这种自抽取查询的方法使检索更加智能和动态。
"""
# openai模型
from langchain_openai.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 初始化llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

"""
定义metadata_field_info，包含了元数据的过滤条件 source 和 page , 其中 source 的作用是告诉 LLM 我们想要的数据来自于哪里， page 告诉 LLM 我们需要提取相关的内容在原始文档的哪一页
LLM会自动从用户的问题中提取出 Filter 和 Search term 两项，然后向量数据库基于这两项去搜索相关的内容
"""
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `matplotlib/第一回：Matplotlib初相识.pdf`, `matplotlib/第二回：艺术画笔见乾坤.pdf`, or `matplotlib/第三回：布局格式定方圆.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer"
    ),
]
document_content_description = "Matplotlib课堂讲义"
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)
question = "他们在第二讲中对Figure做了些什么？"

# results = retriever.invoke(input=question)
# for res in results:
#     print(res.metadata)

# ============= 其他技巧：压缩 =============
"""
LangChain提供了一种“压缩”检索机制。其工作原理是，先使用标准向量检索获得候选文档，然后基于查询语句的语义，使用语言模型压缩这些文档,只保留与问题相关的部分。
    如：
        对“蘑菇的营养价值”这个查询，检索可能返回整篇有关蘑菇的长文档。经压缩后，只提取文档中与“营养价值”相关的句子
        
    当向量数据库返回了所有与问题相关的所有文档块的全部内容后，会有一个Compression LLM来负责对这些返回的文档块的内容进行压缩，
所谓压缩是指仅从文档块中提取出和用户问题相关的内容，并舍弃掉那些不相关的内容。

    这种压缩可以有效提升输出质量，节省通过长文档带来的计算资源浪费，降低成本。上下文相关的压缩检索技术，使得到的支持文档更严格匹配问题需求，是提升问答系统效率的重要手段。
"""
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + doc.page_content for i, doc in enumerate(docs)]))


# 压缩器，负责从向量数据库返回的文档块中提取相关信息
compressor = LLMChainExtractor.from_llm(llm)

# base_compressor为压缩器，base_retriever是定义的 vectordb 产生的检索器
compression_retriever_chinese = ContextualCompressionRetriever(
    base_compressor=compressor,
    # 可以指定搜索类型，如：vectordb.as_retriever(search_type="mmr")，默认是similarity，mmr搜索类型可以过滤结果集，使其中不包含任何重复的信息
    base_retriever=vectordb.as_retriever(search_type="mmr")
)

# 对源文档进行压缩
question = "Matplotlib是什么？"
# compressed_docs_chinese = compression_retriever_chinese.invoke(question)
# pretty_print_docs(compressed_docs_chinese)

"""
vetordb 并不是唯一一种检索文档的工具。 LangChain 还提供了其他检索文档的方式，如：TF-IDF 或 SVM，但是两者的检索效果不如VectorDB
"""
from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载PDF
loader_chinese = PyPDFLoader("matplotlib/第一回：Matplotlib初相识.pdf")
pages_chinese = loader_chinese.load()
all_page_text_chinese = [p.page_content for p in pages_chinese]
joined_page_text_chinese = " ".join(all_page_text_chinese)

# 分割文本
text_splitter_chinese = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits_chinese = text_splitter_chinese.split_text(joined_page_text_chinese)
# 检索
svm_retriever = SVMRetriever.from_texts(splits_chinese, embeddings)
tfidf_retriever = TFIDFRetriever.from_texts(splits_chinese)

svm_docs = svm_retriever.invoke(question)
for svm_doc in svm_docs:
    print(f"SVM检索：{svm_doc}")

print("===================")
tfidf_docs = tfidf_retriever.invoke(question)
for tfidf_doc in tfidf_docs:
    print(f"TFIDF检索：{tfidf_doc}")
