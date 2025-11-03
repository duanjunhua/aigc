"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-28 14:47
@Version: v1.0
@Description: 文档分割
    为什么要进行文档分割？
        1、模型大小和内存限制：GPT模型，特别是大型版本如 GPT-3 或 GPT-4 ，具有数十亿甚至上百亿的参数。在一次前向传播中处理这么多的参数，需要大量的计算能力和内存。
    但大多数硬件设备（例如 GPU 或 TPU ）有内存限制，文档分割使模型能够在这些限制内工作。
        2、计算效率：处理更长的文本序列需要更多的计算资源。通过将长文档分割成更小的块，可以更高效地进行计算。
        3、序列长度限制：GPT 模型有一个固定的最大序列长度，例如2048个 token 。这意味着模型一次只能处理这么多 token 。对于超过这个长度的文档，需要进行分割才能被模型处理。
        3、更好的泛化：通过在多个文档块上进行训练，模型可以更好地学习和泛化到各种不同的文本样式和结构。
        4、数据增强：分割文档可以为训练数据提供更多的样本。如一个长文档可以被分割成多个部分，并分别作为单独的训练样本。

    应该尽量将文本分割为包含完整语义的段落或单元。
"""
# ------------------------ 文档分割方式 ----------------------
"""
    Langchain提供多种文档分割方式，区别在怎么确定块与块之间的边界、块由哪些字符/token组成、以及如何测量块大小，分割方式：
        1、基于字符分割：CharacterTextSplitter
        2、基于Markdown标题分割：MarkdownHeaderTextSplitter
        3、基于Token数量分割：TokenTextSplitter
        4、基于句子语义分割：SentenceTransformerdTokenTextSplitter
        5、字符递归分割（用于将长文本按语义完整性分割成指定大小的块）：RecursiveCharacterTextSplitter
        6、语言指令实现目标分割（用于C++、Python、Ruby、Markdown等）：Language
        7、自然语言文本分割：NLTKTextSplitter
    基于字符分割：Langchain 中文本分割器都根据 chunk_size (块大小)和 chunk_overlap (块与块之间的重叠大小)进行分割。
        chunk_size：指每个块包含的字符或 Token （如单词、句子等）的数量
        chunk_overlap：指两个块之间共享的字符数量，用于保持上下文的连贯性，避免分割丢失上下文信息
"""
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 块集合
chunks = []

txt_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=400,
    chunk_overlap=100
)

# 加载PDF文档
loader = PyPDFLoader("车路云一体化发展：架构、技术与产业全景.pdf")
# 调用 PyPDFLoader Class 的函数 load对pdf文件进行加载
pages = loader.load()

# 数据类型是langchain_core.documents.base.Document
document = pages[0]

# chunks = txt_splitter.split_text(document.page_content)

# ------------------------ 基于字符分割 ----------------------
"""
CharacterTextSplitter 是字符文本分割，分隔符的参数是单个的字符串
RecursiveCharacterTextSplitter 是递归字符文本分割，将按不同的字符递归地分割（按照这个优先级["\n\n", "\n", " ", ""]），这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter 比 CharacterTextSplitter 对文档切割得更加碎片化。RecursiveCharacterTextSplitter 需要关注的是如下4个参数：
    separators - 分隔符字符串数组
    chunk_size - 每个文档的字符数量限制
    chunk_overlap - 两份文档重叠区域的长度
    length_function - 长度计算函数

建议在通用文本中使用递归字符文本分割器
"""
# 导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 设置块大小
chunk_size = 100
# 设置块重叠大小
chunk_overlap = 50

# 初始化递归字符文本分割器
recurs_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    # 设置分割符集，如果需要按照句子进行分隔，则为：separators=["\n\n", "\n", "(?<=\。 )", " ", ""]
    separators=["\n\n", "\n", "、"]
)

# 初始化字符文本分割器，默认是以分行符号分割
txt_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    # 设置分割符
    separator="、"
)
# 基于递归分块
# chunks = recurs_splitter.split_text(document.page_content)

# 基于字符分块
# chunks = txt_splitter.split_text(document.page_content)

# for chunk in chunks:
#     print(chunk)

# ------------------------ 基于字符分割 ----------------------
"""
很多 LLM 的上下文窗口长度限制是按照 Token 来计数的。因此，以 LLM 的视角，按照 Token 对文本进行分隔，通常可以得到更好的结果。
"""
# 使用token分割器进行分割，将块大小设为1，块重叠大小设为0，相当于将任意字符串分割成了单个Token组成的列
from langchain.text_splitter import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size = 1,
    chunk_overlap = 0
)

# 注：目前 LangChain 基于 Token 的分割器还不支持中文
text = "Michael was Studying the AI"

# token长度和字符长度不一样，token通常为4个字符
# chunks = token_splitter.split_text(text)

# ------------------------ 分割Markdown文档 ----------------------
"""
    分块的目的是把具有上下文的文本放在一起，我们可以通过使用指定分隔符来进行分隔，但有些类型的文档（例如 Markdown ）本身就具有可用于分割的结构（如标题），
Markdown 标题文本分割器会根据标题或子标题来分割一个 Markdown 文档，并将标题作为元数据添加到每个块中
"""
# 分割一个自定义 Markdown 文档
# Markdown分割器
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = """# Title\n\n \
## 第一章\n\n \
李白乘舟将欲行\n\n 忽然岸上踏歌声\n\n \
### Section \n\n \
桃花潭水深千尺 \n\n
## 第二章\n\n \
不及汪伦送我情"""

# 定义想要分割的标题列表和名称
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

# 加载文档分割器
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# 文本分割：每个块都包含了页面内容和元数据，元数据中记录了该块所属的标题和子标题
chunks = markdown_splitter.split_text(markdown_document)
for chunk in chunks:
    print(chunk)
