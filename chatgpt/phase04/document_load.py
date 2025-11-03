"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-27 16:58
@Version: v1.0
@Description: 文档加载
"""
# ------------------------ PDF 文档 ----------------------
"""
需要安装第三方库 pypdf
"""
from langchain_community.document_loaders import PyPDFLoader

# 加载PDF文档
# 创建一个 PyPDFLoader Class 实例，输入为待加载的pdf文档路径
loader = PyPDFLoader("车路云一体化发展：架构、技术与产业全景.pdf")

# 调用 PyPDFLoader Class 的函数 load对pdf文件进行加载
pages = loader.load()

# 探索加载的数据
# 数据类型list
# print(type(pages))

# 数据类型是langchain_core.documents.base.Document
document = pages[0]
# print(type(document))

# page_content：包含该文档页面的内容
# print(document.page_content)

# metadata ：为文档页面相关的描述性数据
# print(document.metadata)

# ------------------------ 加载Youtube音频文档 ----------------------
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

video_url = "https://www.youtube.com/watch?v=_PHdzsQaDgw"
local_path="./"

# 创建一个 GenericLoader Class 实例
video_loader = GenericLoader(
    # 将链接url中的Youtube视频的音频下载下来,存在本地路径save_dir
    YoutubeAudioLoader([video_url], local_path),

    # 使用OpenAIWhisperPaser解析器将音频转化为文本
    OpenAIWhisperParser()
)
# 调用 GenericLoader Class 的函数 load对视频的音频文件进行加载
# video_pages = video_loader.load()

# ------------------------ 网页文档 ----------------------
"""
处理网页链接（URLs）
"""
from langchain_community.document_loaders import WebBaseLoader

# 创建一个 WebBaseLoader Class 实例
web_url = "https://github.com/duanjunhua/spring-boot/blob/master/README.md"
header = {
    'User-Agent': 'python-requests/2.27.1',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': '*/*',
    'Connection': 'keep-alive'
}

web_loader = WebBaseLoader(
    web_path=web_url,
    header_template=header,
    verify_ssl=False
)

# 调用 WebBaseLoader Class 的函数 load对文件进行加载
web_pages = web_loader.load()

# 探索加载的数据
for web_page in web_pages:
    print(web_page.metadata)
    print(web_page.page_content)
# 对数据进行进一步处理
import json

# convert_to_json = json.loads(web_page.page_content)




