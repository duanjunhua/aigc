"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-31 17:28
@Version: v1.0
@Description: 搭建一个基于个人文档的聊天机器人

使用 LangChain 框架，访问私有数据并建立个性化的问答系统。
    1、使用 LangChain 的多种文档加载器，从不同源导入各类数据。
    2、将文档分割为语义完整的文本块
    3、创建 Embedding，并将它们放入向量存储器中，并轻松实现语义搜索
    4、与 LLMs 相结合，将检索结果与问题传递给 LLM ，生成对原始问题的答案
    5、
"""

from chatgpt.tool import get_completion

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

# 初始化llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
# 初始化模型
embedding = OpenAIEmbeddings(model="text-embedding-3-small")


def load_db(file, chain_type, k):
    """
    该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库，定义检索器，并创建聊天机器人实例。
    参数:
        file (str): 要加载的 PDF 文件路径。
        chain_type (str): 链类型，用于指定聊天机器人的类型。
        k (int): 在检索过程中，返回最相似的 k 个结果。
    返回:
        qa (ConversationalRetrievalChain): 创建的聊天机器人实例。
    """
    # 载入文档
    loader = PyPDFLoader(file)
    documents = loader.load()
    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # 定义 Embeddings
    embeddings = embedding
    # 根据数据创建向量数据库
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # 定义检索器
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # 创建 chatbot 链，Memory 由外部管理
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return conversation_chain


"""
panel 和 Param 这两个库提供了丰富的组件和小工具，可以用来扩展和增强图形用户界面.
    Panel：可以创建交互式的控制面板
    Param：可以声明输入参数并生成控件
"""
import panel as pn
import param

html_code = """
<style>
    .background-markdown {
        background-color: '#F6F6F6'
    }
</style>
"""


# 用于存储聊天记录、回答、数据库查询和回复
class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.loaded_file = "matplotlib/第一回：Matplotlib初相识.pdf"
        self.qa = load_db(self.loaded_file, "stuff", 4)

    # 将文档加载到聊天机器人中
    def call_load_db(self, count):
        """
        count: 数量
        """
        # 初始化或未指定文件
        if count == 0 or file_input.value is None:
            return pn.pane.Markdown(f"加载文件: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # 本地副本
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style = "solid"
            self.clr_history()
        return pn.pane.Markdown(f"加载文件: {self.loaded_file}")

    # 处理对话链
    def convchain(self, query):
        """
        query: 用户的查询
        """
        if not query:
            return pn.WidgetBox(pn.Row('用户:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        self.panels.extend([
            pn.Row('用户:', pn.pane.Markdown(query, width=600)),
            pn.Row('AI:', pn.pane.Markdown(self.answer, width=600, css_classes=['background-markdown']))
        ])

        pn.extension(raw_css=[html_code])
        # 清除时清除装载指示器
        inp.value = ''
        return pn.WidgetBox(*self.panels, scroll=True)

    # 获取最后发送到数据库的问题
    @param.depends('db_query ', )
    def get_lquest(self):
        pn.extension(raw_css=[html_code])
        if not self.db_query:

            return pn.Column(
                pn.Row(pn.pane.Markdown(f"数据库的最后一个问题:", css_classes=['background-markdown'])),
                pn.Row(pn.pane.Str("截止目前无数据库"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"数据库查询:", css_classes=['background-markdown'])),
            pn.pane.Str(self.db_query)
        )

    # 获取数据库返回的源文件
    @param.depends('db_response',)
    def get_sources(self):
        if not self.db_response:
            return
        result_list=[
            pn.Row(pn.pane.Markdown(f"数据库查询结果:", css_classes=['background-markdown']))
        ]
        pn.extension(raw_css=[html_code])
        for doc in self.db_response:
            result_list.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*result_list, width=600, scroll=True)

    # 获取当前聊天记录
    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("还没有历史")), width=600, scroll=True)
        result_list=[
            pn.Row(pn.pane.Markdown(f"当前聊天记录变量", css_classes=['background-markdown']))
        ]
        pn.extension(raw_css=[html_code])
        for exchange in self.chat_history:
            result_list.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*result_list, width=600, scroll=True)

    # 清除聊天记录
    def clr_history(self, count=0):
        self.chat_history = []
        return


# ------------------------ 初始化聊天机器人 ----------------------
cb = cbfs()

# ------------------------ 定义界面的小部件 ----------------------
# PDF 文件的文件输入小部件
file_input = pn.widgets.FileInput(accept='.pdf')
# 加载数据库的按钮
button_load = pn.widgets.Button(name="加载数据", button_type='primary')
# 清除聊天记录的按钮
button_clearhistory = pn.widgets.Button(name="清楚对话记录", button_type='warning')
# 将清除历史记录功能绑定到按钮上
button_clearhistory.on_click(cb.clr_history)
# 用于用户查询的文本输入小部件
inp = pn.widgets.TextInput(placeholder='请输入您要的提问…')

# 将加载数据库和对话的函数绑定到相应的部件上
bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp)
jpg_pane = pn.pane.Image('./img/convchain.jpg')

# 使用 Panel 定义界面布局
tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator = True, height = 300),
    pn.layout.Divider(),
)
tab2 = pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources),
)
tab3 = pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4 = pn.Column(
    pn.Row(file_input, button_load, bound_button_load),
    pn.Row(button_clearhistory, pn.pane.Markdown("清除对话记录，开始新的对话")),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)
# 将所有选项卡合并为一个仪表盘
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# 与你的数据机器人聊天')),
    pn.Tabs(('对话', tab1), ('数据库', tab2), ('聊天记录', tab3), ('配置', tab4))
)

dashboard.show()

