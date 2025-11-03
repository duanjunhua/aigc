"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-02-11 16:
@Version: v1.0
@Description: 加载.env文件
"""
import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']


# 将读取环境的代码封装为函数
def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']

