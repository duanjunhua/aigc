"""
@Author： Michael J H Duan[JunHua]
@Date: 2025-10-17 16:04
@Version: v1.0
@Description: 
"""
from chatgpt.tool import get_completion_from_messages
import json

with open("products.json", "r") as file:
    products = json.load(file)


def get_product_by_name(name):
    return products.get(name, None)


def get_product_by_name(name):
    return products.get(name, None)


def get_products_by_category(category):
    return [product for product in products.values() if product["category"] == category]


def read_string_to_list(input_string):
    """
    将输入的字符串转换为 Python 列表。
    参数:
    input_string: 输入的字符串，应为有效的 JSON 格式。
    返回:
    list 或 None: 如果输入字符串有效，则返回对应的 Python 列表，否则返回 None。
    """
    if input_string is None:
        return None
    try:
        # 将输入字符串中的单引号替换为双引号，以满足 JSON 格式的要求
        input_string = input_string.replace("'", "\"")
        data = json.loads(input_string)
        return data
    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
    return None


def generate_output_string(data_list):
    """
    根据输入的数据列表生成包含产品或类别信息的字符串。

    参数:
    data_list: 包含字典的列表，每个字典都应包含 "products" 或 "category" 的键。

    返回:
    output_string: 包含产品或类别信息的字符串。
    """
    output_string = ""
    if data_list is None:
        return output_string

    for data in data_list:
        try:
            if "products" in data and data["products"]:
                products_list = data["products"]
                for product_name in products_list:
                    product = get_product_by_name(product_name)
                    if product:
                        output_string += json.dumps(product, indent=4,ensure_ascii=False) + "\n"
                    else:
                        print(f"Error: Product '{product_name}' not found")
            elif "category" in data:
                category_name = data["category"]
                category_products = get_products_by_category(category_name)
                for product in category_products:
                    output_string += json.dumps(product, indent=4, ensure_ascii=False) + "\n"
                else:
                    print("Error: Invalid object format")
        except Exception as e:
            print(f"Error: {e}")
    return output_string


def find_category_and_product_only(user_input,products_and_category):
    """
    从用户输入中获取到产品和类别
    参数：
        @user_input：用户的查询
        @products_and_category：产品类型和对应产品的字典
    """
    delimiter = "####"
    system_message = f"""
        您将提供客户服务查询。客户服务查询将用{delimiter}字符分隔。
        输出一个 Python 列表，列表中的每个对象都是 Json 对象，每个对象的格式如下：
            '类别': <电脑和笔记本, 智能手机和配件, 电视和家庭影院系统,游戏机和配件, 音频设备, 相机和摄像机中的一个>,
        以及
            '名称': <必须在下面允许的产品中找到的产品列表>
        其中类别和产品必须在客户服务查询中找到。
        如果提到了一个产品，它必须与下面允许的产品列表中的正确类别关联。
        如果没有找到产品或类别，输出一个空列表。
        
        根据产品名称和产品类别与客户服务查询的相关性，列出所有相关的产品。
        不要从产品的名称中假设任何特性或属性，如相对质量或价格。
        
        允许的产品以 JSON 格式提供。
        每个项目的键代表类别。
        每个项目的值是该类别中的产品列表。
        允许的产品：{products_and_category}
    """
    few_shot_user_1 = """我想要最贵的电脑。"""
    few_shot_assistant_1 = """
        [{'category': '电脑和笔记本', 'products': ['TechPro 超极本', 'BlueWave 游戏本', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},
        {'role': 'assistant', 'content': few_shot_assistant_1},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
    ]
    return get_completion_from_messages(messages)

