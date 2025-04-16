import json
import os


def load_terms_from_db(file_path="data.json"):
    """
    从指定的 JSON 文件加载专有名词数据
    :param file_path: JSON 文件路径，默认为当前目录下的 data.json
    :return: 包含术语数据的列表，格式为 [{"term": "...", "aliases": [...], "description": "..."}, ...]
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        # 读取并解析 JSON 数据
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 验证数据格式
        required_keys = ["term", "aliases", "description"]
        for item in data:
            if not all(key in item for key in required_keys):
                raise ValueError(f"数据项 {item} 缺少必要字段，需包含 {required_keys}")

        return data

    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}（请检查文件内容是否符合 JSON 格式）")
        return []
    except Exception as e:
        print(f"错误: {e}")
        return []


# 使用示例
if __name__ == "__main__":
    terms = load_terms_from_db()
    if terms:
        print("成功加载以下术语：")
        for term in terms:
            print(f"- {term['term']}(别名：{', '.join(term['aliases'])})")
    else:
        print("未加载到有效数据，请检查文件路径或内容格式。")
