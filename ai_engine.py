import openai
import json


def translate_requirement_to_json(user_input, api_key):
    # 1. 初始化 DeepSeek 客户端
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"  # DeepSeek 的官方接口地址
    )

    # 2. 告诉 AI 它要做什么（Prompt）
    prompt = f"""
    你是一个机电工程下料专家。请从用户的描述中提取下料参数。
    用户描述："{user_input}"

    请严格返回 JSON 格式，不要包含任何解释文字，格式如下：
    {{
        "raw_length": 6.0,
        "raw_count": 5,
        "demand_list": [1.2, 1.2, 0.8]
    }}
    注意：如果用户说“切10根1.2米”，你的 demand_list 里面就要写 10 个 1.2。
    """

    # 3. 发送请求
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={'type': 'json_object'}  # 强制 AI 只吐出 JSON
    )

    # 4. 解析结果
    content = response.choices[0].message.content
    return json.loads(content)
