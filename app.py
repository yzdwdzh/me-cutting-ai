import streamlit as st
import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import openai  # 新增：用于调用 DeepSeek
import json  # 新增：用于解析 AI 的返回结果


# --- 新增：AI 解析函数 (DeepSeek 驱动) ---
def get_needs_from_ai(user_input, api_key):
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    prompt = f"""
    你是一个机电下料专家。请从描述中提取出【零件长度】和【数量】。
    用户描述："{user_input}"

    必须严格返回 JSON 格式（长度为 key，数量为 value），例如：
    {{
        "1.2": 15,
        "0.8": 10
    }}
    如果用户说“切3根2.5米”，JSON 就是 {{"2.5": 3}}。
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={'type': 'json_object'}
    )

    # 将 AI 的 JSON 文本转为 Python 字典，并确保长度是 float
    raw_dict = json.loads(response.choices[0].message.content)
    return {float(k): int(v) for k, v in raw_dict.items()}


# --- 核心算法：保持不变 ---
def solve_cutting_stock_with_splicing(parts_dict, stock_length, kerf=0.003):
    # ... (你原来的算法逻辑，这里省略，保持不变) ...
    processed_parts = []
    total_joints = 0
    for length, count in parts_dict.items():
        if length > stock_length:
            num_full_lengths = int(length // stock_length)
            remainder = length - (num_full_lengths * stock_length)
            total_joints += (num_full_lengths * count)
            for _ in range(count):
                for _ in range(num_full_lengths):
                    processed_parts.append(stock_length)
                if remainder > 0.001: processed_parts.append(remainder)
        else:
            if length > 0.001: processed_parts.extend([length] * count)

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver: return None
    solver.SetTimeLimit(120000)

    num_stock = len(processed_parts)
    x = {}
    for i in range(num_stock):
        for j in range(len(processed_parts)):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
    y = [solver.IntVar(0, 1, f'y_{i}') for i in range(num_stock)]

    for j in range(len(processed_parts)):
        solver.Add(sum(x[i, j] for i in range(num_stock)) == 1)
    for i in range(num_stock):
        solver.Add(sum(x[i, j] * (processed_parts[j] + kerf) for j in range(len(processed_parts))) <= y[i] * (
                stock_length + kerf))

    solver.Minimize(solver.Sum(y))
    if solver.Solve() in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        schemes = []
        for i in range(num_stock):
            if y[i].solution_value() > 0.5:
                scheme = sorted(
                    [processed_parts[j] for j in range(len(processed_parts)) if x[i, j].solution_value() > 0.5])
                schemes.append(tuple(scheme))
        return {'count': sum(y[i].solution_value() for i in range(num_stock)), 'summary': Counter(schemes),
                'joints': total_joints}
    return None


# --- 界面层 ---
st.set_page_config(page_title="AI 机电下料助手", layout="wide")
st.title("🏗️ AI 驱动：管材切割与拼接助手")

# 侧边栏：API 设置和参数
st.sidebar.header("配置")
api_key = st.sidebar.text_input("DeepSeek API Key", type="password")  # 建议必填
kerf = st.sidebar.number_input("锯缝宽度 (m)", value=0.003, min_value=0.000, step=0.001, format="%.3f")
stock_options = [float(s.strip()) for s in st.sidebar.text_input("可选原材(m)", value="6.0, 9.0, 12.0").split(",")]

# 初始化需求池
if 'needs' not in st.session_state:
    st.session_state.needs = {}

# --- 【联动重点 1】：AI 语音/文本识别入口 ---
st.subheader("💡 智能识别需求")
ai_input = st.text_input("在这里输入你的下料需求（大白话）：", placeholder="例如：切1.2米的15根，0.8米的10根...")

if st.button("🪄 AI 自动添加"):
    if not api_key:
        st.error("请在侧边栏填入 API Key 后使用")
    elif ai_input:
        with st.spinner("AI 正在解析数据..."):
            try:
                new_needs = get_needs_from_ai(ai_input, api_key)
                st.session_state.needs.update(new_needs)  # 将 AI 解析出的字典合并到现有需求
                st.success(f"AI 已成功识别并添加 {len(new_needs)} 项需求！")
            except Exception as e:
                st.error(f"AI 解析出错: {e}")

st.divider()

# --- 手动添加区域 (保留你原来的逻辑) ---
st.subheader("📝 需求清单管理")
c1, c2 = st.columns(2)
new_len = c1.number_input("零件长度(m)", min_value=0.001, step=0.001, format="%.3f")
new_cnt = c2.number_input("数量(个)", min_value=1, step=1)
if st.button("✅ 手动添加/更新"):
    st.session_state.needs[new_len] = int(new_cnt)

if st.button("🗑️ 清空所有"):
    st.session_state.needs = {}
    st.rerun()

# 实时显示清单
for length, count in list(st.session_state.needs.items()):
    col_text, col_del = st.columns([6, 1])
    col_text.write(f"📏 {length:.3f}m × {count}个")
    if col_del.button("删除", key=f"del_{length}"):
        del st.session_state.needs[length]
        st.rerun()

# --- 【联动重点 2】：运行优化 ---
if st.button("🚀 运行最优规划分析", type="primary"):
    if not st.session_state.needs:
        st.warning("需求清单为空，请先添加需求。")
    else:
        with st.spinner('正在计算最优方案...'):
            report = []
            total_parts_len = sum(k * v for k, v in st.session_state.needs.items())
            for s_len in stock_options:
                res = solve_cutting_stock_with_splicing(st.session_state.needs, s_len, kerf)
                if res:
                    waste_rate = ((res['count'] * s_len - total_parts_len) / (res['count'] * s_len)) * 100
                    report.append({"原材规格(m)": s_len, "采购根数": res['count'], "需接头数": res['joints'],
                                   "损耗率(%)": round(waste_rate, 2), "方案详情": res['summary']})

            if not report:
                st.error("无法找到可行方案。")
            else:
                df = pd.DataFrame(report).sort_values("损耗率(%)")
                st.table(df.drop(columns=["方案详情"]))

                # 图表显示逻辑 (保持你原有的 Plotly 代码)
                fig_waste = px.bar(df, x="原材规格(m)", y="损耗率(%)", color="损耗率(%)", text="损耗率(%)",
                                   title="不同规格损耗对比")
                fig_waste.update_traces(width=0.4)
                fig_waste.update_layout(xaxis=dict(tickmode='array', tickvals=df["原材规格(m)"]), bargap=0.5)
                st.plotly_chart(fig_waste)

                best = df.iloc[0]
                st.success(f"🏆 推荐规格: {best['原材规格(m)']}m (需接头: {best['需接头数']})")

                # 可视化分布图 (保持你原有的逻辑)
                fig_struct = go.Figure()
                for scheme_idx, (scheme, count) in enumerate(best['方案详情'].items()):
                    for part_len in scheme:
                        fig_struct.add_trace(go.Bar(
                            x=[part_len], y=[f"模式 {scheme_idx + 1} (执行{count}次)"],
                            orientation='h', text=f"{part_len:.3f}m", textposition='inside'
                        ))
                fig_struct.update_layout(barmode='stack', title="推荐方案切割示意图")
                st.plotly_chart(fig_struct)
                # --- 补回消失的文字清单部分 ---
                st.divider()
                st.subheader("📋 详细切割清单")

                # 遍历推荐规格（Best）中的方案详情
                for scheme, count in best['方案详情'].items():
                    # 这里的 scheme 是一个元组，比如 (1.2, 1.2, 0.8)
                    # 我们把它转成文字：1.200m + 1.200m + 0.800m
                    scheme_text = " + ".join([f"{i:.3f}m" for i in scheme])

                    # 用 Streamlit 的卡片组件展示，更美观
                    st.write(f"**模式**: [{scheme_text}] ———— 执行 **{count}** 次")

                # 可选：加一个导出按钮，方便你把结果存下来
                combined_text = "\n".join(
                    [f"模式 [{' + '.join([f'{i:.3f}m' for i in s])}]: {c}次" for s, c in best['方案详情'].items()])
                st.download_button("📂 导出切割清单为文本", combined_text, file_name="cutting_plan.txt")

