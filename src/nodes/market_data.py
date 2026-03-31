"""市场数据获取节点（Skill 驱动）

使用 ReAct 子图 + ChinaLin Skill 获取大盘行情、行业数据、市场情绪等基础数据，
存入 state.market_data。LLM 读取 Skill 文档后动态调用 ChinaLin API。
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    MARKET_DATA_SYSTEM,
    MARKET_DATA_COMPRESS,
    MARKET_DATA_SYNTHESIZE,
)
from src.state import PortfolioState
from src.tools.skill_tools import SKILL_TOOLS, build_available_skills_prompt
from src.utils import NodeLogger

logger = logging.getLogger("portfolio")


def _build_market_data_subgraph():
    """构建市场数据获取 ReAct 子图"""
    cfg = ReActSubgraphConfig(
        node_name="MarketData",
        tools=SKILL_TOOLS,
        system_prompt_template=MARKET_DATA_SYSTEM,
        compress_prompt=MARKET_DATA_COMPRESS,
        synthesize_prompt=MARKET_DATA_SYNTHESIZE,
        compression_switch_attr="enable_market_data",
        max_iterations_attr="max_iterations",
        model_role="research",
    )
    return build_react_subgraph(cfg)


_market_data_subgraph = None


def _get_market_data_subgraph():
    global _market_data_subgraph
    if _market_data_subgraph is None:
        _market_data_subgraph = _build_market_data_subgraph()
    return _market_data_subgraph


def _parse_market_data(compressed: str) -> dict:
    """从 ReAct 子图的压缩输出中解析结构化市场数据。

    尝试从文本中提取 JSON 块，然后转换为与下游节点兼容的 market_data dict。
    """
    market_data: dict = {}

    json_str = _extract_json(compressed)
    if not json_str:
        logger.warning("[MarketData] 无法从压缩输出中提取 JSON，使用原始文本")
        market_data["raw_text"] = compressed
        market_data["fetch_time"] = datetime.now().isoformat()
        return market_data

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[MarketData] JSON 解析失败: {e}")
        market_data["raw_text"] = compressed
        market_data["fetch_time"] = datetime.now().isoformat()
        return market_data

    indices = parsed.get("indices", {})
    for code, info in indices.items():
        market_data[f"index_{code}"] = {
            "name": info.get("name", code),
            "price": _to_float(info.get("price", 0)),
            "change_pct": _to_float(info.get("change_pct", 0)),
            "volume": _to_float(info.get("amount", 0)),
        }

    sentiment = parsed.get("sentiment", {})
    if sentiment:
        s: dict[str, object] = {
            "total_stocks": int(sentiment.get("total_stocks", 0)),
            "up_count": int(sentiment.get("up_count", 0)),
            "down_count": int(sentiment.get("down_count", 0)),
            "limit_up": int(sentiment.get("limit_up", 0)),
            "limit_down": int(sentiment.get("limit_down", 0)),
            "avg_change": _to_float(sentiment.get("avg_change", 0)),
            "total_volume_billion": _to_float(sentiment.get("total_volume_billion", 0)),
        }
        if "main_net_inflow" in sentiment:
            s["main_net_inflow"] = _to_float(sentiment["main_net_inflow"])
        if "up_gt5pct" in sentiment:
            s["up_gt5pct"] = int(sentiment["up_gt5pct"])
        if "down_gt5pct" in sentiment:
            s["down_gt5pct"] = int(sentiment["down_gt5pct"])
        if "volume_vs_yesterday" in sentiment:
            s["volume_vs_yesterday"] = str(sentiment["volume_vs_yesterday"])
        market_data["sentiment"] = s

    sector_data = parsed.get("sector_data", [])
    if sector_data:
        market_data["sector_data"] = sector_data
        market_data["sector_source"] = "chinalin"

    northbound = parsed.get("northbound", {})
    if northbound:
        total = _to_float(northbound.get("total_net", 0))
        sh = _to_float(northbound.get("sh_net", 0))
        sz = _to_float(northbound.get("sz_net", 0))
        if total != 0 or sh != 0 or sz != 0:
            market_data["northbound"] = {
                "total_net": total,
                "sh_net": sh,
                "sz_net": sz,
                "date": northbound.get("date", ""),
                "source": northbound.get("source", "chinalin"),
            }

    market_data["fetch_time"] = datetime.now().isoformat()
    return market_data


def _extract_json(text: str) -> str | None:
    """从文本中提取第一个 JSON 块（支持 ```json 代码块和裸 JSON）。"""
    import re

    code_block = re.search(r'```(?:json)?\s*\n({[\s\S]*?})\s*\n```', text)
    if code_block:
        return code_block.group(1)

    brace_start = text.find('{')
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]
    return None


def _to_float(val) -> float:
    """安全的浮点数转换。"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).replace('%', '').replace(',', '').replace('+', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


async def fetch_market_data(state: PortfolioState, config: RunnableConfig) -> dict:
    """获取市场基础数据（通过 Skill 驱动的 ReAct 子图）"""
    app_config = get_app_config(config)
    nl = NodeLogger("MarketData", app_config.logging.detail_level)
    nl.start("获取市场数据 (Skill 驱动)")

    max_iterations = getattr(app_config.react, "max_iterations", 8)
    system_prompt = MARKET_DATA_SYSTEM.format(
        max_iterations=max_iterations,
        available_skills=build_available_skills_prompt(),
    )

    subgraph = _get_market_data_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                "请获取当前 A 股市场基础数据。\n\n"
                "需要获取：\n"
                "1. 主要指数行情（上证指数、深证成指、沪深300、创业板指）\n"
                "2. 市场总览（涨跌分布、成交额、北向资金）\n"
                "3. 行业板块排名（前15名）\n\n"
                "请先调用 read_skill 了解 API 接口，再逐步获取数据。"
            ))
        ],
        "research_topic": "A股市场数据获取",
        "tool_call_iterations": 0,
        "task_id": "market_data",
        "system_prompt": system_prompt,
    }

    try:
        result = await subgraph.ainvoke(sub_input, config)
        compressed = result.get("compressed_research", "")
        nl._log("info", f"ReAct 子图输出: {len(compressed)} chars")

        market_data = _parse_market_data(compressed)

        has_items: list[str] = []
        missing_items: list[str] = []
        for k, label in [("sentiment", "市场情绪"), ("sector_data", "行业资金流向"),
                          ("northbound", "北向资金")]:
            (has_items if k in market_data else missing_items).append(label)
        index_count = sum(1 for k in market_data if k.startswith("index_"))
        nl._log("info",
            f"数据获取完成: {index_count}个指数, "
            f"已获取=[{', '.join(has_items)}], "
            f"缺失=[{', '.join(missing_items) or '无'}]")

    except Exception as e:
        nl.error(f"Skill 驱动数据获取异常: {e}")
        market_data = {"fetch_time": datetime.now().isoformat(), "error": str(e)}

    nl.complete()
    return {"market_data": market_data}
