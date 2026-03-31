"""市场分析节点

使用 ReAct 子图进行宏观市场分析和行业轮动研判。
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.utils import NodeLogger
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    MARKET_ANALYSIS_SYSTEM,
    MARKET_ANALYSIS_COMPRESS,
    MARKET_ANALYSIS_SYNTHESIZE,
)
from src.state import PortfolioState
from src.tools.market_tools import MARKET_TOOLS
from src.tools.analysis_tools import detect_market_regime
from src.tools.search_tools import web_search
from src.tools.skill_tools import SKILL_TOOLS, build_available_skills_prompt
from src.portfolio_persistence import format_positions_summary

logger = logging.getLogger("portfolio")


def _build_market_analysis_subgraph():
    """构建市场分析 ReAct 子图"""
    cfg = ReActSubgraphConfig(
        node_name="MarketAnalysis",
        tools=SKILL_TOOLS + MARKET_TOOLS + [detect_market_regime, web_search],
        system_prompt_template=MARKET_ANALYSIS_SYSTEM,
        compress_prompt=MARKET_ANALYSIS_COMPRESS,
        synthesize_prompt=MARKET_ANALYSIS_SYNTHESIZE,
        compression_switch_attr="enable_analysis_compression",
        max_iterations_attr="max_iterations",
        model_role="research",
    )
    return build_react_subgraph(cfg)


_market_subgraph = None


def _get_market_subgraph():
    global _market_subgraph
    if _market_subgraph is None:
        _market_subgraph = _build_market_analysis_subgraph()
    return _market_subgraph


async def analyze_market(state: PortfolioState, config: RunnableConfig) -> dict:
    """执行市场分析"""
    app_config = get_app_config(config)
    nl = NodeLogger("MarketAnalysis", app_config.logging.detail_level)

    market_data = state.get("market_data", {})
    market_summary = _format_market_data(market_data)

    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    total_value = state.get("portfolio_value", 0)
    positions_str = ""
    if current_positions:
        positions_str = format_positions_summary({
            "current_positions": current_positions,
            "cash_balance": cash_balance,
            "portfolio_value": total_value,
        })

    index_count = sum(1 for k in market_data if k.startswith("index_"))
    has_sentiment = "sentiment" in market_data
    has_sector = "sector_data" in market_data
    has_northbound = "northbound" in market_data
    sector_src = market_data.get("sector_source", "无")
    nl.input_summary(
        f"市场数据: {index_count}个指数, "
        f"情绪数据={'有' if has_sentiment else '无'}, "
        f"行业资金流向={'有' if has_sector else '无'}({sector_src}), "
        f"北向资金={'有' if has_northbound else '无'}, "
        f"当前持仓: {len(current_positions)}只, "
        f"输入文本{len(market_summary)}字符"
    )
    nl._log("info", f"传入LLM的市场数据全文:\n{market_summary}")

    portfolio_context = ""
    if positions_str:
        portfolio_context = f"\n\n当前投资组合:\n{positions_str}\n\n请在市场分析中结合当前持仓情况，对持仓涉及的行业给予额外关注。"

    position_review = state.get("position_review") or ""
    review_context = ""
    if position_review:
        review_context = (
            f"\n\n## 持仓审查结论（已完成的持仓诊断）\n"
            f"{position_review}\n\n"
            f"请在市场分析中重点关注持仓审查中提到的风险行业和建议替换的方向。"
        )

    max_iterations = getattr(app_config.react, "max_iterations", 8)
    system_prompt = MARKET_ANALYSIS_SYSTEM.format(
        max_iterations=max_iterations,
        available_skills=build_available_skills_prompt(),
    )

    subgraph = _get_market_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                f"请分析当前A股市场环境并给出投资建议。\n\n"
                f"当前市场数据:\n{market_summary}"
                f"{portfolio_context}"
                f"{review_context}"
            ))
        ],
        "research_topic": "A股市场分析",
        "tool_call_iterations": 0,
        "task_id": "market_analysis",
        "system_prompt": system_prompt,
    }

    result = await subgraph.ainvoke(sub_input, config)
    analysis = result.get("compressed_research", "")

    nl.output_summary(f"市场分析报告: {len(analysis):,}字符")

    return {
        "macro_analysis": analysis,
        "progress_log": [{"node": "market_analysis", "status": "completed"}],
    }


def _safe_float(val) -> float:
    """安全的浮点数转换，处理百分号等格式。"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace('%', '').replace(',', '').replace('+', ''))
    except (ValueError, TypeError):
        return 0.0


def _format_market_data(data: dict) -> str:
    """格式化市场数据为文本"""
    lines = []

    for key, val in data.items():
        if key.startswith("index_"):
            vol = val.get('volume', 0)
            vol_str = f", 成交额: {vol/1e8:.0f}亿" if vol else ""
            lines.append(f"{val['name']}: {val['price']}, 涨跌幅: {val['change_pct']}%{vol_str}")

    sentiment = data.get("sentiment", {})
    if sentiment:
        parts = [
            f"上涨{sentiment.get('up_count', 0)}家",
            f"下跌{sentiment.get('down_count', 0)}家",
            f"涨停{sentiment.get('limit_up', 0)}",
            f"跌停{sentiment.get('limit_down', 0)}",
        ]
        up5 = sentiment.get('up_gt5pct')
        dn5 = sentiment.get('down_gt5pct')
        if up5 is not None:
            parts.append(f"涨>5%: {up5}家")
        if dn5 is not None:
            parts.append(f"跌>5%: {dn5}家")

        vol_b = sentiment.get('total_volume_billion', 0)
        parts.append(f"两市成交{vol_b:.0f}亿")

        vol_vs = sentiment.get('volume_vs_yesterday')
        if vol_vs:
            parts.append(f"较昨日{vol_vs}")

        main_flow = sentiment.get('main_net_inflow')
        if main_flow is not None:
            parts.append(f"主力净流入{main_flow:+.1f}亿")

        lines.append(f"\n市场情绪: {', '.join(parts)}")

    # 行业板块资金流向
    sector_data = data.get("sector_data", [])
    sector_source = data.get("sector_source", "")

    if sector_data and sector_source == "eastmoney":
        lines.append("\n行业资金流向(净流入排序):")
        for s in sector_data[:10]:
            name = s.get("名称", "?")
            chg = s.get("今日涨跌幅", 0)
            inflow = s.get("主力净流入_亿", 0)
            lines.append(f"  {name}: 涨跌{chg:+.2f}%, 主力净流入{inflow:+.2f}亿")
    elif sector_data and sector_source == "chinalin":
        lines.append("\n行业板块排名(涨跌幅排序):")
        for s in sector_data[:15]:
            name = s.get("名称", s.get("secu_name", "?"))
            chg = _safe_float(s.get("今日涨跌幅", s.get("change_rate", 0)))
            lines.append(f"  {name}: {chg:+.2f}%")
    elif sector_data:
        lines.append("\n行业资金流向(按净流入估算排序):")
        for s in sector_data[:15]:
            lines.append(
                f"  {s['行业']}: 涨跌{s['平均涨跌幅']:+.2f}%, "
                f"净流入≈{s['净流入估算_亿']:+.1f}亿, "
                f"上涨{s['上涨比例%']:.0f}%, "
                f"领涨{s['领涨股']}({s['领涨涨幅']:+.2f}%), "
                f"成交{s['总成交额_亿']:.0f}亿"
            )

    nb = data.get("northbound", {})
    if nb:
        date_info = nb.get("date") or nb.get("time", "")
        lines.append(
            f"\n北向资金({date_info}): 合计{nb['total_net']:+.2f}亿 "
            f"(沪股通{nb['sh_net']:+.2f}亿, 深股通{nb['sz_net']:+.2f}亿)"
        )

    missing = []
    if not nb:
        missing.append("北向资金(非交易时段或数据为空)")
    if "avg_change" in sentiment and sentiment["avg_change"] == 0:
        missing.append("平均涨跌幅")
    if missing:
        lines.append(f"\n[未获取到: {', '.join(missing)}]")

    lines.append(
        "\n[注: 以上为 MarketData 节点已采集的基础数据。"
        "如需日K线趋势、MA均线、个股资金流向等深度数据，可通过 Skill 或专用工具补充获取。]"
    )

    return "\n".join(lines) if lines else "暂无市场数据"
