"""选股节点

使用 ReAct 子图进行多维度选股，输出候选股票池。
包含程序化调仓触发评估，在有持仓时先判断是否需要调仓。
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.utils import NodeLogger
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    STOCK_SCREENING_SYSTEM,
    STOCK_SCREENING_COMPRESS,
    STOCK_SCREENING_SYNTHESIZE,
    get_style_guidance,
)
from src.state import PortfolioState
from src.tools.market_tools import MARKET_TOOLS
from src.tools.analysis_tools import ANALYSIS_TOOLS
from src.tools.search_tools import web_search
from src.tools.skill_tools import SKILL_TOOLS, build_available_skills_prompt
from src.portfolio_persistence import (
    format_positions_summary,
    load_target_portfolio,
    compute_position_deviations,
)

logger = logging.getLogger("portfolio")


def _build_screening_subgraph():
    """构建选股 ReAct 子图"""
    all_tools = SKILL_TOOLS + MARKET_TOOLS + ANALYSIS_TOOLS + [web_search]
    cfg = ReActSubgraphConfig(
        node_name="StockScreening",
        tools=all_tools,
        system_prompt_template=STOCK_SCREENING_SYSTEM,
        compress_prompt=STOCK_SCREENING_COMPRESS,
        synthesize_prompt=STOCK_SCREENING_SYNTHESIZE,
        compression_switch_attr="enable_analysis_compression",
        max_iterations_attr="screening_max_iterations",
        model_role="research",
    )
    return build_react_subgraph(cfg)


_screening_subgraph = None


def _get_screening_subgraph():
    global _screening_subgraph
    if _screening_subgraph is None:
        _screening_subgraph = _build_screening_subgraph()
    return _screening_subgraph


def _assess_rebalance_triggers(
    current_positions: dict,
    total_value: float,
    cash_balance: float,
    market_data: dict,
    app_config,
    nl: NodeLogger,
) -> str:
    """程序化评估调仓触发条件，返回触发原因摘要。

    未触发任何条件返回空字符串。
    """
    triggers = app_config.rebalance_triggers
    reasons = []

    # 1. 个股盈亏触发
    for code, pos in current_positions.items():
        pnl_pct = pos.get("unrealized_pnl_pct", 0)
        name = pos.get("stock_name", code)
        if abs(pnl_pct) >= triggers.pnl_threshold:
            tag = "浮盈" if pnl_pct > 0 else "浮亏"
            reasons.append(f"[盈亏触发] {name}({code}) {tag}{pnl_pct:+.1f}% 超过阈值±{triggers.pnl_threshold}%")

    # 2. 偏离度触发
    last_target = load_target_portfolio()
    if last_target and current_positions:
        deviations = compute_position_deviations(current_positions, last_target, total_value)
        for d in deviations:
            if abs(d["deviation"]) >= triggers.deviation_threshold:
                reasons.append(f"[偏离触发] {d['name']}({d['code']}) 偏离{d['deviation']:+.1f}% 超过阈值±{triggers.deviation_threshold}%")
            if "目标外" in d["action_hint"]:
                reasons.append(f"[偏离触发] {d['name']}({d['code']}) 不在目标组合中")

    # 3. 持仓天数触发
    today = datetime.now().strftime("%Y-%m-%d")
    for code, pos in current_positions.items():
        buy_date = pos.get("buy_date", "")
        if buy_date:
            try:
                days_held = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(buy_date, "%Y-%m-%d")).days
                if days_held >= triggers.max_hold_days:
                    reasons.append(f"[持仓时间触发] {pos.get('stock_name', code)}({code}) 已持有{days_held}天 ≥{triggers.max_hold_days}天")
            except ValueError:
                pass

    # 4. 大盘异动触发
    for key, val in market_data.items():
        if key.startswith("index_") and "000001" in key:
            chg = abs(val.get("change_pct", 0))
            if chg >= triggers.market_shock_pct:
                reasons.append(f"[大盘异动触发] {val.get('name', '上证指数')} 涨跌幅{val.get('change_pct', 0):+.2f}% 超过±{triggers.market_shock_pct}%")

    # 5. 现金比例过低
    if total_value > 0:
        cash_ratio = cash_balance / total_value
        if cash_ratio < triggers.min_cash_alert:
            reasons.append(f"[现金告警] 现金比例{cash_ratio*100:.1f}% 低于{triggers.min_cash_alert*100:.0f}%")

    if reasons:
        nl._log("info", f"调仓触发条件评估: 触发{len(reasons)}项")
        for r in reasons:
            nl._log("info", f"  {r}")
    else:
        nl._log("info", "调仓触发条件评估: 未触发任何条件，当前持仓无明显异常")

    return "\n".join(reasons) if reasons else ""


async def screen_stocks(state: PortfolioState, config: RunnableConfig) -> dict:
    """执行选股"""
    app_config = get_app_config(config)
    nl = NodeLogger("StockScreening", app_config.logging.detail_level)
    portfolio_cfg = app_config.portfolio
    risk_cfg = app_config.risk

    market_analysis = state.get("macro_analysis", "暂无市场分析")
    market_data = state.get("market_data", {})
    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    total_value = state.get("portfolio_value", 0) or portfolio_cfg.initial_capital

    positions_str = format_positions_summary({
        "current_positions": current_positions,
        "cash_balance": cash_balance,
        "portfolio_value": total_value,
    }) if current_positions else "无当前持仓（新建组合）"

    # 有持仓时进行程序化调仓触发评估
    trigger_context = ""
    if current_positions:
        trigger_reasons = _assess_rebalance_triggers(
            current_positions, total_value, cash_balance, market_data, app_config, nl,
        )
        if trigger_reasons:
            trigger_context = (
                f"\n\n## ⚠ 调仓触发条件评估（程序化检测）\n"
                f"以下条件已触发，选股时需重点关注：\n{trigger_reasons}\n"
                f"\n请优先评估上述触发项涉及的持仓，决定是保留还是替换。"
            )
        else:
            trigger_context = (
                "\n\n## 调仓触发条件评估\n"
                "程序化检测结果：当前持仓未触发任何调仓条件。\n"
                "除非市场分析发现重大变化，否则应以保留现有优质持仓为主，仅做微调。"
            )

    # 引入持仓审查结论
    position_review = state.get("position_review") or ""
    review_context = ""
    if position_review:
        review_context = (
            "\n\n## 持仓审查结论（来自 PositionReview 节点的深度诊断）\n"
            f"{position_review}\n\n"
            "⚠ 请严格遵循持仓审查的结论：\n"
            "- 诊断为「继续持有」的股票，除非你发现新的重大利空，否则不要建议替换\n"
            "- 诊断为「建议减持/清仓」的股票，需要找到替代标的\n"
            "- 诊断为「密切观察」的股票，保留持有但标注观察\n"
        )

    nl.input_summary(
        f"市场分析: {len(market_analysis)}字符, "
        f"持仓审查: {len(position_review)}字符, "
        f"当前持仓: {len(current_positions)}只, "
        f"触发评估: {'有触发' if trigger_context and '⚠' in trigger_context else '无触发'}, "
        f"风格: {portfolio_cfg.style}, 目标持仓: {portfolio_cfg.max_positions}, "
        f"最小成交额: {risk_cfg.min_daily_turnover}万元"
    )

    system_prompt = STOCK_SCREENING_SYSTEM.format(
        market_analysis=market_analysis,
        current_positions=positions_str,
        style=portfolio_cfg.style,
        style_guidance=get_style_guidance(portfolio_cfg.style),
        max_positions=portfolio_cfg.max_positions,
        max_single_weight=portfolio_cfg.max_single_weight * 100,
        max_sector_stocks=portfolio_cfg.max_sector_stocks,
        min_turnover=risk_cfg.min_daily_turnover,
        min_market_cap=risk_cfg.min_market_cap_billion,
        max_iterations=getattr(app_config.react, "screening_max_iterations", 12),
        max_positions_2x=portfolio_cfg.max_positions * 2,
        available_skills=build_available_skills_prompt(),
    )

    subgraph = _get_screening_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                f"请根据以下市场分析结论，进行A股选股。\n\n"
                f"## 市场分析\n{market_analysis}\n\n"
                f"## 当前持仓\n{positions_str}"
                f"{trigger_context}\n\n"
                f"{review_context}\n\n"
                f"## 选股要求\n"
                f"- 投资风格: {portfolio_cfg.style}\n"
                f"- 目标持仓数: {portfolio_cfg.max_positions}\n"
                f"- 最小日均成交额: {risk_cfg.min_daily_turnover}万元\n"
                f"- 当日买入的股票受T+1限制不可卖出，请优先保留优质持仓\n"
                f"- 持仓审查结论是选股的重要输入，请充分参考\n"
            ))
        ],
        "research_topic": "A股选股",
        "tool_call_iterations": 0,
        "task_id": "stock_screening",
        "system_prompt": system_prompt,
    }

    result = await subgraph.ainvoke(sub_input, config)
    screening = result.get("compressed_research", "")

    nl.output_summary(f"选股报告: {len(screening):,}字符")

    return {
        "screening_result": screening,
        "progress_log": [{"node": "stock_screening", "status": "completed"}],
    }
