"""再平衡监控节点

检查持仓偏离度，生成调仓建议。
"""

import json
import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.utils import NodeLogger
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    REBALANCE_MONITOR_SYSTEM,
    REBALANCE_MONITOR_COMPRESS,
    REBALANCE_MONITOR_SYNTHESIZE,
)
from src.state import PortfolioState
from src.tools.risk_tools import RISK_TOOLS
from src.tools.trade_tools import get_current_portfolio, update_portfolio_prices, TRADE_TOOLS
from src.tools.market_tools import get_stock_realtime_quote, think_tool
from src.tools.portfolio_tools import calculate_rebalance_trades
from src.portfolio_persistence import (
    sync_after_trades,
    format_positions_summary,
    load_target_portfolio,
    compute_position_deviations,
    format_deviation_summary,
)

logger = logging.getLogger("portfolio")


def _build_rebalance_subgraph():
    """构建再平衡监控 ReAct 子图"""
    tools = RISK_TOOLS + TRADE_TOOLS + [
        get_stock_realtime_quote, think_tool,
        calculate_rebalance_trades,
    ]
    cfg = ReActSubgraphConfig(
        node_name="RebalanceMonitor",
        tools=tools,
        system_prompt_template=REBALANCE_MONITOR_SYSTEM,
        compress_prompt=REBALANCE_MONITOR_COMPRESS,
        synthesize_prompt=REBALANCE_MONITOR_SYNTHESIZE,
        compression_switch_attr="enable_strategy_compression",
        max_iterations_attr="max_iterations",
        model_role="decision",
    )
    return build_react_subgraph(cfg)


_rebalance_subgraph = None


def _get_rebalance_subgraph():
    global _rebalance_subgraph
    if _rebalance_subgraph is None:
        _rebalance_subgraph = _build_rebalance_subgraph()
    return _rebalance_subgraph


async def monitor_and_rebalance(state: PortfolioState, config: RunnableConfig) -> dict:
    """检查并执行再平衡"""
    app_config = get_app_config(config)
    nl = NodeLogger("RebalanceMonitor", app_config.logging.detail_level)
    portfolio_cfg = app_config.portfolio

    # 优先从 state 获取 target_portfolio，否则从持久化文件加载
    target_portfolio = state.get("target_portfolio", {})
    if not target_portfolio:
        target_portfolio = load_target_portfolio()
        if target_portfolio:
            nl._log("info", f"从持久化文件加载目标组合: {len(target_portfolio)}只")

    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    portfolio_value = state.get("portfolio_value", 0)
    target_str = json.dumps(target_portfolio, ensure_ascii=False, indent=2) if target_portfolio else "无目标配置"
    positions_str = format_positions_summary({
        "current_positions": current_positions,
        "cash_balance": cash_balance,
        "portfolio_value": portfolio_value,
    }) if current_positions else "无当前持仓"

    # 计算偏离度
    deviation_context = ""
    if target_portfolio and current_positions:
        deviations = compute_position_deviations(current_positions, target_portfolio, portfolio_value)
        deviation_summary = format_deviation_summary(deviations, portfolio_cfg.rebalance_threshold)
        deviation_context = f"\n\n## 偏离度分析\n{deviation_summary}"
        nl._log("info", f"偏离度分析:\n{deviation_summary}")

    nl.input_summary(
        f"目标配置: {len(target_portfolio)}只, 当前持仓: {len(current_positions)}只, "
        f"总资产: {portfolio_value:,.0f}元, 偏离度阈值: {portfolio_cfg.rebalance_threshold}%"
    )

    from src.prompts import REBALANCE_MONITOR_SYSTEM
    system_prompt = REBALANCE_MONITOR_SYSTEM.format(
        target_portfolio=target_str,
        current_positions=positions_str,
        rebalance_threshold=portfolio_cfg.rebalance_threshold,
        rebalance_frequency=app_config.schedule.rebalance_frequency,
        max_iterations=getattr(app_config.react, "max_iterations", 8),
    )

    subgraph = _get_rebalance_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                f"请检查组合是否需要再平衡。\n\n"
                f"## 目标配置\n{target_str}\n\n"
                f"## 当前持仓\n{positions_str}"
                f"{deviation_context}\n\n"
                f"## 再平衡规则\n"
                f"- 偏离度阈值: {portfolio_cfg.rebalance_threshold}%\n"
            ))
        ],
        "research_topic": "再平衡检查",
        "tool_call_iterations": 0,
        "task_id": "rebalance_monitor",
        "system_prompt": system_prompt,
    }

    result = await subgraph.ainvoke(sub_input, config)
    rebalance_report = result.get("compressed_research", "")

    nl.output_summary(f"再平衡报告: {len(rebalance_report):,}字符")

    updated_fields = sync_after_trades()
    n = len(updated_fields.get("current_positions", {}))
    pv = updated_fields.get("portfolio_value", 0)
    nl._log("info", f"再平衡后持仓同步: {n}只, 总资产{pv:,.0f}元")

    return {
        "rebalance_report": rebalance_report,
        "current_positions": {"type": "override", "value": updated_fields["current_positions"]},
        "portfolio_value": updated_fields["portfolio_value"],
        "cash_balance": updated_fields["cash_balance"],
        "progress_log": [{"node": "rebalance_monitor", "status": "completed"}],
    }
