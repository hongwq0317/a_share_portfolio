"""交易执行节点

使用 ReAct 子图将策略转化为实际交易操作。
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.utils import NodeLogger
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    TRADE_EXECUTION_SYSTEM,
    TRADE_EXECUTION_COMPRESS,
    TRADE_EXECUTION_SYNTHESIZE,
)
from src.state import PortfolioState
from src.tools.trade_tools import TRADE_TOOLS
from src.tools.skill_tools import build_available_skills_prompt
from src.tools.risk_tools import detect_limit_updown
from src.tools.market_tools import get_stock_realtime_quote, think_tool
from src.portfolio_persistence import sync_after_trades, format_positions_summary

logger = logging.getLogger("portfolio")


def _estimate_turnover_cost(
    positions: dict, portfolio_value: float, trading_cfg
) -> str:
    """预估各档换仓率对应的交易成本，供 LLM 参考决策。"""
    if not positions or portfolio_value <= 0:
        return "无当前持仓，无换仓成本。"

    stock_value = sum(
        p.get("shares", 0) * p.get("current_price", p.get("avg_cost", 0))
        for p in positions.values()
    )
    comm = trading_cfg.commission_rate
    stamp = trading_cfg.stamp_duty_rate
    slip = getattr(trading_cfg, "slippage_pct", 0.1) / 100

    lines = [
        f"当前持仓市值: {stock_value:,.0f}元, 总资产: {portfolio_value:,.0f}元",
        f"费率: 佣金{comm*100:.2f}%(双边) + 印花税{stamp*100:.2f}%(卖出) + 滑点约{slip*100:.1f}%",
        "",
        "| 换仓率 | 交易金额 | 预估成本 | 占总资产 |",
        "|--------|----------|----------|----------|",
    ]
    for pct in [10, 20, 30, 50, 80]:
        trade_amount = stock_value * pct / 100
        cost = trade_amount * (2 * comm + stamp + 2 * slip)
        lines.append(
            f"| {pct}% | {trade_amount:,.0f}元 | {cost:,.0f}元 | {cost/portfolio_value*100:.2f}% |"
        )

    lines.append("")
    lines.append("⚠ 换仓成本是确定性损失，请确保调仓预期收益至少覆盖成本的3倍。")
    return "\n".join(lines)


def _trade_continuation_check(
    ai_text: str, iteration: int, max_iterations: int
) -> str | None:
    """检测交易执行是否未完成，返回提醒消息或 None。

    当 LLM 输出的报告中包含"未执行买入"等标记时，说明卖出完成
    但买入被跳过了，需要提醒 LLM 继续执行买入操作。
    """
    if iteration >= max_iterations - 2:
        return None

    incomplete_markers = [
        "未执行买入", "⚠️ 未执行", "未执行",
        "暂未买入", "未能买入", "暂缓买入",
        "待后续买入", "买入暂缓",
    ]
    has_incomplete = any(marker in ai_text for marker in incomplete_markers)
    if not has_incomplete:
        return None

    return (
        '你刚才输出了报告但有多笔买入操作标记为"未执行"。'
        "卖出已经完成并释放了资金，现在请立即执行剩余的买入操作：\n"
        "1. 调用 get_current_portfolio 确认当前现金余额\n"
        "2. 对策略建议买入的股票逐一调用 simulate_buy\n"
        "3. 如果某只股票确实因涨停、资金不足等原因无法买入，跳过它继续买入下一只\n"
        "4. 所有买入完成后，再调用 get_current_portfolio 进行自检\n\n"
        "⛔ 不要再次输出报告，直接调用工具执行买入。"
    )


def _build_trade_subgraph():
    """构建交易执行 ReAct 子图"""
    tools = TRADE_TOOLS + [get_stock_realtime_quote, detect_limit_updown, think_tool]
    cfg = ReActSubgraphConfig(
        node_name="TradeExecution",
        tools=tools,
        system_prompt_template=TRADE_EXECUTION_SYSTEM,
        compress_prompt=TRADE_EXECUTION_COMPRESS,
        synthesize_prompt=TRADE_EXECUTION_SYNTHESIZE,
        compression_switch_attr="enable_strategy_compression",
        max_iterations_attr="trade_max_iterations",
        model_role="decision",
        no_tool_continuation_check=_trade_continuation_check,
        max_no_tool_retries=2,
    )
    return build_react_subgraph(cfg)


_trade_subgraph = None


def _get_trade_subgraph():
    global _trade_subgraph
    if _trade_subgraph is None:
        _trade_subgraph = _build_trade_subgraph()
    return _trade_subgraph


async def execute_trades(state: PortfolioState, config: RunnableConfig) -> dict:
    """执行交易"""
    app_config = get_app_config(config)
    nl = NodeLogger("TradeExecution", app_config.logging.detail_level)
    trading_cfg = app_config.trading
    portfolio_cfg = app_config.portfolio

    strategy = state.get("strategy_reasoning", "暂无策略")
    risk_report = state.get("risk_report", "暂无风控报告")
    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    portfolio_value = state.get("portfolio_value", 0)
    positions_str = format_positions_summary({
        "current_positions": current_positions,
        "cash_balance": cash_balance,
        "portfolio_value": portfolio_value,
    }) if current_positions else "无当前持仓"

    nl.input_summary(
        f"组合策略: {len(strategy)}字符, 风控报告: {len(risk_report)}字符, "
        f"当前持仓: {len(current_positions)}只, 总资产: {portfolio_value:,.0f}元, "
        f"现金: {cash_balance:,.0f}元, "
        f"交易模式: {trading_cfg.mode}, 佣金率: {trading_cfg.commission_rate}"
    )

    from src.prompts import TRADE_EXECUTION_SYSTEM
    system_prompt = TRADE_EXECUTION_SYSTEM.format(
        strategy_reasoning=strategy,
        risk_report=risk_report,
        current_positions=positions_str,
        trade_mode=trading_cfg.mode,
        commission_rate=trading_cfg.commission_rate,
        stamp_duty_rate=trading_cfg.stamp_duty_rate,
        min_trade_amount=portfolio_cfg.min_trade_amount,
        max_single_weight=f"{portfolio_cfg.max_single_weight*100:.0f}",
        max_sector_weight=f"{portfolio_cfg.max_sector_weight*100:.0f}",
        min_cash_pct=f"{portfolio_cfg.min_cash_ratio*100:.0f}",
        min_rebalance_deviation=portfolio_cfg.min_rebalance_deviation,
        max_iterations=getattr(app_config.react, "trade_max_iterations", 20),
        available_skills=build_available_skills_prompt(),
    )

    cost_context = _estimate_turnover_cost(current_positions, portfolio_value, trading_cfg)

    subgraph = _get_trade_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                f"请根据组合策略和风控建议执行交易。\n\n"
                f"## 组合策略\n{strategy}\n\n"
                f"## 风控报告\n{risk_report}\n\n"
                f"## 当前持仓\n{positions_str}\n\n"
                f"## 交易参数\n"
                f"- 交易模式: {trading_cfg.mode}\n"
                f"- 手续费率: {trading_cfg.commission_rate}\n"
                f"- 印花税率(卖): {trading_cfg.stamp_duty_rate}\n"
                f"- 最小交易金额: {portfolio_cfg.min_trade_amount}元\n\n"
                f"## 换仓成本参考（程序预估）\n{cost_context}\n"
            ))
        ],
        "research_topic": "交易执行",
        "tool_call_iterations": 0,
        "task_id": "trade_execution",
        "system_prompt": system_prompt,
    }

    result = await subgraph.ainvoke(sub_input, config)
    execution_report = result.get("compressed_research", "")

    nl.output_summary(f"交易执行报告: {len(execution_report):,}字符")

    updated_fields = sync_after_trades()
    n = len(updated_fields.get("current_positions", {}))
    pv = updated_fields.get("portfolio_value", 0)
    cash = updated_fields.get("cash_balance", 0)
    nl._log("info", f"交易后持仓同步: {n}只, 总资产{pv:,.0f}元, 现金{cash:,.0f}元")
    nl._log("info", f"持仓明细:\n{format_positions_summary(updated_fields)}")

    return {
        "execution_report": execution_report,
        "current_positions": {"type": "override", "value": updated_fields["current_positions"]},
        "portfolio_value": updated_fields["portfolio_value"],
        "cash_balance": updated_fields["cash_balance"],
        "progress_log": [{"node": "trade_execution", "status": "completed"}],
    }
