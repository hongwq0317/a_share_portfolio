"""风险评估节点

使用 ReAct 子图进行全面风险评估。
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.utils import NodeLogger
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    RISK_ASSESSMENT_SYSTEM,
    RISK_ASSESSMENT_COMPRESS,
    RISK_ASSESSMENT_SYNTHESIZE,
)
from src.state import PortfolioState
from src.tools.risk_tools import RISK_TOOLS
from src.tools.skill_tools import build_available_skills_prompt
from src.tools.attribution_tools import ATTRIBUTION_TOOLS
from src.tools.market_tools import think_tool, get_stock_history
from src.tools.search_tools import web_search
from src.portfolio_persistence import format_positions_summary

logger = logging.getLogger("portfolio")


def _build_risk_subgraph():
    """构建风险评估 ReAct 子图"""
    tools = RISK_TOOLS + ATTRIBUTION_TOOLS + [think_tool, get_stock_history, web_search]
    cfg = ReActSubgraphConfig(
        node_name="RiskAssessment",
        tools=tools,
        system_prompt_template=RISK_ASSESSMENT_SYSTEM,
        compress_prompt=RISK_ASSESSMENT_COMPRESS,
        synthesize_prompt=RISK_ASSESSMENT_SYNTHESIZE,
        compression_switch_attr="enable_strategy_compression",
        max_iterations_attr="max_iterations",
        model_role="decision",
    )
    return build_react_subgraph(cfg)


_risk_subgraph = None


def _get_risk_subgraph():
    global _risk_subgraph
    if _risk_subgraph is None:
        _risk_subgraph = _build_risk_subgraph()
    return _risk_subgraph


async def assess_risk(state: PortfolioState, config: RunnableConfig) -> dict:
    """执行风险评估"""
    app_config = get_app_config(config)
    nl = NodeLogger("RiskAssessment", app_config.logging.detail_level)
    risk_cfg = app_config.risk

    strategy = state.get("strategy_reasoning") or "暂无组合策略"
    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    portfolio_value = state.get("portfolio_value", 0)
    positions_str = format_positions_summary({
        "current_positions": current_positions,
        "cash_balance": cash_balance,
        "portfolio_value": portfolio_value,
    }) if current_positions else "无当前持仓"

    # 引入持仓审查结论
    position_review = state.get("position_review") or ""

    nl.input_summary(
        f"组合策略: {len(strategy)}字符, "
        f"持仓审查: {len(position_review)}字符, "
        f"当前持仓: {len(current_positions)}只, "
        f"总资产: {portfolio_value:,.0f}元, 现金: {cash_balance:,.0f}元, "
        f"风控参数: 止损{risk_cfg.stop_loss_pct}%/止盈{risk_cfg.take_profit_pct}%/"
        f"最大回撤{risk_cfg.max_drawdown_pct}%/VaR{risk_cfg.var_limit_pct}%"
    )

    portfolio_cfg = app_config.portfolio
    from src.prompts import RISK_ASSESSMENT_SYSTEM
    system_prompt = RISK_ASSESSMENT_SYSTEM.format(
        target_portfolio=strategy,
        current_positions=positions_str,
        stop_loss_pct=risk_cfg.stop_loss_pct,
        take_profit_pct=risk_cfg.take_profit_pct,
        max_drawdown_pct=risk_cfg.max_drawdown_pct,
        var_limit_pct=risk_cfg.var_limit_pct,
        volatility_target=risk_cfg.volatility_target,
        max_single_weight=f"{portfolio_cfg.max_single_weight*100:.0f}",
        max_sector_weight=f"{portfolio_cfg.max_sector_weight*100:.0f}",
        min_cash_pct=f"{portfolio_cfg.min_cash_ratio*100:.0f}",
        max_iterations=getattr(app_config.react, "max_iterations", 8),
        available_skills=build_available_skills_prompt(),
    )

    review_context = ""
    if position_review:
        review_context = (
            f"\n\n## 持仓审查结论（来自 PositionReview 节点）\n"
            f"{position_review}\n\n"
            f"⚠ **重要约束**：持仓审查已对每只股票做了5维度深度诊断。\n"
            f"- 评分 ≥ 3.0 且诊断为「继续持有」的股票，风控不应发出减持/清仓指令，"
            f"除非该股浮亏已实际触及止损线({risk_cfg.stop_loss_pct}%)或出现重大新增风险事件\n"
            f"- 只有评分 < 3.0 或诊断为「建议减持/清仓」的股票才应纳入风控强制操作\n"
        )

    subgraph = _get_risk_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                f"请对以下投资组合进行全面风险评估。\n\n"
                f"## 组合策略\n{strategy}\n\n"
                f"## 当前持仓\n{positions_str}"
                f"{review_context}\n\n"
                f"## 风控参数\n"
                f"- 止损线: {risk_cfg.stop_loss_pct}%\n"
                f"- 止盈线: {risk_cfg.take_profit_pct}%\n"
                f"- 最大回撤限额: {risk_cfg.max_drawdown_pct}%\n"
                f"- 日VaR限额(95%): {risk_cfg.var_limit_pct}%\n"
                f"- 波动率目标: {risk_cfg.volatility_target}%\n"
            ))
        ],
        "research_topic": "风险评估",
        "tool_call_iterations": 0,
        "task_id": "risk_assessment",
        "system_prompt": system_prompt,
    }

    result = await subgraph.ainvoke(sub_input, config)
    risk_report = result.get("compressed_research", "")

    nl.output_summary(f"风控报告: {len(risk_report):,}字符")

    return {
        "risk_report": risk_report,
        "progress_log": [{"node": "risk_assessment", "status": "completed"}],
    }
