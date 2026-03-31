"""组合策略节点

使用 ReAct 子图进行组合构建和优化，输出目标配置方案。
"""

import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.utils import NodeLogger
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    PORTFOLIO_STRATEGY_SYSTEM,
    PORTFOLIO_STRATEGY_COMPRESS,
    PORTFOLIO_STRATEGY_SYNTHESIZE,
    get_style_guidance,
)
from src.state import PortfolioState
from src.tools.portfolio_tools import PORTFOLIO_TOOLS
from src.tools.skill_tools import build_available_skills_prompt
from src.tools.trade_tools import estimate_market_impact
from src.tools.market_tools import think_tool
from src.tools.search_tools import web_search
from src.portfolio_persistence import (
    format_positions_summary,
    save_target_portfolio,
    load_target_portfolio,
    compute_position_deviations,
    format_deviation_summary,
)

logger = logging.getLogger("portfolio")


def _build_strategy_subgraph():
    """构建组合策略 ReAct 子图"""
    tools = PORTFOLIO_TOOLS + [estimate_market_impact, think_tool, web_search]
    cfg = ReActSubgraphConfig(
        node_name="PortfolioStrategy",
        tools=tools,
        system_prompt_template=PORTFOLIO_STRATEGY_SYSTEM,
        compress_prompt=PORTFOLIO_STRATEGY_COMPRESS,
        synthesize_prompt=PORTFOLIO_STRATEGY_SYNTHESIZE,
        compression_switch_attr="enable_strategy_compression",
        max_iterations_attr="strategy_max_iterations",
        model_role="decision",
    )
    return build_react_subgraph(cfg)


_strategy_subgraph = None


def _get_strategy_subgraph():
    global _strategy_subgraph
    if _strategy_subgraph is None:
        _strategy_subgraph = _build_strategy_subgraph()
    return _strategy_subgraph


async def build_portfolio_strategy(state: PortfolioState, config: RunnableConfig) -> dict:
    """构建组合策略"""
    app_config = get_app_config(config)
    nl = NodeLogger("PortfolioStrategy", app_config.logging.detail_level)
    portfolio_cfg = app_config.portfolio

    market_analysis = state.get("macro_analysis", "暂无")
    screening_result = state.get("screening_result", "暂无")
    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    total_value = state.get("portfolio_value", 0) or portfolio_cfg.initial_capital

    positions_str = format_positions_summary({
        "current_positions": current_positions,
        "cash_balance": cash_balance,
        "portfolio_value": total_value,
    }) if current_positions else "无当前持仓（新建组合）"

    # 加载上次目标组合，计算偏离度
    last_target = load_target_portfolio()
    deviation_context = ""
    if last_target and current_positions:
        deviations = compute_position_deviations(current_positions, last_target, total_value)
        deviation_summary = format_deviation_summary(deviations, portfolio_cfg.rebalance_threshold)
        deviation_context = f"\n\n## 持仓偏离度（vs 上次目标组合）\n{deviation_summary}"
        nl._log("info", f"偏离度分析:\n{deviation_summary}")

    # 引入持仓审查结论
    position_review = state.get("position_review") or ""
    review_context = ""
    if position_review:
        review_context = (
            f"\n\n## 持仓审查结论\n{position_review}\n\n"
            "请在组合构建中参考持仓诊断结果，对诊断为「继续持有」的股票优先保留。"
        )

    nl.input_summary(
        f"市场分析: {len(market_analysis)}字符, "
        f"选股结果: {len(screening_result)}字符, "
        f"持仓审查: {len(position_review)}字符, "
        f"当前持仓: {len(current_positions)}只, "
        f"上次目标: {len(last_target)}只, "
        f"总资金: {total_value:,.0f}元, 现金: {cash_balance:,.0f}元, "
        f"风格: {portfolio_cfg.style}"
    )

    system_prompt = PORTFOLIO_STRATEGY_SYSTEM.format(
        market_analysis=market_analysis,
        screening_result=screening_result,
        total_value=f"{total_value:,.0f}",
        max_positions=portfolio_cfg.max_positions,
        max_single_weight=f"{portfolio_cfg.max_single_weight*100:.0f}",
        max_sector_weight=f"{portfolio_cfg.max_sector_weight*100:.0f}",
        max_sector_stocks=portfolio_cfg.max_sector_stocks,
        min_cash_pct=f"{portfolio_cfg.min_cash_ratio*100:.0f}",
        max_correlation=portfolio_cfg.max_correlation,
        max_turnover_rate=f"{portfolio_cfg.max_turnover_rate*100:.0f}",
        style=portfolio_cfg.style,
        style_guidance=get_style_guidance(portfolio_cfg.style),
        current_positions=positions_str,
        max_iterations=getattr(app_config.react, "strategy_max_iterations", 10),
        available_skills=build_available_skills_prompt(),
    )

    subgraph = _get_strategy_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                f"请基于以下信息构建投资组合。\n\n"
                f"## 市场分析\n{market_analysis}\n\n"
                f"## 候选股票\n{screening_result}\n\n"
                f"## 当前持仓\n{positions_str}"
                f"{deviation_context}"
                f"{review_context}\n\n"
                f"## 约束条件\n"
                f"- 总资金: {total_value:,.0f}元\n"
                f"- 最大持仓数: {portfolio_cfg.max_positions}\n"
                f"- 单股最大仓位: {portfolio_cfg.max_single_weight*100:.0f}%\n"
                f"- 单行业最大仓位: {portfolio_cfg.max_sector_weight*100:.0f}%\n"
                f"- 最低现金比例: {portfolio_cfg.min_cash_ratio*100:.0f}%\n"
                f"- 投资风格: {portfolio_cfg.style}\n"
            ))
        ],
        "research_topic": "组合构建",
        "tool_call_iterations": 0,
        "task_id": "portfolio_strategy",
        "system_prompt": system_prompt,
    }

    result = await subgraph.ainvoke(sub_input, config)
    strategy = result.get("compressed_research", "")

    # 从策略报告中解析目标组合并持久化
    parsed_target = _parse_target_from_strategy(strategy, nl)
    if parsed_target:
        save_target_portfolio(parsed_target)

    nl.output_summary(f"组合策略报告: {len(strategy):,}字符, 解析目标: {len(parsed_target)}只")

    return_dict = {
        "strategy_reasoning": strategy,
        "progress_log": [{"node": "portfolio_strategy", "status": "completed"}],
    }
    if parsed_target:
        return_dict["target_portfolio"] = {"type": "override", "value": parsed_target}
    return return_dict


def _parse_target_from_strategy(strategy_text: str, nl: NodeLogger) -> dict:
    """从策略报告中解析目标组合配置。

    尝试从 Markdown 表格中提取股票代码和目标权重。
    表格格式: | 股票 | 代码 | 行业 | 权重 | 金额 | ...
    """
    target = {}
    if not strategy_text:
        return target

    code_pattern = re.compile(r'(\d{6})')
    weight_pattern = re.compile(r'(\d+\.?\d*)\s*%')

    for line in strategy_text.split('\n'):
        if '|' not in line or '---' in line:
            continue

        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]

        if len(cells) < 3:
            continue

        code_match = None
        weight_val = None
        name = ""
        sector = ""

        for cell in cells:
            cm = code_pattern.search(cell)
            if cm and cm.group(1)[0] in ('0', '3', '6', '4', '8'):
                code_match = cm.group(1)

            wm = weight_pattern.search(cell)
            if wm:
                try:
                    w = float(wm.group(1))
                    if 0.5 <= w <= 30:
                        weight_val = w
                except ValueError:
                    pass

        if code_match and weight_val:
            if not name:
                for cell in cells:
                    if not code_pattern.search(cell) and not weight_pattern.search(cell) and len(cell) >= 2:
                        if any('\u4e00' <= ch <= '\u9fff' for ch in cell):
                            if not name:
                                name = cell
                            elif not sector:
                                sector = cell

            target[code_match] = {
                "name": name,
                "sector": sector,
                "target_weight": weight_val,
            }

    if target:
        nl._log("info", f"从策略报告解析出目标组合: {len(target)}只")
        for code, info in target.items():
            nl._log("info", f"  {info.get('name', code)}({code}): 目标权重{info['target_weight']:.1f}%", skip_in_brief=True)
    else:
        nl._log("warning", "未能从策略报告中解析出目标组合表格")

    return target
