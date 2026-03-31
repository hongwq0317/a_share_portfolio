"""主图构建模块

构建投资组合的 LangGraph StateGraph，编排所有节点的执行顺序。
支持节点级别的启用/禁用，自动跳过禁用节点。

有持仓时流程:
  START → MarketData → PositionReview → MarketAnalysis → StockScreening
        → PortfolioStrategy → RiskAssessment → TradeExecution
        → RebalanceMonitor → ReportGeneration → END

无持仓时流程:
  START → MarketData → MarketAnalysis → StockScreening → ... (跳过 PositionReview)
"""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.state import PortfolioState, PortfolioInputState
from src.config import get_app_config

logger = logging.getLogger("portfolio")


# ============================================================
# Node 名称常量
# ============================================================

NODE_MARKET_DATA = "market_data"
NODE_POSITION_REVIEW = "position_review"
NODE_MARKET_ANALYSIS = "market_analysis"
NODE_STOCK_SCREENING = "stock_screening"
NODE_PORTFOLIO_STRATEGY = "portfolio_strategy"
NODE_RISK_ASSESSMENT = "risk_assessment"
NODE_TRADE_EXECUTION = "trade_execution"
NODE_REBALANCE_MONITOR = "rebalance_monitor"
NODE_REPORT_GENERATION = "report_generation"

NODE_ORDER = [
    NODE_MARKET_DATA,
    NODE_POSITION_REVIEW,
    NODE_MARKET_ANALYSIS,
    NODE_STOCK_SCREENING,
    NODE_PORTFOLIO_STRATEGY,
    NODE_RISK_ASSESSMENT,
    NODE_TRADE_EXECUTION,
    NODE_REBALANCE_MONITOR,
    NODE_REPORT_GENERATION,
]

NODE_SWITCH_MAP = {
    NODE_MARKET_DATA: "enable_market_data",
    NODE_POSITION_REVIEW: None,
    NODE_MARKET_ANALYSIS: "enable_market_analysis",
    NODE_STOCK_SCREENING: "enable_stock_screening",
    NODE_PORTFOLIO_STRATEGY: "enable_portfolio_strategy",
    NODE_RISK_ASSESSMENT: "enable_risk_assessment",
    NODE_TRADE_EXECUTION: "enable_trade_execution",
    NODE_REBALANCE_MONITOR: "enable_rebalance_monitor",
    NODE_REPORT_GENERATION: None,
}

RUN_MODE_SKIP: dict[str, set[str]] = {
    "full_analysis": set(),
    "rebalance": {
        NODE_MARKET_ANALYSIS, NODE_STOCK_SCREENING, NODE_PORTFOLIO_STRATEGY,
    },
    "risk_check": {
        NODE_STOCK_SCREENING, NODE_PORTFOLIO_STRATEGY,
        NODE_TRADE_EXECUTION, NODE_REBALANCE_MONITOR,
    },
}


# ============================================================
# 路由函数
# ============================================================

def _should_run_position_review(state: PortfolioState) -> bool:
    """判断是否需要执行持仓审查：当前有持仓时执行。"""
    positions = state.get("current_positions", {})
    return bool(positions)


def get_next_enabled_node(
    current_node: str,
    config: RunnableConfig,
    state: PortfolioState | None = None,
) -> str:
    """获取当前节点之后的下一个启用节点。

    同时考虑 config.nodes.enable_* 开关和 state.run_mode 模式跳过规则。
    position_review 节点较为特殊：仅在有持仓时启用，通过 state 判断。
    """
    app_config = get_app_config(config)
    nodes = app_config.nodes

    run_mode = state.get("run_mode", "full_analysis") if state else "full_analysis"
    mode_skip = RUN_MODE_SKIP.get(run_mode, set())

    try:
        current_idx = NODE_ORDER.index(current_node)
    except ValueError:
        return END

    for next_node in NODE_ORDER[current_idx + 1:]:
        if next_node in mode_skip:
            continue

        if next_node == NODE_POSITION_REVIEW:
            if state is not None and _should_run_position_review(state):
                return next_node
            continue

        switch_attr = NODE_SWITCH_MAP.get(next_node)
        if switch_attr is None:
            return next_node
        if getattr(nodes, switch_attr, True):
            return next_node

    return END


def _make_route_fn(current_node: str):
    """创建路由函数的工厂"""
    def route_fn(state: PortfolioState, config: RunnableConfig) -> str:
        return get_next_enabled_node(current_node, config, state)
    route_fn.__name__ = f"route_after_{current_node}"
    return route_fn


def route_entry(state: PortfolioState, config: RunnableConfig) -> str:
    """入口路由：找到第一个启用的节点，同时尊重 run_mode 跳过规则。"""
    app_config = get_app_config(config)
    nodes = app_config.nodes

    run_mode = state.get("run_mode", "full_analysis")
    mode_skip = RUN_MODE_SKIP.get(run_mode, set())
    if mode_skip:
        logger.info(f"[路由] run_mode={run_mode}, 跳过节点: {mode_skip}")

    for node_name in NODE_ORDER:
        if node_name in mode_skip:
            continue
        if node_name == NODE_POSITION_REVIEW:
            if _should_run_position_review(state):
                return node_name
            continue
        switch_attr = NODE_SWITCH_MAP.get(node_name)
        if switch_attr is None:
            return node_name
        if getattr(nodes, switch_attr, True):
            return node_name

    return END


# ============================================================
# 节点包装函数（延迟导入）
# ============================================================

async def _node_market_data(state: PortfolioState, config: RunnableConfig):
    from src.nodes.market_data import fetch_market_data
    return await fetch_market_data(state, config)


async def _node_position_review(state: PortfolioState, config: RunnableConfig):
    from src.nodes.position_review import review_positions
    return await review_positions(state, config)


async def _node_market_analysis(state: PortfolioState, config: RunnableConfig):
    from src.nodes.market_analysis import analyze_market
    return await analyze_market(state, config)


async def _node_stock_screening(state: PortfolioState, config: RunnableConfig):
    from src.nodes.stock_screening import screen_stocks
    return await screen_stocks(state, config)


async def _node_portfolio_strategy(state: PortfolioState, config: RunnableConfig):
    from src.nodes.portfolio_strategy import build_portfolio_strategy
    return await build_portfolio_strategy(state, config)


async def _node_risk_assessment(state: PortfolioState, config: RunnableConfig):
    from src.nodes.risk_assessment import assess_risk
    return await assess_risk(state, config)


async def _node_trade_execution(state: PortfolioState, config: RunnableConfig):
    from src.nodes.trade_execution import execute_trades
    return await execute_trades(state, config)


async def _node_rebalance_monitor(state: PortfolioState, config: RunnableConfig):
    from src.nodes.rebalance_monitor import monitor_and_rebalance
    return await monitor_and_rebalance(state, config)


async def _node_report_generation(state: PortfolioState, config: RunnableConfig):
    from src.nodes.report_generation import generate_report
    return await generate_report(state, config)


# ============================================================
# 主图构建
# ============================================================

def build_portfolio_graph() -> StateGraph:
    """构建投资组合主图"""
    builder = StateGraph(
        PortfolioState,
        input=PortfolioInputState,
    )

    node_fns = {
        NODE_MARKET_DATA: _node_market_data,
        NODE_POSITION_REVIEW: _node_position_review,
        NODE_MARKET_ANALYSIS: _node_market_analysis,
        NODE_STOCK_SCREENING: _node_stock_screening,
        NODE_PORTFOLIO_STRATEGY: _node_portfolio_strategy,
        NODE_RISK_ASSESSMENT: _node_risk_assessment,
        NODE_TRADE_EXECUTION: _node_trade_execution,
        NODE_REBALANCE_MONITOR: _node_rebalance_monitor,
        NODE_REPORT_GENERATION: _node_report_generation,
    }

    for name, fn in node_fns.items():
        builder.add_node(name, fn)

    # 入口条件路由
    all_destinations = {n: n for n in NODE_ORDER}
    all_destinations[END] = END
    builder.add_conditional_edges(START, route_entry, all_destinations)

    # 每个节点后的条件路由
    for i, node_name in enumerate(NODE_ORDER):
        remaining = {n: n for n in NODE_ORDER[i + 1:]}
        remaining[END] = END
        if remaining:
            builder.add_conditional_edges(
                node_name,
                _make_route_fn(node_name),
                remaining,
            )
        else:
            builder.add_edge(node_name, END)

    logger.info("投资组合主图构建完成")
    return builder.compile()


_compiled_graph = None


def get_portfolio_graph():
    """获取编译好的投资组合图（单例）"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_portfolio_graph()
    return _compiled_graph
