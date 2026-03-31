"""状态定义模块

定义主图状态、ReAct 子图状态及相关 Reducer。
"""

import operator
from dataclasses import dataclass, field as dc_field
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------

def override_reducer(current_value, new_value):
    """状态合并：override 覆盖 | dict 合并 | 其它 operator.add"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    if isinstance(current_value, dict) and isinstance(new_value, dict):
        return {**current_value, **new_value}
    return operator.add(current_value, new_value)


# ---------------------------------------------------------------------------
# 持仓数据模型
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """单只股票持仓"""
    stock_code: str = Field(description="股票代码")
    stock_name: str = Field(description="股票名称")
    sector: str = Field(default="", description="所属行业")
    shares: int = Field(description="持仓股数")
    avg_cost: float = Field(description="平均成本价")
    current_price: float = Field(default=0.0, description="当前价格")
    market_value: float = Field(default=0.0, description="市值")
    weight: float = Field(default=0.0, description="仓位占比")
    unrealized_pnl: float = Field(default=0.0, description="未实现盈亏")
    unrealized_pnl_pct: float = Field(default=0.0, description="未实现盈亏百分比")


class TradeOrder(BaseModel):
    """交易订单"""
    stock_code: str = Field(description="股票代码")
    stock_name: str = Field(description="股票名称")
    direction: Literal["buy", "sell"] = Field(description="交易方向")
    quantity: int = Field(description="交易数量（股）")
    order_type: Literal["market", "limit"] = Field(default="market", description="订单类型")
    limit_price: Optional[float] = Field(default=None, description="限价")
    reason: str = Field(default="", description="交易理由")
    priority: int = Field(default=5, description="优先级 1-10，10最高")


class RiskMetrics(BaseModel):
    """风险指标"""
    portfolio_var_95: float = Field(default=0.0, description="95% VaR（日）")
    portfolio_var_99: float = Field(default=0.0, description="99% VaR（日）")
    max_drawdown: float = Field(default=0.0, description="最大回撤")
    current_drawdown: float = Field(default=0.0, description="当前回撤")
    sharpe_ratio: float = Field(default=0.0, description="夏普比率")
    volatility: float = Field(default=0.0, description="年化波动率")
    beta: float = Field(default=0.0, description="Beta 系数")
    concentration_risk: float = Field(default=0.0, description="集中度风险")
    sector_exposure: Dict[str, float] = Field(default_factory=dict, description="行业暴露")
    risk_alerts: List[str] = Field(default_factory=list, description="风险预警信息")


class StockCandidate(BaseModel):
    """选股候选"""
    stock_code: str = Field(description="股票代码")
    stock_name: str = Field(description="股票名称")
    sector: str = Field(default="", description="所属行业")
    score: float = Field(default=0.0, description="综合评分 0-100")
    target_weight: float = Field(default=0.0, description="建议权重")
    reasons: List[str] = Field(default_factory=list, description="推荐理由")
    risk_factors: List[str] = Field(default_factory=list, description="风险因素")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="关键指标")


# ---------------------------------------------------------------------------
# ReAct 子图状态
# ---------------------------------------------------------------------------

class ResearcherSubState(TypedDict):
    """ReAct 子图状态"""
    researcher_messages: Annotated[List[MessageLikeRepresentation], override_reducer]
    research_topic: str
    tool_call_iterations: int
    task_id: str
    system_prompt: NotRequired[str]
    early_exit_not_applicable: NotRequired[bool]
    early_exit_reason: NotRequired[str]
    completed_steps: NotRequired[List[str]]


class ResearcherOutputState(TypedDict):
    """ReAct 子图输出"""
    compressed_research: str
    raw_notes: Annotated[List[str], override_reducer]
    structured_output: NotRequired[Dict[str, Any]]


# ---------------------------------------------------------------------------
# 主图状态
# ---------------------------------------------------------------------------

class PortfolioState(MessagesState):
    """主图状态"""
    # 市场数据
    market_data: Annotated[Dict[str, Any], override_reducer] = {}

    # 持仓审查
    position_review: Optional[str] = None

    # 分析结果
    macro_analysis: Optional[str] = None
    sector_analysis: Optional[str] = None
    screening_result: Optional[str] = None
    stock_candidates: Annotated[Dict[str, Any], override_reducer] = {}

    # 组合策略
    target_portfolio: Annotated[Dict[str, Any], override_reducer] = {}
    strategy_reasoning: Optional[str] = None

    # 当前持仓
    current_positions: Annotated[Dict[str, Any], override_reducer] = {}
    portfolio_value: float = 0.0
    cash_balance: float = 0.0

    # 风险评估
    risk_metrics: Annotated[Dict[str, Any], override_reducer] = {}
    risk_report: Optional[str] = None

    # 交易执行
    trade_orders: Annotated[List[Dict[str, Any]], operator.add] = []
    execution_report: Optional[str] = None

    # 再平衡
    rebalance_needed: bool = False
    rebalance_report: Optional[str] = None

    # 汇总报告
    final_report: Optional[str] = None
    progress_log: Annotated[List[Dict[str, Any]], operator.add] = []


class PortfolioInputState(MessagesState):
    """图输入"""
    run_mode: Optional[str] = None  # "full_analysis" | "rebalance" | "risk_check"

    current_positions: Annotated[Dict[str, Any], override_reducer] = {}
    portfolio_value: float = 0.0
    cash_balance: float = 0.0
