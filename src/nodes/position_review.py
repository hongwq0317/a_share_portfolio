"""持仓审查节点

有持仓时优先于市场分析执行，对每只持仓股进行多维度深度诊断，
输出"继续持有/密切观察/建议减持/建议清仓"结论和调仓紧迫度。

流程位置: MarketData → **PositionReview** → MarketAnalysis → ...
仅在 current_positions 非空时才被图路由调度。

优化: 行情和技术指标在调用 LLM 前即由程序预查完毕，
LLM 专注于 web_search（新闻/风险事件）+ 综合诊断分析。
"""

import logging

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.utils import NodeLogger
from src.nodes.base_react_subgraph import ReActSubgraphConfig, build_react_subgraph
from src.prompts import (
    POSITION_REVIEW_SYSTEM,
    POSITION_REVIEW_COMPRESS,
    POSITION_REVIEW_SYNTHESIZE,
)
from src.state import PortfolioState
from src.tools.market_tools import MARKET_TOOLS
from src.tools.analysis_tools import ANALYSIS_TOOLS
from src.tools.search_tools import web_search
from src.tools.skill_tools import SKILL_TOOLS, build_available_skills_prompt
from src.portfolio_persistence import format_positions_summary

logger = logging.getLogger("portfolio")


def _build_review_subgraph():
    """构建持仓审查 ReAct 子图"""
    all_tools = SKILL_TOOLS + MARKET_TOOLS + ANALYSIS_TOOLS + [web_search]
    cfg = ReActSubgraphConfig(
        node_name="PositionReview",
        tools=all_tools,
        system_prompt_template=POSITION_REVIEW_SYSTEM,
        compress_prompt=POSITION_REVIEW_COMPRESS,
        synthesize_prompt=POSITION_REVIEW_SYNTHESIZE,
        compression_switch_attr="enable_analysis_compression",
        max_iterations_attr="screening_max_iterations",
        model_role="research",
    )
    return build_react_subgraph(cfg)


_review_subgraph = None


def _get_review_subgraph():
    global _review_subgraph
    if _review_subgraph is None:
        _review_subgraph = _build_review_subgraph()
    return _review_subgraph


def _format_market_brief(market_data: dict) -> str:
    """从已获取的市场数据中提取简要概况供持仓审查参考。"""
    lines = []
    for key, val in market_data.items():
        if key.startswith("index_"):
            lines.append(
                f"{val['name']}: {val['price']:.2f} ({val['change_pct']:+.2f}%)"
            )

    sentiment = market_data.get("sentiment", {})
    if sentiment:
        lines.append(
            f"市场情绪: 上涨{sentiment.get('up_count', 0)}家, "
            f"下跌{sentiment.get('down_count', 0)}家, "
            f"涨停{sentiment.get('limit_up', 0)}, 跌停{sentiment.get('limit_down', 0)}"
        )

    nb = market_data.get("northbound", {})
    if nb:
        lines.append(f"北向资金: 合计{nb['total_net']:+.2f}亿")

    return "\n".join(lines) if lines else "市场数据尚未获取"


# ------------------------------------------------------------------
# 预查: 行情 + 技术指标（程序化，不消耗 LLM 迭代轮次）
# ------------------------------------------------------------------

def _prefetch_fund_flow(codes: list[str], nl: NodeLogger) -> str:
    """批量获取个股资金流向，返回格式化文本。"""
    try:
        from src.tools.data_provider import fetch_stock_fund_flow
    except ImportError:
        return "资金流向模块不可用"

    results = []
    success_count = 0
    for code in codes:
        try:
            df = fetch_stock_fund_flow(code, days=3)
            if not df.empty:
                latest = df.iloc[-1]
                main_flow = latest.get("主力净流入_万", 0)
                results.append(
                    f"  {code}: 主力净流入={main_flow:+.0f}万"
                )
                success_count += 1
            else:
                results.append(f"  {code}: 暂无数据")
        except Exception:
            results.append(f"  {code}: 获取失败")

    nl._log("info", f"预查资金流向完成: {success_count}/{len(codes)}只")
    return "\n".join(results) if results else "资金流向数据全部不可用"


def _prefetch_realtime_quotes(codes: list[str], nl: NodeLogger) -> str:
    """批量获取实时行情，返回格式化文本。"""
    try:
        from src.tools.data_provider import fetch_stock_quote
        quotes = fetch_stock_quote(codes)
    except Exception as e:
        nl._log("warning", f"预查实时行情失败: {e}")
        return "实时行情获取失败"

    if not quotes:
        return "实时行情获取失败（无数据）"

    lines = []
    for code in codes:
        q = quotes.get(code)
        if not q:
            lines.append(f"  {code}: 无数据")
            continue
        lines.append(
            f"  {q['name']}({code}): 现价={q['price']}, "
            f"涨跌幅={q['change_pct']:+.2f}%, "
            f"成交额={q['amount']/1e8:.2f}亿, "
            f"换手率={q['turnover']}%, "
            f"PE={q['pe']}, PB={q['pb']}, "
            f"今开={q['open']}, 最高={q['high']}, 最低={q['low']}"
        )

    nl._log("info", f"预查实时行情完成: {len(quotes)}/{len(codes)}只")
    return "\n".join(lines)


def _prefetch_technical_indicators(codes: list[str], nl: NodeLogger) -> str:
    """批量计算技术指标，返回格式化文本。"""
    from src.tools.data_provider import fetch_stock_history

    results = []
    success_count = 0

    for code in codes:
        try:
            df = fetch_stock_history(code, days=180)
            if df.empty or len(df) < 30:
                results.append(f"  {code}: 数据不足")
                continue

            close = df["收盘"].values
            volume = df["成交量"].values

            ma5 = pd.Series(close).rolling(5).mean().iloc[-1]
            ma10 = pd.Series(close).rolling(10).mean().iloc[-1]
            ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
            ma60 = pd.Series(close).rolling(60).mean().iloc[-1] if len(close) >= 60 else None

            ema12 = pd.Series(close).ewm(span=12).mean()
            ema26 = pd.Series(close).ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()
            macd_hist = 2 * (dif - dea)

            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            bb_mid = pd.Series(close).rolling(20).mean()
            bb_std = pd.Series(close).rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std

            vol_ma5 = pd.Series(volume).rolling(5).mean().iloc[-1]
            vol_ratio = volume[-1] / vol_ma5 if vol_ma5 > 0 else 0

            pct_5d = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
            pct_20d = (close[-1] / close[-21] - 1) * 100 if len(close) > 20 else 0

            current = close[-1]
            ma_trend = (
                "多头排列" if ma5 > ma10 > ma20
                else ("空头排列" if ma5 < ma10 < ma20 else "震荡")
            )

            support = bb_lower.iloc[-1]
            resistance = bb_upper.iloc[-1]
            dist_support = (current / support - 1) * 100 if support > 0 else 0
            dist_resistance = (current / resistance - 1) * 100 if resistance > 0 else 0

            results.append(
                f"  【{code}】价={current:.2f} | "
                f"MA趋势={ma_trend} "
                f"MA5={ma5:.2f}/MA10={ma10:.2f}/MA20={ma20:.2f}"
                f"{f'/MA60={ma60:.2f}' if ma60 else ''} | "
                f"MACD柱={macd_hist.iloc[-1]:.3f} DIF={dif.iloc[-1]:.3f} | "
                f"RSI={rsi.iloc[-1]:.1f} | "
                f"布林={bb_lower.iloc[-1]:.2f}~{bb_upper.iloc[-1]:.2f} "
                f"距支撑{dist_support:+.1f}%/距压力{dist_resistance:+.1f}% | "
                f"量比={vol_ratio:.2f}"
                f"({'放量' if vol_ratio > 1.5 else '缩量' if vol_ratio < 0.7 else '正常'}) | "
                f"5日{pct_5d:+.1f}% 20日{pct_20d:+.1f}%"
            )
            success_count += 1
        except Exception as e:
            results.append(f"  {code}: 计算失败({e})")

    nl._log("info", f"预查技术指标完成: {success_count}/{len(codes)}只")
    return "\n".join(results)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

async def review_positions(state: PortfolioState, config: RunnableConfig) -> dict:
    """对当前持仓进行深度健康度诊断。"""
    app_config = get_app_config(config)
    nl = NodeLogger("PositionReview", app_config.logging.detail_level)
    risk_cfg = app_config.risk

    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    total_value = state.get("portfolio_value", 0)
    market_data = state.get("market_data", {})

    if not current_positions:
        nl._log("info", "无持仓，跳过持仓审查")
        return {
            "position_review": "当前无持仓，无需审查。",
            "progress_log": [{"node": "position_review", "status": "skipped"}],
        }

    positions_str = format_positions_summary({
        "current_positions": current_positions,
        "cash_balance": cash_balance,
        "portfolio_value": total_value,
    })

    market_brief = _format_market_brief(market_data)

    nl.input_summary(
        f"持仓{len(current_positions)}只, 总资产{total_value:,.0f}元"
    )

    # ---- 程序化预查: 行情 + 技术指标 ----
    code_list = list(current_positions.keys())
    nl._log("info", f"开始预查 {len(code_list)} 只持仓的行情、技术指标和资金流向...")
    quotes_text = _prefetch_realtime_quotes(code_list, nl)
    tech_text = _prefetch_technical_indicators(code_list, nl)
    fund_flow_text = _prefetch_fund_flow(code_list, nl)

    system_prompt = POSITION_REVIEW_SYSTEM.format(
        current_positions=positions_str,
        market_summary=market_brief,
        stop_loss_pct=risk_cfg.stop_loss_pct,
        take_profit_pct=risk_cfg.take_profit_pct,
        trailing_stop_pct=getattr(risk_cfg, "trailing_stop_pct", 8.0),
        max_iterations=getattr(app_config.react, "screening_max_iterations", 12),
        available_skills=build_available_skills_prompt(),
    )

    stock_list = ", ".join(
        f"{p['stock_name']}({code})" for code, p in current_positions.items()
    )

    subgraph = _get_review_subgraph()
    sub_input = {
        "researcher_messages": [
            HumanMessage(content=(
                f"请对以下{len(current_positions)}只持仓股进行深度诊断：\n"
                f"{stock_list}\n\n"
                f"## 当前持仓明细\n{positions_str}\n\n"
                f"## 当前市场概况\n{market_brief}\n\n"
                f"## 实时行情数据（已预查，无需再调用工具）\n{quotes_text}\n\n"
                f"## 技术指标数据（已预查，无需再调用工具）\n{tech_text}\n\n"
                f"## 资金流向数据（已预查，无需再调用工具）\n{fund_flow_text}\n\n"
                f"## 你的工作重点\n"
                f"行情、技术指标和资金流向已经预先查好了，你不需要再调用 "
                f"get_stock_realtime_quote、calculate_technical_indicators、"
                f"batch_technical_indicators 或 batch_get_stock_fund_flow。\n"
                f"请把全部迭代轮次用于：\n"
                f"1. 用 web_search 搜索每只持仓股的近期新闻、风险事件、研报评级"
                f"（每轮可并行搜索多只）\n"
                f"2. 用 think_tool 综合所有数据进行5维度评分和诊断\n"
                f"3. 输出完整诊断报告\n\n"
                f"## 诊断要求\n"
                f"- 逐只进行5维度评分（基本面/技术面/风控/资金面/事件风险）\n"
                f"- 不因短期波动（≤3%）轻言卖出\n"
                f"- T+1锁定股不可建议卖出\n"
                f"- 最终给出调仓紧迫度判定和对后续节点的具体建议"
            ))
        ],
        "research_topic": "持仓审查",
        "tool_call_iterations": 0,
        "task_id": "position_review",
        "system_prompt": system_prompt,
    }

    result = await subgraph.ainvoke(sub_input, config)
    review_report = result.get("compressed_research", "")

    nl.output_summary(f"持仓审查报告: {len(review_report):,}字符")

    return {
        "position_review": review_report,
        "progress_log": [{"node": "position_review", "status": "completed"}],
    }
