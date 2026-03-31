"""报告生成节点

汇总各阶段分析结果，生成投资组合综合报告。
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config, get_configured_model
from src.prompts import FINAL_REPORT_PROMPT
from src.state import PortfolioState
from src.utils import NodeLogger, extract_text_from_content, log_token_usage
from src.portfolio_persistence import format_positions_summary, get_trade_history_summary

logger = logging.getLogger("portfolio")


async def generate_report(state: PortfolioState, config: RunnableConfig) -> dict:
    """生成投资组合综合报告"""
    app_config = get_app_config(config)
    nl = NodeLogger("Report", app_config.logging.detail_level)
    nl.start("生成综合报告")

    sections = []
    section_stats = []

    current_positions = state.get("current_positions", {})
    cash_balance = state.get("cash_balance", 0)
    portfolio_value = state.get("portfolio_value", 0)
    if current_positions:
        pos_summary = format_positions_summary({
            "current_positions": current_positions,
            "cash_balance": cash_balance,
            "portfolio_value": portfolio_value,
        })
        trade_history = get_trade_history_summary(10)
        portfolio_section = f"### 当前持仓\n{pos_summary}"
        if "无历史交易" not in trade_history:
            portfolio_section += f"\n\n### 交易记录\n{trade_history}"
        sections.append(portfolio_section)
        section_stats.append(f"持仓({len(current_positions)}只)")

    pos_review = state.get("position_review")
    if pos_review and "无需审查" not in pos_review:
        sections.append(f"### 持仓审查\n{pos_review}")
        section_stats.append(f"持仓审查({len(pos_review)}字符)")

    macro = state.get("macro_analysis")
    if macro:
        sections.append(f"### 市场分析\n{macro}")
        section_stats.append(f"市场分析({len(macro)}字符)")

    screening = state.get("screening_result")
    if screening:
        sections.append(f"### 选股分析\n{screening}")
        section_stats.append(f"选股({len(screening)}字符)")

    strategy = state.get("strategy_reasoning")
    if strategy:
        sections.append(f"### 组合策略\n{strategy}")
        section_stats.append(f"策略({len(strategy)}字符)")

    risk = state.get("risk_report")
    if risk:
        sections.append(f"### 风险评估\n{risk}")
        section_stats.append(f"风控({len(risk)}字符)")

    execution = state.get("execution_report")
    if execution:
        sections.append(f"### 交易执行\n{execution}")
        section_stats.append(f"交易({len(execution)}字符)")

    rebalance = state.get("rebalance_report")
    if rebalance:
        sections.append(f"### 再平衡\n{rebalance}")
        section_stats.append(f"再平衡({len(rebalance)}字符)")

    nl.input_summary(f"可用章节: {len(sections)}个 — {', '.join(section_stats)}")

    if not sections:
        nl._log("warning", "⚠ 无可用材料生成报告")
        nl.complete()
        return {"final_report": "无可用数据生成报告"}

    all_sections = "\n\n---\n\n".join(sections)
    total_input_chars = sum(len(s) for s in sections)
    nl._log("info", f"汇总输入: {total_input_chars:,}字符")
    today = datetime.now().strftime("%Y年%m月%d日")

    report_prompt = FINAL_REPORT_PROMPT.format(
        sections=all_sections,
        date=today,
    )

    report_model = get_configured_model(app_config, role="report")
    messages = [
        SystemMessage(content="你是一位专业的投资报告撰写专家。"),
        HumanMessage(content=report_prompt),
    ]

    import time as _time
    final_report = ""
    for attempt in range(1, 4):
        try:
            nl._log("info", f"报告生成尝试 {attempt}/3 ...")
            start = _time.time()
            response = await report_model.ainvoke(messages)
            elapsed = _time.time() - start
            log_token_usage(response, "Report", f"generate/attempt{attempt}", messages=messages)
            final_report = extract_text_from_content(response.content)
            nl._log("info", f"报告生成成功: {len(final_report):,}字符, 耗时{elapsed:.1f}s")
            break
        except Exception as exc:
            nl.error(f"报告生成失败 (尝试 {attempt}/3): {exc}")
            if attempt == 3:
                final_report = f"# 投资组合日报\n日期: {today}\n\n{all_sections}"
                nl._log("info", "使用原始材料拼接作为兜底报告")

    nl.output_summary(f"最终报告: {len(final_report):,}字符")
    nl.complete()
    return {
        "final_report": final_report,
        "progress_log": [{"node": "report_generation", "status": "completed"}],
    }
