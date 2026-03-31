"""A股组合投资系统 - 执行入口

支持多种运行模式:
  - full: 完整流程（市场分析→选股→组合构建→风控→交易→报告）
  - rebalance: 仅再平衡检查
  - risk_check: 仅风险检查
  - schedule: 定时自动运行

使用方式:
    # 完整分析流程
    python run_portfolio.py

    # 仅执行再平衡
    python run_portfolio.py --mode rebalance

    # 仅风险检查
    python run_portfolio.py --mode risk_check

    # 指定配置文件
    python run_portfolio.py -c custom_config.yaml

    # 定时调度模式
    python run_portfolio.py --mode schedule
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, AppConfig
from src.utils import setup_logging, GraphLogger
from src.portfolio_persistence import load_for_graph, format_positions_summary, get_trade_history_summary

logger = logging.getLogger("portfolio")


# ============================================================
# 配置构建
# ============================================================

def build_configurable(config: AppConfig) -> Dict[str, Any]:
    """将 AppConfig 转换为 LangGraph 的 configurable 字典"""
    return {
        "app_config": config,
        "thread_id": str(uuid.uuid4()),
    }


# ============================================================
# 运行模式
# ============================================================

def _load_existing_portfolio() -> Dict[str, Any]:
    """加载已有持仓数据，刷新价格，返回可注入 input_state 的字段。"""
    portfolio_fields = load_for_graph(do_refresh=True)

    n = len(portfolio_fields.get("current_positions", {}))
    if n > 0:
        logger.info(f"[启动] 已加载 {n} 只持仓到图状态")
        summary = format_positions_summary(portfolio_fields)
        logger.info(f"[启动] 持仓快照:\n{summary}")
        trade_summary = get_trade_history_summary(10)
        if "无历史交易" not in trade_summary:
            logger.info(f"[启动] 交易历史:\n{trade_summary}")
    else:
        logger.info("[启动] 当前无持仓（新组合）")

    return portfolio_fields


async def run_full_analysis(config: AppConfig) -> Dict[str, Any]:
    """完整分析流程"""
    configurable = build_configurable(config)
    graph_logger = GraphLogger()
    graph_logger.start("full_analysis")

    enabled_nodes = []
    if config.nodes.enable_market_data:
        enabled_nodes.append("MarketData")
    enabled_nodes.append("PositionReview(有持仓时)")
    if config.nodes.enable_market_analysis:
        enabled_nodes.append("Analysis")
    if config.nodes.enable_stock_screening:
        enabled_nodes.append("Screening")
    if config.nodes.enable_portfolio_strategy:
        enabled_nodes.append("Strategy")
    if config.nodes.enable_risk_assessment:
        enabled_nodes.append("Risk")
    if config.nodes.enable_trade_execution:
        enabled_nodes.append("Trade")
    if config.nodes.enable_rebalance_monitor:
        enabled_nodes.append("Rebalance")
    enabled_nodes.append("Report")
    graph_logger.log_enabled_nodes(enabled_nodes)

    from src.graph import build_portfolio_graph
    graph = build_portfolio_graph()

    portfolio_fields = _load_existing_portfolio()

    input_state = {
        "messages": [],
        "run_mode": "full_analysis",
        **portfolio_fields,
    }
    run_config = {
        "configurable": configurable,
        "recursion_limit": 120,
    }

    try:
        result = await graph.ainvoke(input_state, run_config)
        graph_logger.complete()
        return {
            "success": True,
            "mode": "full_analysis",
            "position_review": result.get("position_review", ""),
            "macro_analysis": result.get("macro_analysis", ""),
            "screening_result": result.get("screening_result", ""),
            "strategy_reasoning": result.get("strategy_reasoning", ""),
            "risk_report": result.get("risk_report", ""),
            "execution_report": result.get("execution_report", ""),
            "rebalance_report": result.get("rebalance_report", ""),
            "final_report": result.get("final_report", ""),
        }
    except Exception as e:
        logger.error(f"分析执行失败: {e}", exc_info=True)
        graph_logger.complete()
        return {"success": False, "error": str(e)}


async def run_rebalance_check(config: AppConfig) -> Dict[str, Any]:
    """仅执行再平衡检查"""
    config.nodes.enable_market_data = True
    config.nodes.enable_market_analysis = False
    config.nodes.enable_stock_screening = False
    config.nodes.enable_portfolio_strategy = False
    config.nodes.enable_risk_assessment = True
    config.nodes.enable_trade_execution = False
    config.nodes.enable_rebalance_monitor = True

    configurable = build_configurable(config)
    graph_logger = GraphLogger()
    graph_logger.start("rebalance")

    from src.graph import build_portfolio_graph
    graph = build_portfolio_graph()

    portfolio_fields = _load_existing_portfolio()

    input_state = {"messages": [], "run_mode": "rebalance", **portfolio_fields}
    run_config = {"configurable": configurable, "recursion_limit": 80}

    try:
        result = await graph.ainvoke(input_state, run_config)
        graph_logger.complete()
        return {
            "success": True,
            "mode": "rebalance",
            "position_review": result.get("position_review", ""),
            "macro_analysis": result.get("macro_analysis", ""),
            "screening_result": result.get("screening_result", ""),
            "strategy_reasoning": result.get("strategy_reasoning", ""),
            "risk_report": result.get("risk_report", ""),
            "execution_report": result.get("execution_report", ""),
            "rebalance_report": result.get("rebalance_report", ""),
            "final_report": result.get("final_report", ""),
        }
    except Exception as e:
        logger.error(f"再平衡检查失败: {e}", exc_info=True)
        graph_logger.complete()
        return {"success": False, "error": str(e)}


async def run_risk_check(config: AppConfig) -> Dict[str, Any]:
    """仅执行风险检查"""
    config.nodes.enable_market_data = True
    config.nodes.enable_market_analysis = False
    config.nodes.enable_stock_screening = False
    config.nodes.enable_portfolio_strategy = False
    config.nodes.enable_risk_assessment = True
    config.nodes.enable_trade_execution = False
    config.nodes.enable_rebalance_monitor = False

    configurable = build_configurable(config)
    graph_logger = GraphLogger()
    graph_logger.start("risk_check")

    from src.graph import build_portfolio_graph
    graph = build_portfolio_graph()

    portfolio_fields = _load_existing_portfolio()

    input_state = {"messages": [], "run_mode": "risk_check", **portfolio_fields}
    run_config = {"configurable": configurable, "recursion_limit": 60}

    try:
        result = await graph.ainvoke(input_state, run_config)
        graph_logger.complete()
        return {
            "success": True,
            "mode": "risk_check",
            "position_review": result.get("position_review", ""),
            "macro_analysis": result.get("macro_analysis", ""),
            "screening_result": result.get("screening_result", ""),
            "strategy_reasoning": result.get("strategy_reasoning", ""),
            "risk_report": result.get("risk_report", ""),
            "execution_report": result.get("execution_report", ""),
            "rebalance_report": result.get("rebalance_report", ""),
            "final_report": result.get("final_report", ""),
        }
    except Exception as e:
        logger.error(f"风险检查失败: {e}", exc_info=True)
        graph_logger.complete()
        return {"success": False, "error": str(e)}


# ============================================================
# 定时调度
# ============================================================

def start_scheduler(config: AppConfig):
    """启动定时调度"""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("请安装 apscheduler: pip install apscheduler")
        return

    scheduler = AsyncIOScheduler()

    # 每个交易日早盘前分析
    analysis_hour, analysis_min = config.schedule.analysis_time.split(":")
    scheduler.add_job(
        run_full_analysis, CronTrigger(
            day_of_week="mon-fri",
            hour=int(analysis_hour), minute=int(analysis_min),
        ),
        args=[config], id="daily_analysis",
    )

    # 收盘后风险检查
    summary_hour, summary_min = config.schedule.summary_time.split(":")
    scheduler.add_job(
        run_risk_check, CronTrigger(
            day_of_week="mon-fri",
            hour=int(summary_hour), minute=int(summary_min),
        ),
        args=[config], id="daily_risk_check",
    )

    # 周度再平衡
    if config.schedule.rebalance_frequency == "weekly":
        scheduler.add_job(
            run_rebalance_check, CronTrigger(
                day_of_week="fri", hour=14, minute=30,
            ),
            args=[config], id="weekly_rebalance",
        )
    elif config.schedule.rebalance_frequency == "daily":
        scheduler.add_job(
            run_rebalance_check, CronTrigger(
                day_of_week="mon-fri", hour=14, minute=30,
            ),
            args=[config], id="daily_rebalance",
        )

    logger.info("定时调度已启动:")
    logger.info(f"  每日分析: {config.schedule.analysis_time} (工作日)")
    logger.info(f"  风险检查: {config.schedule.summary_time} (工作日)")
    logger.info(f"  再平衡: {config.schedule.rebalance_frequency}")

    scheduler.start()

    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("调度已停止")


# ============================================================
# 回测分析
# ============================================================

async def run_backtest_analysis(config: AppConfig) -> Dict[str, Any]:
    """运行回测分析"""
    from src.backtest import run_backtest, format_backtest_report
    from src.portfolio_persistence import load_raw

    logger.info("=" * 60)
    logger.info("回测分析模式")
    logger.info("=" * 60)

    raw_state = load_raw()
    if not raw_state or not raw_state.get("trade_history"):
        logger.error("无交易记录，无法运行回测")
        return {"success": False, "error": "无交易记录"}

    result = run_backtest(
        raw_state,
        benchmark_code=config.portfolio.benchmark,
    )

    if "error" in result:
        return {"success": False, "error": result["error"]}

    report = format_backtest_report(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/backtest_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"回测报告已保存: {report_path} ({len(report)}字符)")

    print(f"\n{'=' * 60}")
    print("回测分析完成")
    print(f"{'=' * 60}")
    print(f"  回测区间: {result['period']['start']} ~ {result['period']['end']}")
    print(f"  交易天数: {result['period']['trading_days']}")
    print(f"  初始资金: {result['initial_capital']:,.0f}元")
    print(f"  当前净值: {result['current_value']:,.0f}元")
    print(f"  累计收益: {result['returns'].get('total_return_pct', 0):+.2f}%")
    if result.get("risk"):
        print(f"  最大回撤: {result['risk'].get('max_drawdown_pct', 0):.2f}%")
        print(f"  夏普比率: {result['risk'].get('sharpe_ratio', 0):.3f}")
    if result.get("benchmark", {}).get("available"):
        bm = result["benchmark"]
        print(f"  基准收益: {bm['benchmark_return_pct']:+.2f}% (沪深300)")
        print(f"  超额收益: {bm['excess_return_pct']:+.2f}%")
    print(f"\n  完整报告: {report_path}")

    return {"success": True, "backtest_result": result, "report_path": report_path}


# ============================================================
# 结果保存
# ============================================================

NODE_REPORT_FIELDS = {
    "position_review": "position_review",
    "market_analysis": "macro_analysis",
    "stock_screening": "screening_result",
    "portfolio_strategy": "strategy_reasoning",
    "risk_assessment": "risk_report",
    "trade_execution": "execution_report",
    "rebalance_monitor": "rebalance_report",
    "report_generation": "final_report",
}


def save_report(result: Dict[str, Any], output_path: str):
    """保存报告到文件"""
    report = result.get("final_report", "")
    if not report:
        logger.warning("没有生成报告内容，跳过保存")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_path.endswith(".json"):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

    logger.info(f"报告已保存到: {output_path} ({len(report)} 字符)")


NODE_DISPLAY_NAMES = {
    "position_review": "00_持仓审查",
    "market_analysis": "01_市场分析",
    "stock_screening": "02_选股报告",
    "portfolio_strategy": "03_组合策略",
    "risk_assessment": "04_风险评估",
    "trade_execution": "05_交易执行",
    "rebalance_monitor": "06_再平衡",
    "report_generation": "07_综合报告",
}


def save_node_reports(result: Dict[str, Any], output_dir: str):
    """将各节点报告保存为独立 markdown 文件，方便直接阅读。"""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for node_name, state_field in NODE_REPORT_FIELDS.items():
        content = result.get(state_field, "")
        if not content:
            continue

        display = NODE_DISPLAY_NAMES.get(node_name, node_name)
        filepath = os.path.join(output_dir, f"{display}.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        saved += 1

    if saved:
        logger.info(f"节点报告已保存到: {output_dir}/ (共 {saved} 个节点)")
    else:
        logger.warning("没有节点报告内容，跳过保存")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="A股组合投资系统 - AI 驱动的自动化投资组合管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_portfolio.py                           # 完整分析流程
  python run_portfolio.py --mode rebalance          # 仅再平衡
  python run_portfolio.py --mode risk_check         # 仅风险检查
  python run_portfolio.py --mode schedule           # 定时调度
  python run_portfolio.py -c custom.yaml -o report.md
        """,
    )

    parser.add_argument("--config", "-c", default="portfolio_config.yaml",
                        help="配置文件路径 (默认: portfolio_config.yaml)")
    parser.add_argument("--mode", "-m",
                        choices=["full", "rebalance", "risk_check", "schedule", "backtest"],
                        default="full",
                        help="运行模式 (默认: full)")
    parser.add_argument("--output", "-o", default=None,
                        help="输出文件路径 (支持 .md / .json)")
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金（覆盖配置文件）")
    parser.add_argument("--style",
                        choices=["aggressive", "balanced", "conservative"],
                        default=None,
                        help="投资风格（覆盖配置文件）")
    parser.add_argument("--log-level", "-l",
                        choices=["full", "normal", "brief"],
                        default=None, help="日志等级")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细日志输出")

    return parser.parse_args()


async def main():
    """主入口函数"""
    args = parse_args()
    config = load_config(args.config)

    setup_logging(log_dir="logs", file_mode=config.logging.file_mode)
    if args.verbose:
        logging.getLogger("portfolio").setLevel(logging.DEBUG)

    if args.log_level:
        config.logging.detail_level = args.log_level
    if args.capital:
        config.portfolio.initial_capital = args.capital
    if args.style:
        config.portfolio.style = args.style

    logger.info("=" * 60)
    logger.info("A股组合投资系统 启动")
    logger.info(f"  运行模式: {args.mode}")
    logger.info(f"  初始资金: {config.portfolio.initial_capital:,.0f}元")
    logger.info(f"  投资风格: {config.portfolio.style}")
    logger.info(f"  交易模式: {config.trading.mode}")
    logger.info("=" * 60)

    if args.mode == "schedule":
        start_scheduler(config)
        return

    if args.mode == "backtest":
        await run_backtest_analysis(config)
        return

    mode_map = {
        "full": run_full_analysis,
        "rebalance": run_rebalance_check,
        "risk_check": run_risk_check,
    }
    runner = mode_map[args.mode]
    result = await runner(config)

    if result.get("success"):
        report = result.get("final_report", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)

        if report:
            logger.info(f"\n{'='*60}")
            logger.info(f"报告生成完成 | 长度: {len(report)} 字符")
            logger.info(f"{'='*60}")

            if args.output:
                save_report(result, args.output)
            else:
                save_report(result, f"reports/{args.mode}_{timestamp}.md")

        node_reports_dir = f"reports/{args.mode}_{timestamp}_nodes"
        save_node_reports(result, node_reports_dir)

        if not report:
            logger.warning("运行完成但未生成最终汇总报告")
    else:
        logger.error(f"运行失败: {result.get('error', '未知错误')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
