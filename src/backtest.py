"""回测分析引擎

基于 portfolio_state.json 中的交易历史，回溯计算：
  1. 每日组合净值曲线（从首笔交易起到今日）
  2. 收益指标：累计收益率、年化收益率、超额收益
  3. 风险指标：最大回撤、年化波动率、夏普比率、Sortino 比率
  4. 交易统计：胜率、盈亏比、平均持仓天数、换手率
  5. 持仓分析：个股贡献、行业分布
  6. 基准对比：vs 沪深300

用法:
    python run_portfolio.py --mode backtest
"""

import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio")


def run_backtest(
    portfolio_state: dict,
    benchmark_code: str = "000300",
    risk_free_rate: float = 0.02,
) -> dict:
    """执行完整回测分析。

    Args:
        portfolio_state: portfolio_state.json 的原始内容
        benchmark_code: 基准指数代码
        risk_free_rate: 无风险利率（年化）

    Returns:
        包含所有回测结果的字典
    """
    trade_history = portfolio_state.get("trade_history", [])
    if not trade_history:
        logger.warning("[回测] 无交易记录，无法回测")
        return {"error": "无交易记录"}

    initial_capital = portfolio_state.get("initial_capital", 1_000_000)
    current_cash = portfolio_state.get("cash", 0)
    current_positions = portfolio_state.get("positions", {})
    closed_positions = portfolio_state.get("closed_positions", [])

    first_trade_date = trade_history[0]["time"][:10]
    logger.info(f"[回测] 开始回测分析: 首笔交易 {first_trade_date}, 初始资金 {initial_capital:,.0f}元")

    # 1. 获取所有涉及股票的历史价格
    all_codes = _collect_all_codes(trade_history, current_positions)
    price_data = _fetch_all_history(all_codes, first_trade_date)
    benchmark_data = _fetch_benchmark_history(benchmark_code, first_trade_date)

    # 2. 重建每日组合净值
    daily_values = _reconstruct_daily_values(
        trade_history, price_data, initial_capital, first_trade_date,
    )

    # 3. 计算收益指标
    returns_metrics = _calculate_returns(daily_values, initial_capital, risk_free_rate)

    # 4. 计算风险指标
    risk_metrics = _calculate_risk_metrics(daily_values, risk_free_rate)

    # 5. 基准对比
    benchmark_metrics = _calculate_benchmark(daily_values, benchmark_data, benchmark_code)

    # 6. 交易统计
    trade_stats = _calculate_trade_stats(trade_history, closed_positions)

    # 7. 持仓分析
    position_analysis = _analyze_positions(
        current_positions, closed_positions, trade_history,
    )

    result = {
        "period": {
            "start": first_trade_date,
            "end": datetime.now().strftime("%Y-%m-%d"),
            "trading_days": len(daily_values),
        },
        "initial_capital": initial_capital,
        "current_value": daily_values[-1]["total_value"] if daily_values else initial_capital,
        "daily_values": daily_values,
        "returns": returns_metrics,
        "risk": risk_metrics,
        "benchmark": benchmark_metrics,
        "trades": trade_stats,
        "positions": position_analysis,
    }

    logger.info(f"[回测] 分析完成: {len(daily_values)}个交易日, "
                f"累计收益{returns_metrics.get('total_return_pct', 0):+.2f}%")
    return result


def _collect_all_codes(trade_history: list, positions: dict) -> list:
    """收集所有交易过和当前持有的股票代码。"""
    codes = set(positions.keys())
    for t in trade_history:
        codes.add(t["code"])
    return sorted(codes)


def _fetch_all_history(codes: list, start_date: str) -> Dict[str, pd.DataFrame]:
    """获取所有股票从 start_date 到今天的历史价格。"""
    from src.tools.data_provider import fetch_stock_history

    days_needed = (datetime.now() - datetime.strptime(start_date, "%Y-%m-%d")).days + 10
    price_data = {}
    for code in codes:
        try:
            df = fetch_stock_history(
                code, days=max(days_needed, 120),
                start_date=start_date,
            )
            if not df.empty:
                price_data[code] = df
                logger.info(f"[回测] 获取 {code} 历史数据: {len(df)}天 ({df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]})")
        except Exception as e:
            logger.warning(f"[回测] 获取 {code} 历史数据失败: {e}")

    return price_data


def _fetch_benchmark_history(benchmark_code: str, start_date: str) -> pd.DataFrame:
    """获取基准指数历史数据。"""
    from src.tools.data_provider import fetch_index_history

    days_needed = (datetime.now() - datetime.strptime(start_date, "%Y-%m-%d")).days + 10
    try:
        df = fetch_index_history(benchmark_code, days=max(days_needed, 252))
        if not df.empty:
            df = df[df["日期"] >= start_date].reset_index(drop=True)
            logger.info(f"[回测] 基准{benchmark_code}: {len(df)}天")
        return df
    except Exception as e:
        logger.warning(f"[回测] 获取基准{benchmark_code}失败: {e}")
        return pd.DataFrame()


def _reconstruct_daily_values(
    trade_history: list,
    price_data: Dict[str, pd.DataFrame],
    initial_capital: float,
    start_date: str,
) -> List[dict]:
    """从交易记录重建每日组合净值。

    逻辑：按日期遍历，跟踪每天的持仓和现金变化，用收盘价计算总资产。
    """
    # 按日期分组交易
    trades_by_date: Dict[str, list] = defaultdict(list)
    for t in trade_history:
        date = t["time"][:10]
        trades_by_date[date].append(t)

    # 获取所有交易日（从价格数据中合并）
    all_dates = set()
    for df in price_data.values():
        if not df.empty:
            all_dates.update(df["日期"].tolist())

    if not all_dates:
        return []

    all_dates = sorted(d for d in all_dates if d >= start_date)

    # 为每只股票建立日期→收盘价映射
    price_map: Dict[str, Dict[str, float]] = {}
    for code, df in price_data.items():
        price_map[code] = dict(zip(df["日期"], df["收盘"]))

    # 逐日重建
    cash = initial_capital
    holdings: Dict[str, dict] = {}  # code -> {shares, avg_cost}
    daily_values = []

    for date in all_dates:
        # 执行当天的交易
        for t in trades_by_date.get(date, []):
            code = t["code"]
            if t["direction"] == "buy":
                amount = t.get("amount", t["price"] * t["shares"])
                fee = t.get("commission", 0)
                cash -= (amount + fee)

                if code in holdings:
                    old = holdings[code]
                    total_shares = old["shares"] + t["shares"]
                    total_cost = old["shares"] * old["avg_cost"] + t["shares"] * t["price"]
                    holdings[code] = {
                        "shares": total_shares,
                        "avg_cost": total_cost / total_shares if total_shares > 0 else 0,
                        "name": t.get("name", code),
                    }
                else:
                    holdings[code] = {
                        "shares": t["shares"],
                        "avg_cost": t["price"],
                        "name": t.get("name", code),
                    }

            elif t["direction"] == "sell":
                amount = t.get("amount", t["price"] * t["shares"])
                fee = t.get("commission", 0) + t.get("stamp_duty", 0)
                cash += (amount - fee)

                if code in holdings:
                    holdings[code]["shares"] -= t["shares"]
                    if holdings[code]["shares"] <= 0:
                        del holdings[code]

        # 用收盘价计算持仓市值
        stock_value = 0
        for code, h in holdings.items():
            price = price_map.get(code, {}).get(date, h.get("avg_cost", 0))
            stock_value += h["shares"] * price

        total_value = cash + stock_value

        daily_values.append({
            "date": date,
            "total_value": round(total_value, 2),
            "stock_value": round(stock_value, 2),
            "cash": round(cash, 2),
            "position_count": len(holdings),
            "stock_ratio": round(stock_value / total_value * 100, 2) if total_value > 0 else 0,
        })

    return daily_values


def _calculate_returns(
    daily_values: list,
    initial_capital: float,
    risk_free_rate: float,
) -> dict:
    """计算收益指标。"""
    if len(daily_values) < 2:
        return {"total_return_pct": 0, "annualized_return_pct": 0}

    final_value = daily_values[-1]["total_value"]
    total_return = final_value - initial_capital
    total_return_pct = total_return / initial_capital * 100

    trading_days = len(daily_values)
    years = trading_days / 252
    annualized_return_pct = 0
    if years > 0 and final_value > 0:
        annualized_return_pct = ((final_value / initial_capital) ** (1 / years) - 1) * 100

    # 日收益率序列
    values = [d["total_value"] for d in daily_values]
    daily_returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            daily_returns.append((values[i] / values[i - 1]) - 1)

    return {
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return_pct, 2),
        "trading_days": trading_days,
        "daily_returns": daily_returns,
    }


def _calculate_risk_metrics(daily_values: list, risk_free_rate: float) -> dict:
    """计算风险指标：最大回撤、波动率、夏普比率等。"""
    if len(daily_values) < 5:
        return {}

    values = [d["total_value"] for d in daily_values]
    daily_returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            daily_returns.append((values[i] / values[i - 1]) - 1)

    if not daily_returns:
        return {}

    # 最大回撤
    peak = values[0]
    max_drawdown = 0
    max_dd_start = max_dd_end = daily_values[0]["date"]
    dd_start = daily_values[0]["date"]

    for i, v in enumerate(values):
        if v > peak:
            peak = v
            dd_start = daily_values[i]["date"]
        dd = (v - peak) / peak
        if dd < max_drawdown:
            max_drawdown = dd
            max_dd_start = dd_start
            max_dd_end = daily_values[i]["date"]

    # 年化波动率
    if len(daily_returns) >= 2:
        avg_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(252)
    else:
        daily_vol = annual_vol = 0

    # 夏普比率
    rf_daily = risk_free_rate / 252
    sharpe = 0
    if daily_vol > 0:
        excess_return = sum(daily_returns) / len(daily_returns) - rf_daily
        sharpe = excess_return / daily_vol * math.sqrt(252)

    # Sortino 比率（只考虑下行风险）
    downside_returns = [r for r in daily_returns if r < rf_daily]
    sortino = 0
    if downside_returns:
        downside_var = sum((r - rf_daily) ** 2 for r in downside_returns) / len(downside_returns)
        downside_vol = math.sqrt(downside_var)
        if downside_vol > 0:
            excess_return = sum(daily_returns) / len(daily_returns) - rf_daily
            sortino = excess_return / downside_vol * math.sqrt(252)

    # 胜率（按日）
    winning_days = sum(1 for r in daily_returns if r > 0)
    daily_win_rate = winning_days / len(daily_returns) * 100 if daily_returns else 0

    return {
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "max_drawdown_period": f"{max_dd_start} ~ {max_dd_end}",
        "annual_volatility_pct": round(annual_vol * 100, 2),
        "daily_volatility_pct": round(daily_vol * 100, 4),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "daily_win_rate_pct": round(daily_win_rate, 1),
    }


def _calculate_benchmark(daily_values: list, benchmark_df: pd.DataFrame, benchmark_code: str = "000300") -> dict:
    """基准对比：计算超额收益和相对指标。"""
    if not daily_values or benchmark_df.empty:
        return {"available": False}

    start_date = daily_values[0]["date"]
    end_date = daily_values[-1]["date"]

    bm = benchmark_df[(benchmark_df["日期"] >= start_date) & (benchmark_df["日期"] <= end_date)]
    if len(bm) < 2:
        return {"available": False}

    bm_start = bm.iloc[0]["收盘"]
    bm_end = bm.iloc[-1]["收盘"]
    bm_return_pct = (bm_end / bm_start - 1) * 100

    portfolio_return_pct = (daily_values[-1]["total_value"] / daily_values[0]["total_value"] - 1) * 100
    excess_return_pct = portfolio_return_pct - bm_return_pct

    # 日度对齐计算 Beta 和 Alpha
    bm_date_map = dict(zip(bm["日期"], bm["收盘"]))
    portfolio_returns = []
    bm_returns = []
    prev_pv = None
    prev_bm = None

    for d in daily_values:
        date = d["date"]
        bm_price = bm_date_map.get(date)
        if bm_price and prev_pv is not None and prev_bm is not None:
            portfolio_returns.append(d["total_value"] / prev_pv - 1)
            bm_returns.append(bm_price / prev_bm - 1)
        prev_pv = d["total_value"]
        if bm_price:
            prev_bm = bm_price

    beta = 0
    alpha_annual = 0
    if len(portfolio_returns) >= 10:
        n = len(portfolio_returns)
        avg_p = sum(portfolio_returns) / n
        avg_b = sum(bm_returns) / n
        cov = sum((portfolio_returns[i] - avg_p) * (bm_returns[i] - avg_b) for i in range(n)) / (n - 1)
        var_b = sum((bm_returns[i] - avg_b) ** 2 for i in range(n)) / (n - 1)
        if var_b > 0:
            beta = cov / var_b
            alpha_daily = avg_p - beta * avg_b
            alpha_annual = alpha_daily * 252 * 100

    return {
        "available": True,
        "benchmark_code": benchmark_code,
        "benchmark_return_pct": round(bm_return_pct, 2),
        "portfolio_return_pct": round(portfolio_return_pct, 2),
        "excess_return_pct": round(excess_return_pct, 2),
        "beta": round(beta, 3),
        "alpha_annual_pct": round(alpha_annual, 2),
    }


def _calculate_trade_stats(trade_history: list, closed_positions: list) -> dict:
    """交易统计：笔数、胜率、盈亏比、费用等。"""
    buys = [t for t in trade_history if t["direction"] == "buy"]
    sells = [t for t in trade_history if t["direction"] == "sell"]

    total_buy_amount = sum(t.get("amount", 0) for t in buys)
    total_sell_amount = sum(t.get("amount", 0) for t in sells)
    total_commission = sum(t.get("commission", 0) for t in trade_history)
    total_stamp_duty = sum(t.get("stamp_duty", 0) for t in sells)
    total_fees = total_commission + total_stamp_duty

    # 基于卖出交易的胜率
    wins = [t for t in sells if t.get("pnl_amount", 0) > 0]
    losses = [t for t in sells if t.get("pnl_amount", 0) < 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0

    avg_win = sum(t.get("pnl_amount", 0) for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t.get("pnl_amount", 0) for t in losses) / len(losses)) if losses else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # 基于 closed_positions 的持仓天数分析
    hold_days = []
    for cp in closed_positions:
        bd = cp.get("buy_date", "")
        sd = cp.get("sell_date", "")
        if bd and sd:
            try:
                days = (datetime.strptime(sd, "%Y-%m-%d") - datetime.strptime(bd, "%Y-%m-%d")).days
                hold_days.append(days)
            except ValueError:
                pass

    return {
        "total_trades": len(trade_history),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "total_buy_amount": round(total_buy_amount, 2),
        "total_sell_amount": round(total_sell_amount, 2),
        "total_fees": round(total_fees, 2),
        "total_commission": round(total_commission, 2),
        "total_stamp_duty": round(total_stamp_duty, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "N/A",
        "closed_count": len(closed_positions),
        "avg_hold_days": round(sum(hold_days) / len(hold_days), 1) if hold_days else "N/A",
        "max_hold_days": max(hold_days) if hold_days else "N/A",
        "min_hold_days": min(hold_days) if hold_days else "N/A",
    }


def _analyze_positions(
    current_positions: dict,
    closed_positions: list,
    trade_history: list,
) -> dict:
    """持仓分析：个股贡献和行业分布。"""
    # 当前持仓按市值排序
    current = []
    for code, pos in current_positions.items():
        price = pos.get("current_price", pos.get("avg_cost", 0))
        shares = pos.get("shares", 0)
        mv = shares * price
        cost = shares * pos.get("avg_cost", 0)
        pnl = mv - cost
        pnl_pct = (price / pos["avg_cost"] - 1) * 100 if pos.get("avg_cost", 0) > 0 else 0
        current.append({
            "code": code,
            "name": pos.get("name", code),
            "shares": shares,
            "avg_cost": pos.get("avg_cost", 0),
            "current_price": price,
            "market_value": round(mv, 2),
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
        })
    current.sort(key=lambda x: x["market_value"], reverse=True)

    # 已平仓按盈亏排序
    closed_sorted = sorted(closed_positions, key=lambda x: float(x.get("realized_pnl", 0)), reverse=True)

    # 个股累计贡献（已实现 + 未实现）
    code_pnl: Dict[str, float] = defaultdict(float)
    code_names: Dict[str, str] = {}
    for cp in closed_positions:
        code_pnl[cp["code"]] += cp.get("realized_pnl", 0)
        code_names[cp["code"]] = cp.get("name", cp["code"])
    for c in current:
        code_pnl[c["code"]] += c["unrealized_pnl"]
        code_names[c["code"]] = c["name"]

    contributors = sorted(
        [{"code": k, "name": code_names.get(k, k), "total_pnl": round(v, 2)} for k, v in code_pnl.items()],
        key=lambda x: x["total_pnl"], reverse=True,
    )

    return {
        "current_holdings": current,
        "closed_best": closed_sorted[:5] if closed_sorted else [],
        "closed_worst": closed_sorted[-5:] if len(closed_sorted) > 5 else [],
        "top_contributors": contributors[:5],
        "bottom_contributors": contributors[-5:] if len(contributors) > 5 else [],
    }


def format_backtest_report(result: dict) -> str:
    """将回测结果格式化为 Markdown 报告。"""
    if "error" in result:
        return f"# 回测报告\n\n{result['error']}"

    period = result["period"]
    returns = result["returns"]
    risk = result["risk"]
    benchmark = result["benchmark"]
    trades = result["trades"]
    positions = result["positions"]

    lines = [
        "# 投资组合回测分析报告",
        f"**回测区间**: {period['start']} ~ {period['end']} ({period['trading_days']}个交易日)",
        f"**初始资金**: {result['initial_capital']:,.0f}元",
        f"**当前净值**: {result['current_value']:,.0f}元",
        "",
        "---",
        "",
        "## 一、收益概况",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 累计收益 | {returns.get('total_return', 0):+,.0f}元 ({returns.get('total_return_pct', 0):+.2f}%) |",
        f"| 年化收益率 | {returns.get('annualized_return_pct', 0):+.2f}% |",
    ]

    if benchmark.get("available"):
        lines.extend([
            f"| 基准收益(沪深300) | {benchmark['benchmark_return_pct']:+.2f}% |",
            f"| 超额收益 | {benchmark['excess_return_pct']:+.2f}% |",
            f"| Beta | {benchmark['beta']:.3f} |",
            f"| 年化Alpha | {benchmark['alpha_annual_pct']:+.2f}% |",
        ])

    lines.extend([
        "",
        "## 二、风险指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
    ])

    if risk:
        lines.extend([
            f"| 最大回撤 | {risk.get('max_drawdown_pct', 0):.2f}% |",
            f"| 最大回撤区间 | {risk.get('max_drawdown_period', 'N/A')} |",
            f"| 年化波动率 | {risk.get('annual_volatility_pct', 0):.2f}% |",
            f"| 夏普比率 | {risk.get('sharpe_ratio', 0):.3f} |",
            f"| Sortino比率 | {risk.get('sortino_ratio', 0):.3f} |",
            f"| 日胜率 | {risk.get('daily_win_rate_pct', 0):.1f}% |",
        ])
    else:
        lines.append("| 数据不足 | - |")

    lines.extend([
        "",
        "## 三、交易统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 总交易笔数 | {trades['total_trades']} (买{trades['buy_count']}笔, 卖{trades['sell_count']}笔) |",
        f"| 总买入金额 | {trades['total_buy_amount']:,.0f}元 |",
        f"| 总卖出金额 | {trades['total_sell_amount']:,.0f}元 |",
        f"| 累计交易费用 | {trades['total_fees']:,.0f}元 (佣金{trades['total_commission']:,.0f} + 印花税{trades['total_stamp_duty']:,.0f}) |",
        f"| 卖出胜率 | {trades['win_rate_pct']:.1f}% |",
        f"| 平均盈利 | {trades['avg_win']:+,.0f}元 |",
        f"| 平均亏损 | -{trades['avg_loss']:,.0f}元 |",
        f"| 盈亏比 | {trades['profit_factor']} |",
        f"| 已平仓笔数 | {trades['closed_count']} |",
        f"| 平均持仓天数 | {trades['avg_hold_days']} |",
    ])

    # 当前持仓
    current = positions.get("current_holdings", [])
    if current:
        lines.extend([
            "",
            "## 四、当前持仓",
            "",
            "| 股票 | 代码 | 持仓 | 成本 | 现价 | 市值 | 浮动盈亏 |",
            "|------|------|------|------|------|------|----------|",
        ])
        for h in current:
            lines.append(
                f"| {h['name']} | {h['code']} | {h['shares']}股 | "
                f"{h['avg_cost']:.2f} | {h['current_price']:.2f} | "
                f"{h['market_value']:,.0f}元 | "
                f"{h['unrealized_pnl']:+,.0f}元 ({h['unrealized_pnl_pct']:+.1f}%) |"
            )

    # 个股贡献排行
    top = positions.get("top_contributors", [])
    bottom = positions.get("bottom_contributors", [])
    if top:
        lines.extend([
            "",
            "## 五、个股盈亏贡献",
            "",
            "### 盈利前5",
            "| 股票 | 代码 | 累计贡献 |",
            "|------|------|----------|",
        ])
        for c in top:
            lines.append(f"| {c['name']} | {c['code']} | {c['total_pnl']:+,.0f}元 |")

    if bottom and len(positions.get("top_contributors", [])) > 5:
        lines.extend([
            "",
            "### 亏损前5",
            "| 股票 | 代码 | 累计贡献 |",
            "|------|------|----------|",
        ])
        for c in bottom:
            lines.append(f"| {c['name']} | {c['code']} | {c['total_pnl']:+,.0f}元 |")

    # 净值曲线摘要
    daily_values = result.get("daily_values", [])
    if len(daily_values) >= 5:
        lines.extend([
            "",
            "## 六、净值走势（部分）",
            "",
            "| 日期 | 总资产 | 股票市值 | 现金 | 仓位 |",
            "|------|--------|----------|------|------|",
        ])
        # 首5天 + 尾5天
        show = daily_values[:5]
        if len(daily_values) > 10:
            show.append({"date": "...", "total_value": 0, "stock_value": 0, "cash": 0, "stock_ratio": 0})
        show.extend(daily_values[-5:])
        for d in show:
            if d["date"] == "...":
                lines.append("| ... | ... | ... | ... | ... |")
            else:
                lines.append(
                    f"| {d['date']} | {d['total_value']:,.0f} | "
                    f"{d['stock_value']:,.0f} | {d['cash']:,.0f} | {d['stock_ratio']:.1f}% |"
                )

    lines.extend([
        "",
        "---",
        f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    return "\n".join(lines)


# ====================================================================
#  Walk-Forward 回测框架
# ====================================================================

def walk_forward_backtest(
    stock_universe: list[str],
    strategy_fn,
    initial_capital: float = 1_000_000,
    train_days: int = 120,
    test_days: int = 20,
    total_days: int = 504,
    benchmark_code: str = "000300",
    risk_free_rate: float = 0.02,
) -> dict:
    """Walk-Forward 回测：滚动训练+测试，避免过拟合。

    流程（每轮）：
    1. 用过去 train_days 天数据训练策略（如因子打分、协方差估计）
    2. 得到目标组合权重
    3. 在接下来 test_days 天按该权重持有，记录收益
    4. 滚动到下一轮

    Args:
        stock_universe: 可选股票池
        strategy_fn: 策略函数，签名 (codes, returns_df) -> dict[str, float]（代码→权重）
        initial_capital: 初始资金
        train_days: 训练窗口长度
        test_days: 测试窗口长度
        total_days: 总回测天数
        benchmark_code: 基准指数代码
        risk_free_rate: 无风险利率（年化）

    Returns:
        包含每轮结果和汇总指标的字典
    """
    from src.tools.data_provider import fetch_stock_history, fetch_index_history

    logger.info(f"[WF回测] 开始: {len(stock_universe)}只股票, "
                f"训练{train_days}天/测试{test_days}天, 总{total_days}天")

    all_prices = {}
    for code in stock_universe:
        try:
            df = fetch_stock_history(code, days=total_days + train_days + 50)
            if not df.empty and len(df) > train_days + test_days:
                all_prices[code] = df.set_index("日期")["收盘"]
        except Exception:
            continue

    if len(all_prices) < 3:
        return {"error": "可用股票数据不足"}

    price_df = pd.DataFrame(all_prices).dropna()
    if len(price_df) < train_days + test_days * 2:
        return {"error": f"价格数据不足: {len(price_df)}天 < 需要{train_days + test_days * 2}天"}

    returns_df = price_df.pct_change().dropna()
    dates = returns_df.index.tolist()

    bench_prices = None
    try:
        bench_df = fetch_index_history(benchmark_code, days=total_days + train_days + 50)
        if not bench_df.empty:
            bench_prices = bench_df.set_index("日期")["收盘"]
    except Exception:
        pass

    rounds = []
    portfolio_values = [initial_capital]
    current_value = initial_capital

    start_idx = train_days
    while start_idx + test_days <= len(dates):
        train_start = max(0, start_idx - train_days)
        train_end = start_idx
        test_start = start_idx
        test_end = min(start_idx + test_days, len(dates))

        train_returns = returns_df.iloc[train_start:train_end]

        try:
            weights = strategy_fn(list(train_returns.columns), train_returns)
        except Exception as e:
            logger.warning(f"[WF回测] 策略函数在第{len(rounds)+1}轮失败: {e}")
            weights = {code: 1.0 / len(train_returns.columns) for code in train_returns.columns}

        w_arr = np.array([weights.get(c, 0) for c in returns_df.columns])
        w_sum = w_arr.sum()
        if w_sum > 0:
            w_arr = w_arr / w_sum

        test_returns = returns_df.iloc[test_start:test_end]
        port_daily = (test_returns.values * w_arr).sum(axis=1)

        round_return = float(np.prod(1 + port_daily) - 1)
        current_value *= (1 + round_return)

        for daily_r in port_daily:
            portfolio_values.append(portfolio_values[-1] * (1 + daily_r))

        bench_return = 0
        if bench_prices is not None:
            test_date_start = dates[test_start]
            test_date_end = dates[test_end - 1]
            if test_date_start in bench_prices.index and test_date_end in bench_prices.index:
                bench_return = float(bench_prices[test_date_end] / bench_prices[test_date_start] - 1)

        top_stocks = sorted(weights.items(), key=lambda x: -x[1])[:5]
        rounds.append({
            "round": len(rounds) + 1,
            "train_period": f"{dates[train_start]}~{dates[train_end - 1]}",
            "test_period": f"{dates[test_start]}~{dates[test_end - 1]}",
            "portfolio_return": round(round_return * 100, 3),
            "benchmark_return": round(bench_return * 100, 3),
            "excess_return": round((round_return - bench_return) * 100, 3),
            "top_holdings": [(c, round(w * 100, 1)) for c, w in top_stocks],
        })

        start_idx += test_days

    if not rounds:
        return {"error": "回测轮次为零"}

    all_returns = [r["portfolio_return"] / 100 for r in rounds]
    total_return = (current_value / initial_capital - 1)
    n_rounds = len(rounds)
    years = n_rounds * test_days / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    pv = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    max_dd = float(drawdown.min())

    daily_rets = np.diff(pv) / pv[:-1]
    annual_vol = float(np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 1 else 0
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

    win_rounds = sum(1 for r in rounds if r["excess_return"] > 0)
    win_rate = win_rounds / n_rounds * 100

    result = {
        "summary": {
            "total_rounds": n_rounds,
            "total_return_pct": round(total_return * 100, 2),
            "annual_return_pct": round(annual_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "annual_volatility_pct": round(annual_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate_vs_benchmark_pct": round(win_rate, 1),
            "avg_round_return_pct": round(np.mean(all_returns) * 100, 3),
            "avg_round_excess_pct": round(np.mean([r["excess_return"] for r in rounds]), 3),
        },
        "rounds": rounds,
        "portfolio_values": [round(v, 2) for v in portfolio_values],
    }

    logger.info(
        f"[WF回测] 完成: {n_rounds}轮, "
        f"累计{total_return * 100:+.2f}%, 年化{annual_return * 100:+.2f}%, "
        f"夏普{sharpe:.3f}, 最大回撤{max_dd * 100:.2f}%"
    )
    return result


def format_walk_forward_report(result: dict) -> str:
    """将 walk-forward 回测结果格式化为 Markdown 报告。"""
    if "error" in result:
        return f"# Walk-Forward 回测报告\n\n{result['error']}"

    s = result["summary"]
    rounds = result["rounds"]

    lines = [
        "# Walk-Forward 回测报告",
        "",
        "## 一、总体表现",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 回测轮次 | {s['total_rounds']} |",
        f"| 累计收益率 | {s['total_return_pct']:+.2f}% |",
        f"| 年化收益率 | {s['annual_return_pct']:+.2f}% |",
        f"| 最大回撤 | {s['max_drawdown_pct']:.2f}% |",
        f"| 年化波动率 | {s['annual_volatility_pct']:.2f}% |",
        f"| 夏普比率 | {s['sharpe_ratio']:.3f} |",
        f"| 跑赢基准胜率 | {s['win_rate_vs_benchmark_pct']:.1f}% |",
        f"| 平均每轮超额收益 | {s['avg_round_excess_pct']:+.3f}% |",
        "",
        "## 二、分轮明细",
        "",
        "| 轮次 | 测试区间 | 组合收益 | 基准收益 | 超额 | 主要持仓 |",
        "|------|----------|----------|----------|------|----------|",
    ]

    for r in rounds:
        top = ", ".join(f"{c}({w}%)" for c, w in r["top_holdings"][:3])
        lines.append(
            f"| {r['round']} | {r['test_period']} | "
            f"{r['portfolio_return']:+.2f}% | {r['benchmark_return']:+.2f}% | "
            f"{r['excess_return']:+.2f}% | {top} |"
        )

    lines.extend([
        "",
        "---",
        f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    return "\n".join(lines)
