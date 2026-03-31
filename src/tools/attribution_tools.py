"""收益归因分析工具

提供 Brinson-Hood-Beebower (BHB) 归因模型、因子暴露分析和信号衰减跟踪。
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.tools.data_provider import fetch_index_history, fetch_stock_history

logger = logging.getLogger("portfolio")


def _get_benchmark_sector_weights(benchmark_code: str) -> dict[str, float]:
    """获取基准指数的行业权重估算（从腾讯全市场数据计算）。"""
    try:
        from src.tools.data_provider import fetch_all_stocks
        from src.tools.analysis_tools import _classify_sector

        df = fetch_all_stocks()
        if df.empty:
            return {}

        sector_caps: dict[str, float] = defaultdict(float)
        total_cap = 0.0
        for _, row in df.iterrows():
            sector = _classify_sector(row.get("名称", ""))
            cap = row.get("总市值", 0) or 1
            sector_caps[sector] += cap
            total_cap += cap

        if total_cap <= 0:
            return {}

        return {sector: cap / total_cap for sector, cap in sector_caps.items()}
    except Exception as e:
        logger.debug(f"[归因] 获取基准行业权重失败: {e}")
        return {}


def _get_benchmark_sector_returns(
    benchmark_code: str,
    sectors: set[str],
    period_days: int,
) -> dict[str, float]:
    """估算基准各行业的收益率（从腾讯全市场数据计算）。"""
    try:
        from src.tools.data_provider import fetch_all_stocks
        from src.tools.analysis_tools import _classify_sector

        df = fetch_all_stocks()
        if df.empty:
            return {}

        sector_returns: dict[str, list[float]] = defaultdict(list)
        for _, row in df.iterrows():
            sector = _classify_sector(row.get("名称", ""))
            change = row.get("涨跌幅", 0)
            if sector in sectors and change is not None:
                sector_returns[sector].append(change)

        return {
            sector: np.mean(rets) / 100 if rets else 0
            for sector, rets in sector_returns.items()
        }
    except Exception:
        return {}


@tool
def calculate_brinson_attribution(
    portfolio_holdings: dict[str, dict],
    benchmark_code: str = "000300",
    period_days: int = 20,
) -> str:
    """Brinson-Hood-Beebower (BHB) 收益归因分析。

    将组合相对基准的超额收益分解为三个来源：
    1. 资产配置效应：超配表现好的行业 / 低配表现差的行业
    2. 个股选择效应：在同一行业内选到跑赢行业均值的个股
    3. 交互效应：配置与选股的协同贡献

    Args:
        portfolio_holdings: 组合持仓 {代码: {weight: 权重(0-1), sector: 行业名}}
        benchmark_code: 基准指数代码
        period_days: 归因区间天数
    """
    if not portfolio_holdings:
        return "无持仓，无法进行归因分析"

    stock_returns = {}
    for code in portfolio_holdings:
        try:
            df = fetch_stock_history(code, days=period_days + 10)
            if not df.empty and len(df) > period_days:
                start = df["收盘"].iloc[-period_days - 1]
                end = df["收盘"].iloc[-1]
                stock_returns[code] = (end / start - 1) if start > 0 else 0
        except Exception:
            stock_returns[code] = 0

    bench_return = 0.0
    try:
        bench_df = fetch_index_history(benchmark_code, days=period_days + 10)
        if not bench_df.empty and len(bench_df) > period_days:
            b_start = bench_df["收盘"].iloc[-period_days - 1]
            b_end = bench_df["收盘"].iloc[-1]
            bench_return = (b_end / b_start - 1) if b_start > 0 else 0
    except Exception:
        pass

    sector_portfolio: dict[str, dict] = defaultdict(lambda: {"weight": 0, "weighted_return": 0})
    for code, info in portfolio_holdings.items():
        sector = info.get("sector", "其他") or "其他"
        w = info.get("weight", 0)
        r = stock_returns.get(code, 0)
        sector_portfolio[sector]["weight"] += w
        sector_portfolio[sector]["weighted_return"] += w * r

    all_sectors = set(sector_portfolio.keys())

    bench_weights = _get_benchmark_sector_weights(benchmark_code)
    bench_returns = _get_benchmark_sector_returns(benchmark_code, all_sectors, period_days)
    using_real_benchmark = bool(bench_weights)

    if not bench_weights:
        n_sectors = max(len(all_sectors), 1)
        bench_weights = {s: 1.0 / n_sectors for s in all_sectors}

    total_allocation = 0.0
    total_selection = 0.0
    total_interaction = 0.0
    sector_details = []

    portfolio_return = sum(
        info.get("weight", 0) * stock_returns.get(code, 0)
        for code, info in portfolio_holdings.items()
    )

    for sector in sorted(all_sectors):
        sp = sector_portfolio[sector]
        wp = sp["weight"]
        rp = sp["weighted_return"] / wp if wp > 0 else 0

        wb = bench_weights.get(sector, 0)
        rb = bench_returns.get(sector, bench_return)

        allocation = (wp - wb) * rb
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)

        total_allocation += allocation
        total_selection += selection
        total_interaction += interaction

        sector_details.append({
            "行业": sector,
            "组合权重": f"{wp * 100:.1f}%",
            "基准权重": f"{wb * 100:.1f}%",
            "组合收益": f"{rp * 100:.2f}%",
            "基准收益": f"{rb * 100:.2f}%",
            "配置效应": f"{allocation * 100:.3f}%",
            "选股效应": f"{selection * 100:.3f}%",
            "交互效应": f"{interaction * 100:.3f}%",
        })

    excess_return = portfolio_return - bench_return
    explained = total_allocation + total_selection + total_interaction

    detail_df = pd.DataFrame(sector_details)
    detail_str = detail_df.to_string(index=False) if not detail_df.empty else "无数据"

    alloc_pct = abs(total_allocation) / abs(explained) * 100 if abs(explained) > 1e-8 else 0
    select_pct = abs(total_selection) / abs(explained) * 100 if abs(explained) > 1e-8 else 0

    benchmark_note = "使用全市场行业市值估算" if using_real_benchmark else "⚠ 使用等权近似（无法获取市值数据）"

    return (
        f"Brinson BHB 收益归因分析 ({period_days}日):\n\n"
        f"  组合收益: {portfolio_return * 100:+.3f}%\n"
        f"  基准收益({benchmark_code}): {bench_return * 100:+.3f}%\n"
        f"  超额收益: {excess_return * 100:+.3f}%\n\n"
        f"  归因分解:\n"
        f"    资产配置效应: {total_allocation * 100:+.3f}% (占比{alloc_pct:.0f}%)\n"
        f"    个股选择效应: {total_selection * 100:+.3f}% (占比{select_pct:.0f}%)\n"
        f"    交互效应:     {total_interaction * 100:+.3f}%\n"
        f"    合计:         {explained * 100:+.3f}%\n\n"
        f"  行业归因明细:\n{detail_str}\n\n"
        f"  解读: {'选股贡献更大' if abs(total_selection) > abs(total_allocation) else '行业配置贡献更大'}"
        f"{'，交互效应显著' if abs(total_interaction) > 0.001 else ''}\n"
        f"  基准行业权重: {benchmark_note}"
    )


@tool
def analyze_factor_exposure(
    stock_codes: list[str],
    weights: list[float],
    benchmark_code: str = "000300",
) -> str:
    """分析组合的因子暴露：价值、动量、波动率、规模、流动性。

    帮助理解组合收益来源和风险敞口，判断当前因子暴露是否与市场环境匹配。

    Args:
        stock_codes: 股票代码列表
        weights: 对应权重列表
        benchmark_code: 基准指数代码
    """
    try:
        from src.tools.data_provider import fetch_all_stocks, fetch_stock_quote

        quotes = fetch_stock_quote(stock_codes[:20])
        if not quotes:
            return "获取行情数据失败"

        factors = {"value": [], "momentum": [], "volatility": [], "size": [], "liquidity": []}
        w_arr = np.array(weights[:len(stock_codes)])
        if len(w_arr) < len(stock_codes):
            w_arr = np.pad(w_arr, (0, len(stock_codes) - len(w_arr)))
        w_sum = w_arr.sum()
        if w_sum > 0:
            w_arr = w_arr / w_sum

        for i, code in enumerate(stock_codes[:20]):
            q = quotes.get(code, {})
            w = w_arr[i] if i < len(w_arr) else 0

            pe = q.get("pe", 0)
            pb = q.get("pb", 0)
            change_pct = q.get("change_pct", 0)
            amount = q.get("amount", 0)
            total_val = q.get("total_value", 0)
            turnover = q.get("turnover_rate", 0)

            if pe > 0:
                factors["value"].append(w * (1.0 / pe))
            if change_pct:
                factors["momentum"].append(w * change_pct)
            if amount:
                factors["liquidity"].append(w * amount)
            if total_val:
                factors["size"].append(w * np.log(max(total_val, 1)))

            try:
                df = fetch_stock_history(code, days=30)
                if not df.empty and len(df) > 5:
                    vol = df["收盘"].pct_change().dropna().std() * np.sqrt(252)
                    factors["volatility"].append(w * vol)
            except Exception:
                pass

        value_exp = sum(factors["value"])
        mom_exp = sum(factors["momentum"])
        vol_exp = sum(factors["volatility"])
        size_exp = sum(factors["size"])
        liq_exp = sum(factors["liquidity"])

        val_label = "偏价值" if value_exp > 0.05 else ("偏成长" if value_exp < 0.02 else "均衡")
        mom_label = "正动量" if mom_exp > 0 else "负动量"
        vol_label = "高波动" if vol_exp > 0.35 else ("低波动" if vol_exp < 0.20 else "中等波动")

        return (
            f"组合因子暴露分析:\n\n"
            f"  价值因子(1/PE加权): {value_exp:.4f} → {val_label}\n"
            f"  动量因子(涨跌幅加权): {mom_exp:+.2f}% → {mom_label}\n"
            f"  波动率因子(年化波动率加权): {vol_exp * 100:.1f}% → {vol_label}\n"
            f"  规模因子(ln市值加权): {size_exp:.2f}\n"
            f"  流动性因子(成交额加权): {liq_exp / 1e8:.1f}亿\n\n"
            f"  风格倾向: {val_label} + {vol_label}\n"
            f"  建议: "
            + ("高波动+负动量组合，建议降低仓位或增加防御性配置" if vol_exp > 0.35 and mom_exp < 0
               else "因子暴露较为均衡" if val_label == "均衡" and vol_label == "中等波动"
               else f"当前{val_label}+{vol_label}，注意与市场regime匹配")
        )
    except Exception as e:
        return f"因子暴露分析失败: {e}"


@tool
def track_signal_decay(
    stock_codes: list[str],
    lookback_periods: list[int] = [5, 10, 20, 60],
) -> str:
    """跟踪选股信号衰减：检查不同回看期的动量因子IC，判断信号是否仍然有效。

    IC (Information Coefficient) 衡量因子得分与未来收益的相关性。
    IC 持续下降意味着因子失效，需要切换策略。

    Args:
        stock_codes: 股票代码列表（用于计算因子）
        lookback_periods: 回看周期列表（天数）
    """
    try:
        from src.tools.data_provider import fetch_stock_quote

        quotes = fetch_stock_quote(stock_codes[:30])
        if not quotes or len(quotes) < 5:
            return "数据不足（需要至少5只有行情数据的股票）"

        results = []
        for period in lookback_periods:
            past_returns = []
            future_returns = []

            for code in stock_codes[:30]:
                try:
                    df = fetch_stock_history(code, days=period + 30)
                    if df.empty or len(df) < period + 5:
                        continue
                    past_r = (df["收盘"].iloc[-period - 1] / df["收盘"].iloc[-period - 6] - 1) if df["收盘"].iloc[-period - 6] > 0 else 0
                    future_r = (df["收盘"].iloc[-1] / df["收盘"].iloc[-period - 1] - 1) if df["收盘"].iloc[-period - 1] > 0 else 0
                    past_returns.append(past_r)
                    future_returns.append(future_r)
                except Exception:
                    continue

            if len(past_returns) >= 5:
                ic = float(np.corrcoef(past_returns, future_returns)[0, 1])
                ic_label = "强" if abs(ic) > 0.3 else ("中等" if abs(ic) > 0.1 else "弱/失效")
                results.append(f"  {period}日动量IC: {ic:+.3f} ({ic_label})")
            else:
                results.append(f"  {period}日动量IC: 数据不足")

        avg_ic = np.mean([float(r.split(":")[1].split("(")[0]) for r in results if "数据不足" not in r]) if results else 0

        decay_warning = ""
        if avg_ic < 0.05:
            decay_warning = "\n\n⚠ 动量信号整体较弱，建议减少动量因子权重或切换为均值回复策略"
        elif avg_ic < 0:
            decay_warning = "\n\n⚠ 动量信号反转，短期动量策略可能产生负收益，考虑暂停动量相关交易"

        return (
            f"信号衰减跟踪 ({len(stock_codes)}只股票):\n\n"
            + "\n".join(results)
            + f"\n\n  平均IC: {avg_ic:+.3f}"
            + decay_warning
        )
    except Exception as e:
        return f"信号衰减跟踪失败: {e}"


ATTRIBUTION_TOOLS = [
    calculate_brinson_attribution,
    analyze_factor_exposure,
    track_signal_decay,
]
