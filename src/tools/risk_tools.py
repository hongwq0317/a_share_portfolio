"""风险管理工具

提供 VaR 计算（含 EWMA、Cornish-Fisher）、回撤分析、压力测试（历史场景）、
止损检测（追踪止损、ATR 止损、分批止盈）、组合熔断、涨跌停检测等风控能力。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy.stats import norm, skew, kurtosis

from src.tools.data_provider import fetch_index_history, fetch_stock_history, fetch_stock_quote

logger = logging.getLogger("portfolio")


def _validate_positive(name: str, value: float) -> str | None:
    """Return error message if value is not positive, else None."""
    if value is None or value <= 0:
        return f"参数错误: {name} 必须为正数，当前值={value}"
    return None


# ====================================================================
#  内部工具函数
# ====================================================================

def _fetch_returns(stock_codes: list[str], days: int = 252) -> Optional[pd.DataFrame]:
    """获取多只股票的日收益率矩阵"""
    price_dict = {}
    for code in stock_codes:
        try:
            df = fetch_stock_history(code, days=days + 30)
            if not df.empty and len(df) > 20:
                price_dict[code] = df.set_index("日期")["收盘"]
        except Exception:
            continue

    if not price_dict:
        return None

    prices = pd.DataFrame(price_dict).dropna()
    return prices.pct_change().dropna()


def _ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> float:
    """EWMA (RiskMetrics) 波动率估计，捕捉波动率聚集效应。

    λ=0.94 为 RiskMetrics 日频数据标准参数。近期波动对估计的权重更高，
    相比等权历史波动率，能更及时地反映市场风险变化。
    """
    n = len(returns)
    if n < 10:
        return float(returns.std())
    var = float(returns[:10].var())
    for i in range(10, n):
        var = lam * var + (1 - lam) * returns[i] ** 2
    return np.sqrt(var)


def _cornish_fisher_var(returns: np.ndarray, confidence: float) -> float:
    """Cornish-Fisher VaR，修正正态假设以考虑偏度和峰度。

    z_cf = z + (z²-1)·S/6 + (z³-3z)·K/24 - (2z³-5z)·S²/36
    适配 A 股厚尾分布特征，比参数法 VaR 更准确。
    """
    z = norm.ppf(1 - confidence)
    s = float(skew(returns))
    k = float(kurtosis(returns, fisher=True))

    z_cf = (z
            + (z ** 2 - 1) * s / 6
            + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)

    mu = float(returns.mean())
    sigma = float(returns.std())
    return -(mu + z_cf * sigma)


# ====================================================================
#  VaR 计算
# ====================================================================

@tool
def calculate_portfolio_var(
    stock_codes: list[str],
    weights: list[float],
    total_value: float,
    confidence_levels: list[float] = [0.95, 0.99],
    days: int = 252,
) -> str:
    """计算组合 VaR，支持历史模拟、参数法、EWMA、Cornish-Fisher 四种方法。

    相比传统正态假设：
    - EWMA 波动率捕捉近期波动聚集效应（RiskMetrics 方法）
    - Cornish-Fisher 修正偏度和峰度，适配 A 股厚尾分布

    Args:
        stock_codes: 股票代码列表
        weights: 对应权重列表
        total_value: 组合总市值
        confidence_levels: 置信度列表
        days: 历史数据天数
    """
    err = _validate_positive("total_value", total_value)
    if err:
        return err
    returns = _fetch_returns(stock_codes, days)
    if returns is None:
        return "数据不足，无法计算VaR"

    n_cols = len(returns.columns)
    w = np.array(weights[:n_cols])
    if len(w) < n_cols:
        w = np.pad(w, (0, n_cols - len(w)), constant_values=0)
    w_sum = w.sum()
    if w_sum <= 0:
        return "权重无效：权重总和为零，无法计算VaR"
    w = w / w_sum

    port_returns = (returns * w).sum(axis=1).values

    ret_skew = float(skew(port_returns))
    ret_kurt = float(kurtosis(port_returns, fisher=True))
    sigma = float(port_returns.std())
    ewma_vol = _ewma_volatility(port_returns)
    mu = float(port_returns.mean())

    results = []

    for conf in confidence_levels:
        hist_var = abs(float(np.percentile(port_returns, (1 - conf) * 100)))
        z_score = norm.ppf(1 - conf)
        param_var = -(mu + z_score * sigma)
        ewma_var = -(mu + z_score * ewma_vol)
        cf_var = _cornish_fisher_var(port_returns, conf)

        results.append(
            f"  {conf * 100:.0f}%置信度:\n"
            f"    历史模拟法: {hist_var * 100:.2f}% ({hist_var * total_value:,.0f}元)\n"
            f"    参数法(正态): {param_var * 100:.2f}% ({param_var * total_value:,.0f}元)\n"
            f"    EWMA动态VaR: {ewma_var * 100:.2f}% ({ewma_var * total_value:,.0f}元)\n"
            f"    Cornish-Fisher: {cf_var * 100:.2f}% ({cf_var * total_value:,.0f}元)"
        )

    for conf in confidence_levels:
        threshold = float(np.percentile(port_returns, (1 - conf) * 100))
        tail = port_returns[port_returns <= threshold]
        cvar = abs(float(tail.mean())) if len(tail) > 0 else 0
        results.append(
            f"  CVaR({conf * 100:.0f}%): {cvar * 100:.2f}% ({cvar * total_value:,.0f}元)"
        )

    skew_label = "左偏/厚左尾" if ret_skew < -0.5 else ("右偏" if ret_skew > 0.5 else "近似对称")
    kurt_label = "厚尾" if ret_kurt > 1 else ("薄尾" if ret_kurt < -0.5 else "近似正态")

    return (
        f"组合 VaR 分析 (总市值: {total_value:,.0f}元):\n"
        + "\n".join(results)
        + f"\n\n  分布特征:\n"
        f"    日均收益: {mu * 100:.3f}%\n"
        f"    日波动率(等权): {sigma * 100:.3f}%\n"
        f"    日波动率(EWMA): {ewma_vol * 100:.3f}%\n"
        f"    年化波动率: {sigma * np.sqrt(252) * 100:.2f}%\n"
        f"    偏度: {ret_skew:.3f} ({skew_label})\n"
        f"    超额峰度: {ret_kurt:.3f} ({kurt_label})"
    )


# ====================================================================
#  回撤分析
# ====================================================================

@tool
def calculate_max_drawdown(stock_code: str, days: int = 252) -> str:
    """计算股票/组合的最大回撤。

    Args:
        stock_code: 股票代码
        days: 分析天数
    """
    try:
        df = fetch_stock_history(stock_code, days=days)
        if df.empty:
            return f"{stock_code}: 无数据"

        close = df["收盘"].values
        peak = np.maximum.accumulate(close)
        drawdown = (close - peak) / peak
        max_dd = drawdown.min()
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(close[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

        current_peak = np.max(close)
        current_dd = (close[-1] - current_peak) / current_peak

        return (
            f"{stock_code} 回撤分析 ({days}天):\n"
            f"  最大回撤: {max_dd * 100:.2f}%\n"
            f"  回撤起点: {df['日期'].iloc[peak_idx]} (价格: {close[peak_idx]:.2f})\n"
            f"  回撤低点: {df['日期'].iloc[max_dd_idx]} (价格: {close[max_dd_idx]:.2f})\n"
            f"  当前回撤: {current_dd * 100:.2f}% (距历史高点)\n"
            f"  当前价格: {close[-1]:.2f}"
        )
    except Exception as e:
        return f"回撤计算失败: {e}"


# ====================================================================
#  止损/止盈检测
# ====================================================================

@tool
def check_stop_loss(
    stock_code: str,
    avg_cost: float,
    current_price: float,
    stop_loss_pct: float = -8.0,
    take_profit_pct: float = 30.0,
) -> str:
    """检测是否触发止损/止盈（基础版）。

    Args:
        stock_code: 股票代码
        avg_cost: 平均成本价
        current_price: 当前价格
        stop_loss_pct: 止损线(%)，如 -8
        take_profit_pct: 止盈线(%)，如 30
    """
    err = _validate_positive("avg_cost", avg_cost) or _validate_positive("current_price", current_price)
    if err:
        return err
    pnl_pct = (current_price / avg_cost - 1) * 100

    alerts = []
    if pnl_pct <= stop_loss_pct:
        alerts.append(f"触发止损! 当前亏损{pnl_pct:.2f}% <= 止损线{stop_loss_pct}%")
    if pnl_pct >= take_profit_pct:
        alerts.append(f"触发止盈! 当前盈利{pnl_pct:.2f}% >= 止盈线{take_profit_pct}%")

    status = "正常"
    if alerts:
        status = "止损" if pnl_pct <= stop_loss_pct else "止盈"

    return (
        f"{stock_code} 止损止盈检测:\n"
        f"  成本: {avg_cost:.2f}, 现价: {current_price:.2f}\n"
        f"  盈亏: {pnl_pct:+.2f}%\n"
        f"  止损线: {stop_loss_pct}%, 止盈线: {take_profit_pct}%\n"
        f"  状态: {status}\n"
        + ("\n".join(f"  ⚠️ {a}" for a in alerts) if alerts else "  无预警")
    )


@tool
def check_advanced_stop_loss(
    stock_code: str,
    avg_cost: float,
    current_price: float,
    buy_date: str = "",
    highest_since_buy: float = 0,
    trailing_stop_pct: float = 15.0,
    atr_multiplier: float = 2.5,
    time_stop_days: int = 45,
    time_stop_min_return: float = 0.0,
) -> str:
    """高级止损止盈检测（追踪止损 + ATR 止损 + 时间止损 + 分批止盈）。

    注意：追踪止损和ATR止损是辅助参考信号，不等同于硬止损。
    只有浮亏达到配置的硬止损线才应强制卖出。追踪止损触发仅代表"需要关注"。

    - 追踪止损：从买入后最高点回撤超过阈值触发（关注信号，非强制卖出）
    - ATR 止损：基于波动率动态调整止损位（关注信号，非强制卖出）
    - 时间止损：持仓超期且未达预期收益触发（建议观察）
    - 分批止盈：盈利达到不同阶梯时建议分批卖出（锁定部分利润）

    Args:
        stock_code: 股票代码
        avg_cost: 平均成本价
        current_price: 当前价格
        buy_date: 买入日期(YYYY-MM-DD)
        highest_since_buy: 买入后最高价（0 则自动从历史数据计算）
        trailing_stop_pct: 追踪止损回撤比例(%), 默认15%
        atr_multiplier: ATR 止损倍数, 默认2.5
        time_stop_days: 时间止损天数
        time_stop_min_return: 时间止损最低收益率(%)
    """
    err = _validate_positive("avg_cost", avg_cost) or _validate_positive("current_price", current_price)
    if err:
        return err
    pnl_pct = (current_price / avg_cost - 1) * 100
    alerts = []
    atr_val = None

    # 计算持仓天数，用于追踪止损豁免判断
    _hold_days = 999
    if buy_date:
        try:
            from datetime import datetime as _dt
            _hold_days = (_dt.now() - _dt.strptime(buy_date, "%Y-%m-%d")).days
        except ValueError:
            pass

    try:
        df = fetch_stock_history(stock_code, days=120)
        if not df.empty and len(df) > 20:
            # --- 追踪止损（持仓 >= 5天才启用，短持仓仅用硬止损）---
            if highest_since_buy <= 0:
                if buy_date:
                    recent = df[df["日期"] >= buy_date]
                    highest_since_buy = float(recent["最高"].max()) if not recent.empty else float(df["最高"].tail(30).max())
                else:
                    highest_since_buy = float(df["最高"].tail(30).max())

            trail_dd = (current_price / highest_since_buy - 1) * 100
            if _hold_days < 5:
                if trail_dd <= -trailing_stop_pct:
                    alerts.append(
                        f"ℹ️ 追踪止损豁免: 持仓仅{_hold_days}天(<5天)，"
                        f"虽回撤{trail_dd:.1f}%超阈值，但短持仓不适用追踪止损，仅用硬止损判断"
                    )
            else:
                if trail_dd <= -trailing_stop_pct:
                    alerts.append(
                        f"🟡 追踪止损关注: 从最高{highest_since_buy:.2f}回撤{trail_dd:.1f}% "
                        f"> 阈值-{trailing_stop_pct}% (持仓{_hold_days}天)。"
                        f"注意：追踪止损仅为关注信号，如持仓审查评分≥3.0则不必强制卖出"
                    )
                elif trail_dd <= -trailing_stop_pct * 0.7:
                    alerts.append(
                        f"ℹ️ 追踪止损预警: 从最高{highest_since_buy:.2f}回撤{trail_dd:.1f}%, "
                        f"接近阈值-{trailing_stop_pct}%，持续关注"
                    )

            # --- ATR 止损 ---
            high = df["最高"].values
            low = df["最低"].values
            close = df["收盘"].values
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]),
                ),
            )
            atr_val = float(pd.Series(tr).rolling(14).mean().iloc[-1])
            atr_stop_level = close[-1] - atr_multiplier * atr_val

            if current_price <= atr_stop_level:
                alerts.append(
                    f"🟡 ATR止损关注: 现价{current_price:.2f} <= "
                    f"止损位{atr_stop_level:.2f} (收盘{close[-1]:.2f} - {atr_multiplier}×ATR{atr_val:.2f})。"
                    f"注意：ATR止损仅为关注信号，需结合硬止损和持仓审查综合判断"
                )
            elif current_price <= atr_stop_level * 1.02:
                alerts.append(
                    f"ℹ️ ATR止损预警: 现价{current_price:.2f}接近止损位{atr_stop_level:.2f}"
                )
    except Exception as e:
        alerts.append(f"⚠ 追踪/ATR止损计算异常: {e}")

    # --- 时间止损 ---
    hold_days_val = None
    if buy_date:
        try:
            from datetime import datetime
            hold_days_val = (datetime.now() - datetime.strptime(buy_date, "%Y-%m-%d")).days
            if hold_days_val > time_stop_days and pnl_pct < time_stop_min_return:
                alerts.append(
                    f"🟡 时间止损触发: 持仓{hold_days_val}天 > {time_stop_days}天, "
                    f"收益{pnl_pct:.1f}% < 目标{time_stop_min_return}%"
                )
        except ValueError:
            pass

    # --- 分批止盈 ---
    tiers = [
        {"pct": 20, "sell_ratio": 0.3, "label": "第一档(+20%)"},
        {"pct": 40, "sell_ratio": 0.5, "label": "第二档(+40%)"},
        {"pct": 60, "sell_ratio": 1.0, "label": "第三档(+60%清仓)"},
    ]
    for tier in tiers:
        if pnl_pct >= tier["pct"]:
            alerts.append(
                f"🟢 分批止盈 {tier['label']}: 盈利{pnl_pct:.1f}% >= {tier['pct']}%, "
                f"建议卖出{tier['sell_ratio'] * 100:.0f}%仓位"
            )

    # 硬止损是唯一的红色警报
    hard_stop_pct = -12.0  # 与配置文件的 stop_loss_pct 保持一致
    if pnl_pct <= hard_stop_pct:
        alerts.insert(0,
            f"🔴 硬止损触发! 浮亏{pnl_pct:.1f}%已达止损线{hard_stop_pct}%，必须卖出"
        )

    status = "正常"
    if any("🔴" in a for a in alerts):
        status = "硬止损触发（必须执行）"
    elif any("🟡" in a for a in alerts):
        status = "关注信号（非强制）"
    elif any("🟢" in a for a in alerts):
        status = "止盈信号"

    info_lines = [
        f"{stock_code} 高级止损止盈检测:",
        f"  成本: {avg_cost:.2f}, 现价: {current_price:.2f}, 盈亏: {pnl_pct:+.2f}%",
        f"  买入后最高: {highest_since_buy:.2f}",
    ]
    if atr_val is not None:
        info_lines.append(f"  14日ATR: {atr_val:.2f}")
    if hold_days_val is not None:
        info_lines.append(f"  持仓天数: {hold_days_val}")
    info_lines.append(f"  综合状态: {status}\n")

    return "\n".join(info_lines) + (
        "\n".join(f"  {a}" for a in alerts) if alerts else "  ✅ 所有止损指标正常"
    )


# ====================================================================
#  压力测试（确定性 + 历史场景）
# ====================================================================

HISTORICAL_SCENARIOS = {
    "2015_crash": {
        "name": "2015年股灾(6月-8月)",
        "description": "去杠杆引发系统性暴跌，3周内沪指从5178跌至3373",
        "index_shock": -0.45,
        "high_beta_extra": -0.15,
        "duration_days": 15,
    },
    "2018_trade_war": {
        "name": "2018年中美贸易战(Q3-Q4)",
        "description": "中美贸易摩擦升级叠加去杠杆，沪指全年跌约25%",
        "index_shock": -0.25,
        "high_beta_extra": -0.10,
        "duration_days": 60,
    },
    "2020_covid": {
        "name": "2020年新冠疫情(2月)",
        "description": "春节后首日千股跌停，1周内快速下杀后V型反弹",
        "index_shock": -0.12,
        "high_beta_extra": -0.05,
        "duration_days": 5,
    },
    "2022_lockdown": {
        "name": "2022年上海封控(Q2)",
        "description": "疫情封控+经济预期恶化，沪指跌至2863",
        "index_shock": -0.18,
        "high_beta_extra": -0.08,
        "duration_days": 30,
    },
    "market_crash": {
        "name": "假设: 极端暴跌",
        "description": "大盘连续暴跌，类似熔断行情",
        "index_shock": -0.20,
        "high_beta_extra": -0.10,
        "duration_days": 5,
    },
    "rate_hike": {
        "name": "假设: 利率大幅上升",
        "description": "央行收紧货币政策，成长股杀估值",
        "index_shock": -0.10,
        "high_beta_extra": -0.08,
        "duration_days": 20,
    },
    "liquidity_crisis": {
        "name": "假设: 流动性危机",
        "description": "千股跌停式流动性枯竭，卖不出去",
        "index_shock": -0.15,
        "high_beta_extra": -0.12,
        "duration_days": 3,
    },
}


@tool
def stress_test(
    stock_codes: list[str],
    weights: list[float],
    total_value: float,
    scenario: str = "2015_crash",
) -> str:
    """组合压力测试（确定性结果，支持 A 股历史极端事件和假设场景）。

    支持场景: 2015_crash, 2018_trade_war, 2020_covid, 2022_lockdown,
             market_crash, rate_hike, liquidity_crisis

    Args:
        stock_codes: 股票代码列表
        weights: 对应权重列表
        total_value: 组合总市值
        scenario: 压力测试场景
    """
    s = HISTORICAL_SCENARIOS.get(scenario)
    if s is None:
        available = ", ".join(HISTORICAL_SCENARIOS.keys())
        return f"未知场景'{scenario}'，可用场景: {available}"

    returns = _fetch_returns(stock_codes, 252)
    if returns is None:
        return "数据不足，无法进行压力测试"

    n_cols = len(returns.columns)
    w = np.array(weights[:n_cols])
    if len(w) < n_cols:
        w = np.pad(w, (0, n_cols - len(w)), constant_values=0)
    w_sum = w.sum()
    if w_sum <= 0:
        return "权重无效：权重总和为零，无法进行压力测试"
    w = w / w_sum

    port_returns = (returns * w).sum(axis=1)
    port_vol = float(np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w))))

    stock_vols = returns.std() * np.sqrt(252)
    avg_vol = float(stock_vols.mean())

    # 确定性压力损失：基准冲击 × Beta 调整
    base_shock = s["index_shock"]
    beta_adj = min(max(port_vol / avg_vol if avg_vol > 0 else 1.0, 0.5), 2.0)
    stressed_loss = base_shock * beta_adj + s["high_beta_extra"] * max(beta_adj - 1, 0)
    loss_amount = abs(stressed_loss) * total_value

    # Block Bootstrap 蒙特卡洛（保留序列依赖性）
    n_simulations = 10000
    port_daily = port_returns.values
    block_size = min(5, max(1, len(port_daily) // 10))
    duration = s["duration_days"]
    rng = np.random.RandomState(42)

    sim_returns = np.empty(n_simulations)
    n_blocks = max(1, duration // block_size)
    for i in range(n_simulations):
        blocks = []
        for _ in range(n_blocks):
            start = rng.randint(0, max(1, len(port_daily) - block_size))
            blocks.extend(port_daily[start:start + block_size].tolist())
        sim_returns[i] = np.prod([1 + r for r in blocks[:duration]]) - 1

    worst_1pct = float(np.percentile(sim_returns, 1)) * total_value
    worst_5pct = float(np.percentile(sim_returns, 5)) * total_value

    stock_impacts = []
    for i, code in enumerate(returns.columns):
        vol_ratio = float(stock_vols.iloc[i]) / avg_vol if avg_vol > 0 else 1
        stock_loss = base_shock * vol_ratio
        stock_loss_amt = abs(stock_loss) * w[i] * total_value
        stock_impacts.append(
            f"    {code}: 预估跌{abs(stock_loss) * 100:.1f}%, 损失{stock_loss_amt:,.0f}元(占比{w[i]*100:.1f}%)"
        )

    return (
        f"压力测试 - {s['name']}:\n"
        f"  场景描述: {s['description']}\n"
        f"  持续时间: 约{duration}个交易日\n"
        f"  组合总值: {total_value:,.0f}元\n"
        f"  基准冲击: {base_shock * 100:.0f}%\n"
        f"  组合Beta调整: {beta_adj:.2f}\n"
        f"  组合预估损失: {stressed_loss * 100:.2f}% ({loss_amount:,.0f}元)\n\n"
        f"  蒙特卡洛({duration}日, {n_simulations:,}次, Block Bootstrap):\n"
        f"    最差1%分位: {worst_1pct:,.0f}元\n"
        f"    最差5%分位: {worst_5pct:,.0f}元\n\n"
        f"  个股冲击分解:\n"
        + "\n".join(stock_impacts)
        + f"\n\n  当前年化波动率: {port_vol * 100:.2f}%"
    )


# ====================================================================
#  组合熔断检查
# ====================================================================

@tool
def check_portfolio_circuit_breaker(
    positions: dict[str, dict],
    total_value: float,
    initial_value: float,
    daily_loss_limit: float = -3.0,
    total_drawdown_limit: float = -15.0,
) -> str:
    """组合级熔断检查：日内亏损或总回撤超阈值时触发强制降仓。

    不同于个股止损，这是组合维度的系统性风控，防止组合净值崩溃。

    Args:
        positions: 持仓 {代码: {shares, avg_cost, current_price, prev_close}}
        total_value: 组合当前总市值
        initial_value: 组合初始资金或历史高点净值
        daily_loss_limit: 日内亏损熔断线(%)
        total_drawdown_limit: 总回撤熔断线(%)
    """
    daily_pnl = 0.0
    prev_total = 0.0
    for _code, pos in positions.items():
        shares = pos.get("shares", 0)
        cur = pos.get("current_price", 0)
        prev = pos.get("prev_close", pos.get("avg_cost", cur))
        daily_pnl += (cur - prev) * shares
        prev_total += prev * shares

    daily_pnl_pct = (daily_pnl / prev_total * 100) if prev_total > 0 else 0
    total_dd_pct = (total_value / initial_value - 1) * 100 if initial_value > 0 else 0

    alerts = []
    actions = []

    if daily_pnl_pct <= daily_loss_limit:
        alerts.append(
            f"🔴 日内熔断触发! 组合日内亏损{daily_pnl_pct:.2f}% <= 阈值{daily_loss_limit}%\n"
            f"    强制指令: 减仓至50%以下，优先卖出跌幅最大的持仓"
        )
        actions.append("日内熔断-强制减仓")
    elif daily_pnl_pct <= daily_loss_limit * 0.7:
        alerts.append(
            f"🟡 日内亏损预警: {daily_pnl_pct:.2f}%, 接近熔断线{daily_loss_limit}%"
        )
        actions.append("日内亏损预警")

    if total_dd_pct <= total_drawdown_limit:
        alerts.append(
            f"🔴 总回撤熔断触发! 组合总回撤{total_dd_pct:.2f}% <= 阈值{total_drawdown_limit}%\n"
            f"    强制指令: 清仓至仅保留20%仓位，暂停新建仓"
        )
        actions.append("总回撤熔断-大幅减仓")
    elif total_dd_pct <= total_drawdown_limit * 0.7:
        alerts.append(
            f"🟡 总回撤预警: {total_dd_pct:.2f}%, 接近熔断线{total_drawdown_limit}%"
        )
        actions.append("总回撤预警")

    if not actions:
        action = "正常"
    elif any("熔断" in a for a in actions):
        action = " + ".join(a for a in actions if "熔断" in a)
    else:
        action = " + ".join(actions)

    return (
        f"组合熔断检查:\n"
        f"  日内盈亏: {daily_pnl:+,.0f}元 ({daily_pnl_pct:+.2f}%)\n"
        f"  总回撤: {total_dd_pct:+.2f}% (初始{initial_value:,.0f} → 当前{total_value:,.0f})\n"
        f"  日内熔断线: {daily_loss_limit}%, 总回撤熔断线: {total_drawdown_limit}%\n"
        f"  状态: {action}\n\n"
        + ("\n".join(f"  {a}" for a in alerts) if alerts else "  ✅ 组合级风控正常")
    )


# ====================================================================
#  涨跌停检测
# ====================================================================

@tool
def detect_limit_updown(stock_codes: list[str]) -> str:
    """检测股票涨跌停状态，影响交易可执行性。

    涨停时挂买单无法成交，跌停时挂卖单无法成交。
    不同板块涨跌停幅度不同：主板±10%、创业板/科创板±20%、北交所±30%、ST±5%。

    Args:
        stock_codes: 股票代码列表
    """
    quotes = fetch_stock_quote(stock_codes[:30])
    if not quotes:
        return "行情数据获取失败"

    results = []
    limit_up_list = []
    limit_down_list = []

    for code in stock_codes[:30]:
        q = quotes.get(code)
        if not q:
            results.append(f"  {code}: 无数据")
            continue

        price = q["price"]
        change_pct = q.get("change_pct", 0)
        name = q.get("name", code)

        if code.startswith(("300", "301", "688", "689")):
            limit_pct = 20.0
        elif code.startswith(("4", "8")) and not code.startswith("88"):
            limit_pct = 30.0
        else:
            limit_pct = 10.0

        if "ST" in name.upper():
            limit_pct = 5.0

        status = "正常交易"
        if change_pct >= limit_pct - 0.1:
            status = f"涨停(+{limit_pct}%) - 无法买入"
            limit_up_list.append(f"{name}({code})")
        elif change_pct <= -(limit_pct - 0.1):
            status = f"跌停(-{limit_pct}%) - 无法卖出"
            limit_down_list.append(f"{name}({code})")
        elif change_pct >= limit_pct * 0.9:
            status = f"接近涨停({change_pct:+.1f}%) - 买入需谨慎"
        elif change_pct <= -(limit_pct * 0.9):
            status = f"接近跌停({change_pct:+.1f}%) - 卖出可能困难"

        results.append(f"  {name}({code}): {price:.2f} ({change_pct:+.2f}%) → {status}")

    summary = [f"涨跌停检测({len(stock_codes)}只):"]
    if limit_up_list:
        summary.append(f"  ⚠ 涨停(不可买入): {', '.join(limit_up_list)}")
    if limit_down_list:
        summary.append(f"  ⚠ 跌停(不可卖出): {', '.join(limit_down_list)}")
    if not limit_up_list and not limit_down_list:
        summary.append("  ✅ 无涨跌停限制")

    return "\n".join(summary) + "\n\n" + "\n".join(results)


# ====================================================================
#  仓位检查 & Beta
# ====================================================================

@tool
def check_position_limits(
    positions: dict[str, float],
    sector_map: dict[str, str],
    total_value: float,
    max_single_weight: float = 0.15,
    max_sector_weight: float = 0.35,
) -> str:
    """检查持仓是否违反仓位限制。

    Args:
        positions: 持仓 {股票代码: 市值}
        sector_map: 行业映射 {股票代码: 行业名称}
        total_value: 组合总市值
        max_single_weight: 单股最大仓位
        max_sector_weight: 单行业最大仓位
    """
    if total_value <= 0:
        return "仓位检查失败: total_value 必须为正数"
    violations = []
    warnings = []

    for code, value in positions.items():
        weight = value / total_value
        if weight > max_single_weight:
            violations.append(
                f"[超限] {code}: {weight * 100:.1f}% > {max_single_weight * 100:.0f}%"
            )
        elif weight > max_single_weight * 0.8:
            warnings.append(
                f"[预警] {code}: {weight * 100:.1f}% 接近上限{max_single_weight * 100:.0f}%"
            )

    sector_exposure = {}
    for code, value in positions.items():
        sector = sector_map.get(code, "未知")
        sector_exposure[sector] = sector_exposure.get(sector, 0) + value

    for sector, value in sector_exposure.items():
        weight = value / total_value if total_value > 0 else 0
        if weight > max_sector_weight:
            violations.append(
                f"[超限] {sector}: {weight * 100:.1f}% > {max_sector_weight * 100:.0f}%"
            )
        elif weight > max_sector_weight * 0.8:
            warnings.append(
                f"[预警] {sector}: {weight * 100:.1f}% 接近上限{max_sector_weight * 100:.0f}%"
            )

    sector_str = "\n".join(
        f"  {s}: {v / total_value * 100:.1f}%"
        for s, v in sorted(sector_exposure.items(), key=lambda x: -x[1])
    )

    result = f"仓位限制检查:\n行业暴露:\n{sector_str}\n"
    if violations:
        result += "\n违规项:\n" + "\n".join(f"  {v}" for v in violations)
    if warnings:
        result += "\n预警项:\n" + "\n".join(f"  {w}" for w in warnings)
    if not violations and not warnings:
        result += "\n所有仓位在限制范围内"

    return result


@tool
def calculate_portfolio_beta(
    stock_codes: list[str],
    weights: list[float],
    benchmark: str = "000300",
) -> str:
    """计算组合的 Beta 系数和相对基准的风险指标。

    Args:
        stock_codes: 股票代码列表
        weights: 权重列表
        benchmark: 基准指数代码
    """
    try:
        bench_df = fetch_index_history(benchmark, days=300)
        if bench_df.empty:
            return "基准指数数据获取失败"
        bench_returns = bench_df.set_index("日期")["收盘"].pct_change().dropna()

        stock_returns = _fetch_returns(stock_codes, 252)
        if stock_returns is None:
            return "股票数据获取失败"

        w = np.array(weights[:len(stock_returns.columns)])
        w = w / w.sum()
        port_returns = (stock_returns * w).sum(axis=1)

        common_idx = port_returns.index.intersection(bench_returns.index)
        if len(common_idx) < 30:
            return "共同交易日不足"
        pr = port_returns.loc[common_idx]
        br = bench_returns.loc[common_idx]

        cov_mat = np.cov(pr, br)
        beta = cov_mat[0, 1] / cov_mat[1, 1]
        alpha = (pr.mean() - beta * br.mean()) * 252

        tracking_error = (pr - br).std() * np.sqrt(252)
        info_ratio = (pr.mean() - br.mean()) * 252 / tracking_error if tracking_error > 0 else 0

        return (
            f"组合相对基准({benchmark})分析:\n"
            f"  Beta: {beta:.3f}\n"
            f"  Alpha(年化): {alpha * 100:.2f}%\n"
            f"  跟踪误差(年化): {tracking_error * 100:.2f}%\n"
            f"  信息比率: {info_ratio:.3f}\n"
            f"  组合年化收益: {pr.mean() * 252 * 100:.2f}%\n"
            f"  基准年化收益: {br.mean() * 252 * 100:.2f}%\n"
            f"  超额收益: {(pr.mean() - br.mean()) * 252 * 100:.2f}%"
        )
    except Exception as e:
        return f"Beta计算失败: {e}"


# ====================================================================
#  T+1 锁仓比例检测
# ====================================================================

@tool
def check_t1_lock_ratio(
    positions: dict[str, dict],
    total_value: float,
    max_lock_ratio: float = 0.6,
) -> str:
    """检测 T+1 锁仓比例，评估次日流动性风险。

    A股 T+1 制度下，当日买入的股票次日才能卖出。如果锁仓比例过高，
    当日遇到极端行情时无法及时止损，流动性风险显著上升。

    Args:
        positions: 持仓 {代码: {shares, today_bought_shares, current_price, name}}
        total_value: 组合总市值（含现金）
        max_lock_ratio: 锁仓比例警戒线（默认60%）
    """
    if total_value <= 0:
        return "参数错误: total_value 必须为正数"

    locked_value = 0.0
    sellable_value = 0.0
    details = []

    for code, pos in positions.items():
        shares = pos.get("shares", 0)
        today_bought = pos.get("today_bought_shares", 0)
        price = pos.get("current_price", 0)
        name = pos.get("name", code)

        if shares <= 0 or price <= 0:
            continue

        locked_shares = min(today_bought, shares)
        sellable_shares = max(shares - locked_shares, 0)

        stock_locked_val = locked_shares * price
        stock_sellable_val = sellable_shares * price
        locked_value += stock_locked_val
        sellable_value += stock_sellable_val

        if locked_shares > 0:
            lock_pct = locked_shares / shares * 100
            details.append(
                f"  {name}({code}): 总{shares}股, 锁仓{locked_shares}股({lock_pct:.0f}%), "
                f"锁仓市值{stock_locked_val:,.0f}元"
            )

    stock_total = locked_value + sellable_value
    lock_ratio = locked_value / total_value if total_value > 0 else 0
    sellable_ratio = sellable_value / total_value if total_value > 0 else 0

    alerts = []
    if lock_ratio >= max_lock_ratio:
        alerts.append(
            f"🔴 锁仓比例过高! {lock_ratio:.1%} >= 警戒线{max_lock_ratio:.0%}\n"
            f"    风险: 极端行情下仅{sellable_ratio:.1%}仓位可卖出止损\n"
            f"    建议: 下次调仓时控制单日买入量，或分批建仓"
        )
    elif lock_ratio >= max_lock_ratio * 0.7:
        alerts.append(
            f"🟡 锁仓比例偏高: {lock_ratio:.1%}，接近警戒线{max_lock_ratio:.0%}"
        )

    status = "正常"
    if lock_ratio >= max_lock_ratio:
        status = "锁仓过高-流动性风险"
    elif lock_ratio >= max_lock_ratio * 0.7:
        status = "锁仓偏高-需关注"

    result_lines = [
        "T+1 锁仓比例检测:",
        f"  组合总值: {total_value:,.0f}元",
        f"  持仓市值: {stock_total:,.0f}元",
        f"  锁仓市值: {locked_value:,.0f}元 ({lock_ratio:.1%})",
        f"  可卖市值: {sellable_value:,.0f}元 ({sellable_ratio:.1%})",
        f"  状态: {status}",
    ]

    if details:
        result_lines.append("\n  锁仓明细:")
        result_lines.extend(details)

    if alerts:
        result_lines.append("")
        result_lines.extend(f"  {a}" for a in alerts)
    else:
        result_lines.append(f"\n  ✅ 锁仓比例正常（{lock_ratio:.1%} < {max_lock_ratio:.0%}）")

    return "\n".join(result_lines)


# ====================================================================
#  工具导出
# ====================================================================

RISK_TOOLS = [
    calculate_portfolio_var,
    calculate_max_drawdown,
    check_stop_loss,
    check_advanced_stop_loss,
    stress_test,
    check_position_limits,
    calculate_portfolio_beta,
    check_portfolio_circuit_breaker,
    detect_limit_updown,
    check_t1_lock_ratio,
]
