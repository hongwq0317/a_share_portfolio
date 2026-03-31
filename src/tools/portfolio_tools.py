"""组合构建与优化工具

提供均值-方差优化（Ledoit-Wolf 收缩协方差）、最小方差组合、
风险平价、Black-Litterman、等权等组合构建能力。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.tools.data_provider import fetch_stock_history

logger = logging.getLogger("portfolio")


# ====================================================================
#  内部工具函数
# ====================================================================

def _fetch_returns(stock_codes: list[str], days: int = 120) -> Optional[pd.DataFrame]:
    """获取多只股票的日收益率矩阵"""
    price_dict = {}
    for code in stock_codes:
        try:
            df = fetch_stock_history(code, days=days + 30)
            if not df.empty and len(df) > 20:
                price_dict[code] = df.set_index("日期")["收盘"]
        except Exception:
            continue

    if len(price_dict) < 2:
        return None

    prices = pd.DataFrame(price_dict).dropna()
    return prices.pct_change().dropna()


def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf 收缩协方差矩阵估计。

    对角目标（shrink toward diagonal）：
    Σ_shrunk = α·diag(S) + (1-α)·S

    相比原始样本协方差矩阵，收缩估计在高维/小样本场景下更稳定，
    避免优化器产生极端权重。α 越大越保守（趋向等方差独立假设）。
    """
    X = returns.values - returns.values.mean(axis=0)
    n, p = X.shape

    S = X.T @ X / (n - 1)
    F = np.diag(np.diag(S))

    # 计算最优收缩强度 (Ledoit & Wolf 2004, Lemma 3.1 简化版)
    d2 = np.sum((S - F) ** 2) / p

    b2 = 0.0
    for k in range(n):
        xk = np.outer(X[k], X[k])
        b2 += np.sum((xk - S) ** 2)
    b2 /= (n ** 2 * p)

    alpha = min(max(b2 / d2, 0.0), 1.0) if d2 > 0 else 0.5

    logger.debug(f"[组合] Ledoit-Wolf shrinkage intensity: {alpha:.4f}")
    return alpha * F + (1 - alpha) * S


# ====================================================================
#  等权组合
# ====================================================================

@tool
def calculate_equal_weight_portfolio(stock_codes: list[str]) -> str:
    """计算等权重组合的预期表现。

    Args:
        stock_codes: 股票代码列表
    """
    returns = _fetch_returns(stock_codes)
    if returns is None:
        return "数据不足，无法计算"

    n = len(returns.columns)
    weights = np.array([1.0 / n] * n)
    port_return = (returns.mean() * weights).sum() * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = port_return / port_vol if port_vol > 0 else 0

    allocation = "\n".join(f"  {code}: {1 / n * 100:.1f}%" for code in returns.columns)
    return (
        f"等权组合分析 ({n}只股票):\n"
        f"配置:\n{allocation}\n\n"
        f"预期年化收益: {port_return * 100:.2f}%\n"
        f"年化波动率: {port_vol * 100:.2f}%\n"
        f"夏普比率: {sharpe:.2f}\n"
        f"数据区间: {len(returns)}个交易日"
    )


# ====================================================================
#  均值-方差优化（Ledoit-Wolf 收缩）
# ====================================================================

@tool
def calculate_mean_variance_optimization(
    stock_codes: list[str],
    target_return: Optional[float] = None,
    max_weight: float = 0.3,
    current_weights: Optional[list[float]] = None,
    turnover_penalty: float = 0.005,
) -> str:
    """均值-方差优化（马科维茨模型），使用 Ledoit-Wolf 收缩协方差矩阵。

    收缩协方差避免了原始样本协方差在高维/小样本下的不稳定性，
    使优化结果更鲁棒、不易产生极端权重。
    支持换手率惩罚：对偏离当前持仓的权重变动收取隐含交易成本，
    避免因微小优化增益导致的过度交易。

    Args:
        stock_codes: 股票代码列表
        target_return: 目标年化收益率（如0.15表示15%），不指定则求最大夏普
        max_weight: 单只股票最大权重
        current_weights: 当前持仓权重列表（与stock_codes对应），用于计算换手率惩罚
        turnover_penalty: 换手率惩罚系数（默认0.5%单边交易成本）
    """
    returns = _fetch_returns(stock_codes)
    if returns is None:
        return "数据不足，无法优化"

    try:
        from scipy.optimize import minimize

        n = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = _ledoit_wolf_shrinkage(returns) * 252

        w_current = np.zeros(n)
        if current_weights:
            w_current = np.array(current_weights[:n])
            if len(w_current) < n:
                w_current = np.pad(w_current, (0, n - len(w_current)))

        def neg_sharpe(w):
            ret = np.dot(w, mean_returns)
            cost = turnover_penalty * np.sum(np.abs(w - w_current)) if turnover_penalty > 0 else 0
            net_ret = ret - cost
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -net_ret / vol if vol > 0 else 0

        def portfolio_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.dot(w, mean_returns) - target_return,
            })

        bounds = [(0, max_weight)] * n
        init_w = np.array([1.0 / n] * n)

        objective = portfolio_vol if target_return is not None else neg_sharpe
        result = minimize(
            objective, init_w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500},
        )

        fallback_note = ""
        if not result.success and target_return is not None:
            logger.info(
                f"[组合] MVO target_return={target_return:.1%} 不可行 ({result.message})，"
                f"回退到最大夏普"
            )
            fallback_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            result = minimize(
                neg_sharpe, init_w, method="SLSQP",
                bounds=bounds, constraints=fallback_constraints,
                options={"maxiter": 500},
            )
            fallback_note = (
                f"\n⚠ 原目标收益{target_return:.1%}不可行，已自动回退为最大夏普组合"
            )

        if not result.success:
            return f"优化未收敛: {result.message}"

        opt_w = result.x
        opt_ret = np.dot(opt_w, mean_returns)
        opt_vol = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
        opt_sharpe = opt_ret / opt_vol if opt_vol > 0 else 0

        turnover = np.sum(np.abs(opt_w - w_current))
        est_cost = turnover * turnover_penalty

        allocation = []
        for code, w in zip(returns.columns, opt_w):
            if w > 0.01:
                allocation.append(f"  {code}: {w * 100:.1f}%")

        cost_note = ""
        if turnover > 0.01:
            cost_note = (
                f"\n预估换手率: {turnover * 100:.1f}% (单边)"
                f"\n预估交易成本: {est_cost * 100:.3f}%"
            )

        return (
            f"均值-方差优化结果 (Ledoit-Wolf收缩协方差):\n"
            f"{'最大夏普组合' if target_return is None else f'目标收益{target_return * 100:.1f}%组合'}:\n"
            f"配置:\n" + "\n".join(allocation) + "\n\n"
            f"预期年化收益: {opt_ret * 100:.2f}%\n"
            f"年化波动率: {opt_vol * 100:.2f}%\n"
            f"夏普比率: {opt_sharpe:.2f}"
            + cost_note
            + fallback_note
        )
    except Exception as e:
        return f"优化失败: {e}"


# ====================================================================
#  最小方差组合
# ====================================================================

@tool
def calculate_minimum_variance_portfolio(
    stock_codes: list[str],
    max_weight: float = 0.3,
) -> str:
    """最小方差组合：不依赖收益率预测，仅追求最低组合波动率。

    优势：避免了用历史均值估计预期收益的不可靠性（这是经典 MVO 的核心缺陷），
    在实证中通常比最大夏普组合更稳健。

    Args:
        stock_codes: 股票代码列表
        max_weight: 单只股票最大权重
    """
    returns = _fetch_returns(stock_codes)
    if returns is None:
        return "数据不足，无法计算"

    try:
        from scipy.optimize import minimize

        n = len(returns.columns)
        cov_matrix = _ledoit_wolf_shrinkage(returns) * 252

        def portfolio_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight)] * n
        init_w = np.array([1.0 / n] * n)

        result = minimize(
            portfolio_vol, init_w, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        if not result.success:
            return f"优化未收敛: {result.message}"

        opt_w = result.x
        mean_returns = returns.mean() * 252
        opt_ret = np.dot(opt_w, mean_returns)
        opt_vol = portfolio_vol(opt_w)
        opt_sharpe = opt_ret / opt_vol if opt_vol > 0 else 0

        allocation = []
        for code, w in zip(returns.columns, opt_w):
            if w > 0.01:
                allocation.append(f"  {code}: {w * 100:.1f}%")

        return (
            f"最小方差组合 (Ledoit-Wolf收缩协方差):\n"
            f"配置:\n" + "\n".join(allocation) + "\n\n"
            f"历史年化收益(参考): {opt_ret * 100:.2f}%\n"
            f"年化波动率: {opt_vol * 100:.2f}%\n"
            f"夏普比率: {opt_sharpe:.2f}\n"
            f"⚠ 收益数据仅为历史参考，最小方差策略不依赖收益率预测"
        )
    except Exception as e:
        return f"最小方差计算失败: {e}"


# ====================================================================
#  Black-Litterman 模型
# ====================================================================

@tool
def calculate_black_litterman_portfolio(
    stock_codes: list[str],
    views: list[dict],
    max_weight: float = 0.3,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> str:
    """Black-Litterman 组合优化：融合市场均衡收益和主观观点。

    解决经典 MVO 直接用历史均值估计预期收益的问题。BL 模型以市场隐含均衡收益为起点，
    叠加分析师对个股/行业的主观判断，得到更合理的预期收益估计。

    Args:
        stock_codes: 股票代码列表
        views: 观点列表，每项 {"stock": "代码", "return": 年化预期收益率(如0.15=15%), "confidence": 0-1}
        max_weight: 单只股票最大权重
        tau: 不确定性缩放因子（越大越信任主观观点），通常 0.01-0.1
        risk_aversion: 风险厌恶系数（越大越保守），通常 2-4
    """
    returns = _fetch_returns(stock_codes, 252)
    if returns is None:
        return "数据不足，无法计算"

    try:
        from scipy.optimize import minimize

        n = len(returns.columns)
        codes = list(returns.columns)
        cov_matrix = _ledoit_wolf_shrinkage(returns) * 252

        # 用实时市值作为均衡权重（BL 模型的正确实现）
        w_eq = np.array([1.0 / n] * n)
        try:
            from src.tools.data_provider import fetch_stock_quote
            quotes = fetch_stock_quote(codes)
            market_caps = []
            for code in codes:
                q = quotes.get(code, {})
                cap = q.get("total_value", 0) or q.get("market_cap", 0)
                if cap <= 0:
                    cap = q.get("amount", 0) * 100
                market_caps.append(max(cap, 1))
            mc_arr = np.array(market_caps, dtype=float)
            mc_sum = mc_arr.sum()
            if mc_sum > 0:
                w_eq = mc_arr / mc_sum
                logger.debug(f"[BL] 使用市值加权均衡: {dict(zip(codes, (w_eq*100).round(1)))}")
        except Exception as e:
            logger.warning(f"[BL] 获取市值失败，退化为等权均衡: {e}")

        pi = risk_aversion * cov_matrix @ w_eq

        if not views:
            allocation = []
            for code, w, r in zip(codes, w_eq, pi):
                allocation.append(f"  {code}: 权重{w * 100:.1f}%, 隐含收益{r * 100:.2f}%")
            return (
                f"Black-Litterman (无主观观点 = 市场均衡):\n"
                f"配置:\n" + "\n".join(allocation) + "\n\n"
                f"此为市场隐含均衡配置，请传入 views 参数添加主观判断"
            )

        # 构建观点矩阵 P 和 Q
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, v in enumerate(views):
            stock = v.get("stock", "")
            if stock in codes:
                j = codes.index(stock)
                P[i, j] = 1.0
                Q[i] = v.get("return", 0)
                conf = max(0.01, min(1.0, v.get("confidence", 0.5)))
                omega_diag[i] = (1 - conf) / conf * tau * cov_matrix[j, j]

        Omega = np.diag(omega_diag)

        # BL 后验收益: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 · [(τΣ)^-1·π + P'Ω^-1·Q]
        tau_cov_inv = np.linalg.inv(tau * cov_matrix)
        try:
            omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            omega_inv = np.diag(1.0 / (omega_diag + 1e-10))

        M = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
        bl_returns = M @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)
        bl_cov = cov_matrix + M

        # 优化（最大夏普）
        def neg_sharpe(w):
            ret = np.dot(w, bl_returns)
            vol = np.sqrt(np.dot(w.T, np.dot(bl_cov, w)))
            return -ret / vol if vol > 0 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight)] * n
        init_w = np.array([1.0 / n] * n)

        result = minimize(neg_sharpe, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_w = result.x

        opt_ret = np.dot(opt_w, bl_returns)
        opt_vol = np.sqrt(np.dot(opt_w.T, np.dot(bl_cov, opt_w)))
        opt_sharpe = opt_ret / opt_vol if opt_vol > 0 else 0

        allocation = []
        for code, w, eq_r, bl_r in zip(codes, opt_w, pi, bl_returns):
            if w > 0.01:
                allocation.append(
                    f"  {code}: {w * 100:.1f}% (均衡收益{eq_r * 100:.1f}% → BL收益{bl_r * 100:.1f}%)"
                )

        view_str = "\n".join(
            f"  {v.get('stock', '?')}: 预期{v.get('return', 0) * 100:.1f}%, 置信度{v.get('confidence', 0.5):.0%}"
            for v in views
        )

        return (
            f"Black-Litterman 组合优化:\n"
            f"主观观点:\n{view_str}\n\n"
            f"配置:\n" + "\n".join(allocation) + "\n\n"
            f"BL后验年化收益: {opt_ret * 100:.2f}%\n"
            f"年化波动率: {opt_vol * 100:.2f}%\n"
            f"夏普比率: {opt_sharpe:.2f}\n"
            f"τ={tau}, 风险厌恶δ={risk_aversion}"
        )
    except Exception as e:
        return f"Black-Litterman计算失败: {e}"


# ====================================================================
#  风险平价组合
# ====================================================================

@tool
def calculate_risk_parity_portfolio(stock_codes: list[str]) -> str:
    """风险平价组合：使每只股票对组合风险的贡献相等。

    Args:
        stock_codes: 股票代码列表
    """
    returns = _fetch_returns(stock_codes)
    if returns is None:
        return "数据不足，无法计算"

    try:
        from scipy.optimize import minimize

        n = len(returns.columns)
        cov_matrix = _ledoit_wolf_shrinkage(returns) * 252

        def risk_parity_obj(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if port_vol < 1e-10:
                return 1e10
            marginal_contrib = np.dot(cov_matrix, w) / port_vol
            risk_contrib = w * marginal_contrib
            target_risk = port_vol / n
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.02, 0.5)] * n
        init_w = np.array([1.0 / n] * n)

        result = minimize(
            risk_parity_obj, init_w, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        opt_w = result.x
        mean_returns = returns.mean() * 252
        opt_ret = np.dot(opt_w, mean_returns)
        opt_vol = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
        opt_sharpe = opt_ret / opt_vol if opt_vol > 0 else 0

        allocation = []
        for code, w in zip(returns.columns, opt_w):
            allocation.append(f"  {code}: {w * 100:.1f}%")

        return (
            f"风险平价组合 (Ledoit-Wolf收缩协方差):\n"
            f"配置:\n" + "\n".join(allocation) + "\n\n"
            f"预期年化收益: {opt_ret * 100:.2f}%\n"
            f"年化波动率: {opt_vol * 100:.2f}%\n"
            f"夏普比率: {opt_sharpe:.2f}"
        )
    except Exception as e:
        return f"风险平价计算失败: {e}"


# ====================================================================
#  再平衡 & 相关性
# ====================================================================

@tool
def calculate_rebalance_trades(
    current_holdings: dict[str, float],
    target_weights: dict[str, float],
    total_value: float,
    min_trade_amount: float = 5000,
) -> str:
    """计算从当前持仓到目标组合的调仓交易单。

    Args:
        current_holdings: 当前持仓 {股票代码: 当前市值}
        target_weights: 目标权重 {股票代码: 目标占比(0-1)}
        total_value: 组合总市值
        min_trade_amount: 最小交易金额
    """
    trades = []

    all_codes = set(list(current_holdings.keys()) + list(target_weights.keys()))
    for code in sorted(all_codes):
        current_value = current_holdings.get(code, 0)
        target_value = target_weights.get(code, 0) * total_value
        diff = target_value - current_value

        if abs(diff) < min_trade_amount:
            continue

        direction = "买入" if diff > 0 else "卖出"
        trades.append(f"  {direction} {code}: {abs(diff):.0f}元 "
                       f"(当前{current_value:.0f} → 目标{target_value:.0f})")

    if not trades:
        return "无需调仓：所有持仓已接近目标配置"

    buy_total = sum(
        target_weights.get(c, 0) * total_value - current_holdings.get(c, 0)
        for c in all_codes
        if target_weights.get(c, 0) * total_value - current_holdings.get(c, 0) > min_trade_amount
    )
    sell_total = sum(
        current_holdings.get(c, 0) - target_weights.get(c, 0) * total_value
        for c in all_codes
        if current_holdings.get(c, 0) - target_weights.get(c, 0) * total_value > min_trade_amount
    )

    est_cost = (buy_total + sell_total) * 0.001
    return (
        f"调仓交易计划:\n"
        + "\n".join(trades)
        + f"\n\n总买入: {buy_total:.0f}元, 总卖出: {sell_total:.0f}元"
        + f"\n预估交易成本: {est_cost:,.0f}元 (按双边0.1%估算)"
    )


@tool
def calculate_correlation_matrix(stock_codes: list[str]) -> str:
    """计算股票之间的相关性矩阵。

    Args:
        stock_codes: 股票代码列表
    """
    returns = _fetch_returns(stock_codes)
    if returns is None:
        return "数据不足，无法计算相关性"

    corr = returns.corr()
    avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()

    high_corr_pairs = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if abs(corr.iloc[i, j]) > 0.7:
                high_corr_pairs.append(
                    f"  {corr.index[i]} - {corr.columns[j]}: {corr.iloc[i, j]:.3f}"
                )

    result = f"相关性矩阵:\n{corr.round(3).to_string()}\n\n平均相关系数: {avg_corr:.3f}"
    if high_corr_pairs:
        result += f"\n\n高相关性配对(>0.7):\n" + "\n".join(high_corr_pairs)
    else:
        result += "\n\n未发现高相关性配对(>0.7)，分散化较好"
    return result


# ====================================================================
#  工具导出
# ====================================================================

PORTFOLIO_TOOLS = [
    calculate_equal_weight_portfolio,
    calculate_mean_variance_optimization,
    calculate_minimum_variance_portfolio,
    calculate_black_litterman_portfolio,
    calculate_risk_parity_portfolio,
    calculate_rebalance_trades,
    calculate_correlation_matrix,
]
