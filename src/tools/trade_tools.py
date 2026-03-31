"""交易执行工具

提供模拟交易、市场冲击估算、涨跌停检测和持仓管理能力。
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("portfolio")


def _estimate_market_impact(order_amount: float, daily_turnover: float) -> float:
    """估算市场冲击成本（基于 Almgren-Chriss 简化模型）。

    冲击成本 ≈ k × sqrt(order_amount / daily_turnover)
    k 为经验系数，A 股通常取 0.1~0.3。

    Returns:
        预估冲击成本（占交易金额的比例，如 0.002 = 0.2%）
    """
    if daily_turnover <= 0:
        return 0.005
    participation = order_amount / daily_turnover
    k = 0.15
    return k * math.sqrt(participation)


def _check_limit_status(stock_code: str, name: str, direction: str) -> Optional[str]:
    """检查股票是否处于涨跌停，返回阻止交易的原因（如有）。"""
    try:
        from src.tools.data_provider import fetch_stock_quote
        quotes = fetch_stock_quote([stock_code])
        if not quotes or stock_code not in quotes:
            return None
        q = quotes[stock_code]
        change_pct = q.get("change_pct", 0)
        qname = q.get("name", name)

        if stock_code.startswith(("300", "301", "688", "689")):
            limit_pct = 20.0
        elif stock_code.startswith(("4", "8")) and not stock_code.startswith("88"):
            limit_pct = 30.0
        else:
            limit_pct = 10.0
        if "ST" in qname.upper():
            limit_pct = 5.0

        if direction == "buy" and change_pct >= limit_pct - 0.1:
            return f"涨停({change_pct:+.2f}%)无法买入，等回调后再考虑"
        if direction == "sell" and change_pct <= -(limit_pct - 0.1):
            return f"跌停({change_pct:+.2f}%)可能无法卖出，挂单或次日再卖"
    except Exception:
        pass
    return None

PORTFOLIO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


def _ensure_data_dir():
    os.makedirs(PORTFOLIO_DATA_DIR, exist_ok=True)


def _load_portfolio_state() -> dict:
    """加载持仓状态"""
    _ensure_data_dir()
    path = os.path.join(PORTFOLIO_DATA_DIR, "portfolio_state.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "positions": {},
        "closed_positions": [],
        "cash": 1000000,
        "initial_capital": 1000000,
        "realized_pnl": 0,
        "total_fees": 0,
        "total_invested": 0,
        "trade_history": [],
        "created_at": datetime.now().isoformat(),
    }


def _save_portfolio_state(state: dict):
    """保存持仓状态"""
    _ensure_data_dir()
    state["updated_at"] = datetime.now().isoformat()
    path = os.path.join(PORTFOLIO_DATA_DIR, "portfolio_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


@tool
def get_current_portfolio() -> str:
    """获取当前组合持仓详情。"""
    state = _load_portfolio_state()
    positions = state.get("positions", {})
    cash = state.get("cash", 0)

    if not positions:
        return f"当前无持仓\n现金余额: {cash:,.0f}元"

    today = datetime.now().strftime("%Y-%m-%d")
    lines = ["当前持仓:"]
    total_value = cash
    for code, pos in sorted(positions.items()):
        price = pos.get("current_price", pos.get("avg_cost", 0))
        total_shares = pos.get("shares", 0)
        mv = total_shares * price
        pnl_pct = (price / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0
        pnl_amount = (price - pos["avg_cost"]) * total_shares
        total_value += mv
        if pos.get("last_buy_date") == today:
            today_bought = pos.get("today_bought_shares", total_shares)
        elif pos.get("buy_date") == today and "last_buy_date" not in pos:
            today_bought = total_shares
        else:
            today_bought = 0
        sellable = total_shares - today_bought
        t1_info = f", 可卖{sellable}股" if today_bought > 0 else ""
        lines.append(
            f"  {pos.get('name', code)}({code}): "
            f"{total_shares}股{t1_info}, 成本{pos['avg_cost']:.2f}, "
            f"现价{price:.2f}, "
            f"市值{mv:,.0f}元, 盈亏{pnl_pct:+.1f}%({pnl_amount:+,.0f}元)"
        )

    initial_capital = state.get("initial_capital", 1000000)
    realized_pnl = state.get("realized_pnl", 0)
    total_fees = state.get("total_fees", 0)
    unrealized_pnl = sum(
        (pos.get("current_price", pos["avg_cost"]) - pos["avg_cost"]) * pos["shares"]
        for pos in positions.values()
    )
    total_return = realized_pnl + unrealized_pnl - total_fees
    total_return_pct = total_return / initial_capital * 100 if initial_capital > 0 else 0

    lines.append(f"\n现金: {cash:,.0f}元")
    lines.append(f"总资产: {total_value:,.0f}元")
    lines.append(f"仓位: {(total_value - cash) / total_value * 100:.1f}%" if total_value > 0 else "仓位: 0%")
    lines.append(f"\n已实现盈亏: {realized_pnl:+,.0f}元")
    lines.append(f"未实现盈亏: {unrealized_pnl:+,.0f}元")
    lines.append(f"累计收益: {total_return:+,.0f}元 ({total_return_pct:+.2f}%)")
    lines.append(f"累计费用: {total_fees:,.0f}元")

    return "\n".join(lines)


@tool
def simulate_buy(
    stock_code: str,
    stock_name: str,
    price: float,
    amount: float,
    reason: str = "",
) -> str:
    """模拟买入股票。

    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        price: 买入价格
        amount: 买入金额（元）
        reason: 买入理由
    """
    limit_msg = _check_limit_status(stock_code, stock_name, "buy")
    if limit_msg:
        return f"买入失败: {stock_name}({stock_code}) {limit_msg}"

    state = _load_portfolio_state()
    cash = state.get("cash", 0)

    min_lot_cost = price * 100
    shares = int(amount / price / 100) * 100
    if shares <= 0:
        if cash >= min_lot_cost + 5:
            shares = 100
            logger.info(
                f"[交易] {stock_name}({stock_code}) 传入金额{amount:.0f}元不足一手"
                f"({min_lot_cost:.0f}元), 自动调整为1手(100股)"
            )
        else:
            return (
                f"买入失败: 金额{amount:.0f}元不足一手 "
                f"(需至少{min_lot_cost:.0f}元, 可用现金{cash:,.0f}元)"
            )

    actual_amount = shares * price
    commission = max(actual_amount * 0.0003, 5)

    daily_turnover = actual_amount * 50
    try:
        from src.tools.data_provider import fetch_stock_quote
        q = fetch_stock_quote([stock_code])
        if q and stock_code in q:
            reported_amount = q[stock_code].get("amount", 0)
            if reported_amount > 0:
                daily_turnover = reported_amount
    except Exception:
        pass
    impact_pct = _estimate_market_impact(actual_amount, daily_turnover)
    impact_cost = actual_amount * impact_pct
    total_cost = actual_amount + commission

    if total_cost > cash:
        max_shares = int(cash / price / 100) * 100
        return f"买入失败: 现金不足 (需{total_cost:,.0f}元, 可用{cash:,.0f}元, 最多买{max_shares}股)"

    today = datetime.now().strftime("%Y-%m-%d")
    positions = state.get("positions", {})
    if stock_code in positions:
        old = positions[stock_code]
        old_value = old["shares"] * old["avg_cost"]
        new_shares = old["shares"] + shares
        new_avg_cost = (old_value + actual_amount) / new_shares
        prev_today = old.get("today_bought_shares", 0) if old.get("last_buy_date") == today else 0
        positions[stock_code] = {
            **old,
            "shares": new_shares,
            "avg_cost": round(new_avg_cost, 3),
            "current_price": price,
            "last_buy_date": today,
            "today_bought_shares": prev_today + shares,
        }
    else:
        positions[stock_code] = {
            "name": stock_name,
            "shares": shares,
            "avg_cost": price,
            "current_price": price,
            "sector": "",
            "buy_date": today,
            "last_buy_date": today,
            "today_bought_shares": shares,
        }

    state["positions"] = positions
    state["cash"] = cash - total_cost
    state["total_fees"] = state.get("total_fees", 0) + commission
    state["trade_history"].append({
        "time": datetime.now().isoformat(),
        "direction": "buy",
        "code": stock_code,
        "name": stock_name,
        "price": price,
        "shares": shares,
        "amount": actual_amount,
        "commission": round(commission, 2),
        "reason": reason,
    })

    _save_portfolio_state(state)
    return (
        f"买入成功: {stock_name}({stock_code})\n"
        f"  价格: {price:.2f}, 数量: {shares}股, 金额: {actual_amount:,.0f}元\n"
        f"  手续费: {commission:.2f}元, 预估冲击成本: {impact_cost:.0f}元({impact_pct*100:.2f}%)\n"
        f"  剩余现金: {state['cash']:,.0f}元\n"
        f"  买入理由: {reason}"
    )


@tool
def simulate_sell(
    stock_code: str,
    price: float,
    shares: Optional[int] = None,
    reason: str = "",
) -> str:
    """模拟卖出股票（遵循A股T+1规则：当日买入的股票不可当日卖出）。

    Args:
        stock_code: 股票代码
        price: 卖出价格
        shares: 卖出数量（股），不指定则卖出全部可卖股份（受T+1限制）
        reason: 卖出理由
    """
    state = _load_portfolio_state()
    positions = state.get("positions", {})

    if stock_code not in positions:
        return f"卖出失败: 未持有 {stock_code}"

    pos = positions[stock_code]

    limit_msg = _check_limit_status(stock_code, pos.get("name", stock_code), "sell")
    if limit_msg:
        return f"卖出警告: {pos.get('name', stock_code)}({stock_code}) {limit_msg}（仍尝试挂单）"

    today = datetime.now().strftime("%Y-%m-%d")
    if pos.get("last_buy_date") == today:
        today_bought = pos.get("today_bought_shares", pos["shares"])
    elif pos.get("buy_date") == today and "last_buy_date" not in pos:
        today_bought = pos["shares"]
    else:
        today_bought = 0
    sellable_shares = pos["shares"] - today_bought
    if sellable_shares <= 0:
        return (
            f"卖出失败: T+1限制，{pos.get('name', stock_code)}({stock_code}) "
            f"全部{pos['shares']}股均为今日买入，需下一个交易日才能卖出"
        )

    sell_shares = shares if shares and shares <= sellable_shares else sellable_shares
    sell_shares = (sell_shares // 100) * 100
    if sell_shares <= 0:
        return (
            f"卖出失败: 可卖股数不足一手 "
            f"(持有{pos['shares']}股, 今日买入{today_bought}股, 可卖{sellable_shares}股)"
        )

    actual_amount = sell_shares * price
    commission = max(actual_amount * 0.0003, 5)
    stamp_duty = actual_amount * 0.0005
    net_amount = actual_amount - commission - stamp_duty
    trade_fees = commission + stamp_duty

    pnl = (price / pos["avg_cost"] - 1) * 100
    pnl_amount = (price - pos["avg_cost"]) * sell_shares

    remaining = pos["shares"] - sell_shares
    is_full_sell = remaining <= 0

    if is_full_sell:
        # 清仓：保存到历史持仓
        closed_entry = {
            "code": stock_code,
            "name": pos.get("name", stock_code),
            "sector": pos.get("sector", ""),
            "buy_date": pos.get("buy_date", ""),
            "sell_date": datetime.now().strftime("%Y-%m-%d"),
            "shares": sell_shares,
            "avg_cost": pos["avg_cost"],
            "sell_price": price,
            "buy_amount": round(pos["avg_cost"] * sell_shares, 2),
            "sell_amount": round(actual_amount, 2),
            "realized_pnl": round(pnl_amount, 2),
            "realized_pnl_pct": round(pnl, 2),
            "reason": reason,
        }
        if "closed_positions" not in state:
            state["closed_positions"] = []
        state["closed_positions"].append(closed_entry)
        del positions[stock_code]
    else:
        updated_pos = {**pos, "shares": remaining, "current_price": price}
        tbs = updated_pos.get("today_bought_shares", 0)
        if tbs > remaining:
            updated_pos["today_bought_shares"] = remaining
        positions[stock_code] = updated_pos

    state["positions"] = positions
    state["cash"] = state.get("cash", 0) + net_amount
    state["realized_pnl"] = state.get("realized_pnl", 0) + pnl_amount
    state["total_fees"] = state.get("total_fees", 0) + trade_fees
    state["trade_history"].append({
        "time": datetime.now().isoformat(),
        "direction": "sell",
        "code": stock_code,
        "name": pos.get("name", stock_code),
        "price": price,
        "shares": sell_shares,
        "amount": actual_amount,
        "commission": round(commission, 2),
        "stamp_duty": round(stamp_duty, 2),
        "pnl_pct": round(pnl, 2),
        "pnl_amount": round(pnl_amount, 2),
        "reason": reason,
    })

    _save_portfolio_state(state)
    remaining_info = "清仓（已归入历史持仓）" if is_full_sell else f"剩余{remaining}股(今日买入{today_bought}股不可卖)"
    return (
        f"卖出成功: {pos.get('name', stock_code)}({stock_code})\n"
        f"  价格: {price:.2f}, 数量: {sell_shares}股, 金额: {actual_amount:,.0f}元\n"
        f"  手续费: {commission:.2f}元, 印花税: {stamp_duty:.2f}元\n"
        f"  盈亏: {pnl:+.2f}% ({pnl_amount:+,.0f}元)\n"
        f"  {remaining_info}\n"
        f"  现金余额: {state['cash']:,.0f}元\n"
        f"  卖出理由: {reason}"
    )


@tool
def get_trade_history(last_n: int = 20) -> str:
    """获取最近交易记录。

    Args:
        last_n: 显示最近N条记录
    """
    state = _load_portfolio_state()
    history = state.get("trade_history", [])

    if not history:
        return "无交易记录"

    lines = [f"最近{min(last_n, len(history))}条交易记录:"]
    for trade in history[-last_n:]:
        direction = "买入" if trade["direction"] == "buy" else "卖出"
        pnl_str = ""
        if trade["direction"] == "sell":
            pnl_str = f", 盈亏{trade.get('pnl_pct', 0):+.1f}%({trade.get('pnl_amount', 0):+,.0f}元)"
        lines.append(
            f"  [{trade['time'][:16]}] {direction} "
            f"{trade.get('name', trade['code'])}({trade['code']}): "
            f"{trade['shares']}股 @{trade['price']:.2f}{pnl_str}"
        )

    buy_count = sum(1 for t in history if t["direction"] == "buy")
    sell_count = sum(1 for t in history if t["direction"] == "sell")
    total_pnl = sum(t.get("pnl_amount", 0) for t in history if t["direction"] == "sell")
    win_trades = sum(1 for t in history if t["direction"] == "sell" and t.get("pnl_amount", 0) > 0)
    win_rate = win_trades / sell_count * 100 if sell_count > 0 else 0

    lines.append(f"\n统计: 买入{buy_count}次, 卖出{sell_count}次, 胜率{win_rate:.0f}%, 总已实现盈亏{total_pnl:+,.0f}元")
    return "\n".join(lines)


@tool
def update_portfolio_prices(price_updates: dict[str, float]) -> str:
    """更新持仓股票的最新价格。

    Args:
        price_updates: 价格更新 {股票代码: 最新价格}
    """
    state = _load_portfolio_state()
    positions = state.get("positions", {})

    updated = []
    for code, new_price in price_updates.items():
        if code in positions:
            old_price = positions[code].get("current_price", positions[code]["avg_cost"])
            positions[code]["current_price"] = new_price
            change = (new_price / old_price - 1) * 100 if old_price > 0 else 0
            updated.append(f"  {code}: {old_price:.2f} → {new_price:.2f} ({change:+.2f}%)")

    state["positions"] = positions
    _save_portfolio_state(state)

    if updated:
        return "价格更新:\n" + "\n".join(updated)
    return "无持仓需要更新"


@tool
def reset_portfolio(initial_capital: float = 1000000) -> str:
    """重置组合到初始状态。

    Args:
        initial_capital: 初始资金
    """
    state = {
        "positions": {},
        "closed_positions": [],
        "cash": initial_capital,
        "initial_capital": initial_capital,
        "realized_pnl": 0,
        "total_fees": 0,
        "total_invested": 0,
        "trade_history": [],
        "created_at": datetime.now().isoformat(),
    }
    _save_portfolio_state(state)
    return f"组合已重置，初始资金: {initial_capital:,.0f}元"


@tool
def estimate_market_impact(
    stock_code: str,
    order_amount: float,
    daily_turnover: float = 0,
) -> str:
    """估算交易的市场冲击成本（Almgren-Chriss 简化模型）。

    市场冲击 = 大额委托推动价格偏离的隐性成本。参与率越高（委托量/日均成交越大），
    冲击越大。对低流动性股票尤其重要。

    Args:
        stock_code: 股票代码
        order_amount: 委托金额（元）
        daily_turnover: 日均成交额（元），0 则自动获取
    """
    if daily_turnover <= 0:
        try:
            from src.tools.data_provider import fetch_stock_quote
            quotes = fetch_stock_quote([stock_code])
            if quotes and stock_code in quotes:
                daily_turnover = quotes[stock_code].get("amount", 0)
        except Exception:
            pass

    if daily_turnover <= 0:
        return f"{stock_code}: 无法获取成交额数据，请手动提供 daily_turnover 参数"

    participation = order_amount / daily_turnover
    impact = _estimate_market_impact(order_amount, daily_turnover)
    impact_amount = order_amount * impact

    days_to_liquidate = order_amount / (daily_turnover * 0.1) if daily_turnover > 0 else float("inf")

    risk_level = "低" if participation < 0.01 else ("中" if participation < 0.05 else ("高" if participation < 0.2 else "极高"))

    return (
        f"{stock_code} 市场冲击估算:\n"
        f"  委托金额: {order_amount:,.0f}元\n"
        f"  日均成交额: {daily_turnover:,.0f}元\n"
        f"  参与率: {participation * 100:.2f}% (委托量/日均成交)\n"
        f"  预估冲击成本: {impact * 100:.3f}% ({impact_amount:,.0f}元)\n"
        f"  预估卖出天数: {days_to_liquidate:.1f}天 (按10%参与率)\n"
        f"  冲击风险: {risk_level}\n"
        f"  建议: {'可一次性交易' if participation < 0.05 else '建议分批交易以降低冲击' if participation < 0.2 else '委托量过大，必须分多日执行'}"
    )


TRADE_TOOLS = [
    get_current_portfolio,
    simulate_buy,
    simulate_sell,
    get_trade_history,
    update_portfolio_prices,
    reset_portfolio,
    estimate_market_impact,
]
