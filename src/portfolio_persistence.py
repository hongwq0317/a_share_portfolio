"""持仓数据持久化与图状态同步

负责:
  1. 启动时从 data/portfolio_state.json 加载持仓 → 注入 graph input_state
  2. 启动时用实时行情刷新持仓价格（含沪深300基准）
  3. 节点执行完交易后，从 JSON 重新加载并同步回 graph state
  4. 管理历史持仓（closed_positions）和累计收益统计
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger("portfolio")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
STATE_FILE = os.path.join(DATA_DIR, "portfolio_state.json")


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_raw() -> dict:
    """原始读取 portfolio_state.json"""
    _ensure_dir()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_raw(data: dict):
    """原始写入 portfolio_state.json"""
    _ensure_dir()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _migrate_state(raw_state: dict) -> dict:
    """迁移旧格式数据：补充缺失字段，从 trade_history 重建 closed_positions。"""
    changed = False

    if "initial_capital" not in raw_state:
        raw_state["initial_capital"] = 1000000
        changed = True

    if "realized_pnl" not in raw_state:
        total = sum(
            t.get("pnl_amount", 0)
            for t in raw_state.get("trade_history", [])
            if t.get("direction") == "sell"
        )
        raw_state["realized_pnl"] = round(total, 2)
        changed = True

    if "total_fees" not in raw_state:
        total = sum(
            t.get("commission", 0) + t.get("stamp_duty", 0)
            for t in raw_state.get("trade_history", [])
        )
        raw_state["total_fees"] = round(total, 2)
        changed = True

    if "closed_positions" not in raw_state:
        raw_state["closed_positions"] = _rebuild_closed_positions(raw_state)
        changed = True

    if changed:
        save_raw(raw_state)
        logger.info("[数据迁移] 已补充缺失字段（closed_positions, realized_pnl, total_fees 等）")

    return raw_state


def _rebuild_closed_positions(raw_state: dict) -> list:
    """从 trade_history 重建已平仓记录。

    对每个已完全卖出的股票，从买卖配对中生成 closed_positions 记录。
    """
    history = raw_state.get("trade_history", [])
    current_codes = set(raw_state.get("positions", {}).keys())

    # 收集每个代码的买卖记录
    code_buys: Dict[str, list] = {}
    code_sells: Dict[str, list] = {}
    code_names: Dict[str, str] = {}
    code_sectors: Dict[str, str] = {}

    for t in history:
        code = t["code"]
        code_names[code] = t.get("name", code)
        if t["direction"] == "buy":
            code_buys.setdefault(code, []).append(t)
        else:
            code_sells.setdefault(code, []).append(t)

    closed = []
    for code, sells in code_sells.items():
        buys = code_buys.get(code, [])
        if not buys:
            continue

        first_buy_date = buys[0].get("time", "")[:10]
        avg_cost = buys[0].get("price", 0)
        if len(buys) > 1:
            total_shares = sum(b.get("shares", 0) for b in buys)
            total_cost = sum(b.get("shares", 0) * b.get("price", 0) for b in buys)
            avg_cost = total_cost / total_shares if total_shares > 0 else 0

        for sell in sells:
            sell_shares = sell.get("shares", 0)
            sell_price = sell.get("price", 0)
            pnl = sell.get("pnl_amount", 0)
            pnl_pct = sell.get("pnl_pct", 0)

            closed.append({
                "code": code,
                "name": code_names.get(code, code),
                "sector": code_sectors.get(code, ""),
                "buy_date": first_buy_date,
                "sell_date": sell.get("time", "")[:10],
                "shares": sell_shares,
                "avg_cost": round(avg_cost, 2),
                "sell_price": sell_price,
                "buy_amount": round(avg_cost * sell_shares, 2),
                "sell_amount": round(sell_price * sell_shares, 2),
                "realized_pnl": round(pnl, 2),
                "realized_pnl_pct": round(pnl_pct, 2),
                "reason": sell.get("reason", ""),
            })

    closed.sort(key=lambda x: x.get("sell_date", ""))
    return closed


def _fetch_benchmark_price() -> float:
    """获取沪深300当前价格，失败返回 0。"""
    try:
        from src.tools.data_provider import fetch_index_quote
        quotes = fetch_index_quote(["000300"])
        return quotes.get("000300", {}).get("price", 0)
    except Exception as e:
        logger.warning(f"[基准] 获取沪深300行情失败: {e}")
        return 0


def refresh_prices(raw_state: dict) -> dict:
    """用实时行情刷新持仓价格，同时更新沪深300基准。"""
    positions = raw_state.get("positions", {})

    # 刷新个股价格
    if positions:
        codes = list(positions.keys())
        try:
            from src.tools.data_provider import fetch_stock_quote
            quotes = fetch_stock_quote(codes)
        except Exception as e:
            logger.warning(f"[持仓刷新] 获取实时行情失败，保持原价格: {e}")
            quotes = {}

        updated_count = 0
        for code, pos in positions.items():
            q = quotes.get(code)
            if q and q.get("price", 0) > 0:
                old_price = pos.get("current_price", pos.get("avg_cost", 0))
                pos["current_price"] = q["price"]
                if old_price and old_price > 0:
                    change = (q["price"] / old_price - 1) * 100
                    logger.info(
                        f"  {pos.get('name', code)}({code}): "
                        f"{old_price:.2f} → {q['price']:.2f} ({change:+.2f}%)"
                    )
                updated_count += 1

        raw_state["positions"] = positions
        if updated_count > 0:
            logger.info(f"[持仓刷新] 已更新 {updated_count}/{len(positions)} 只股票价格")

    # 刷新沪深300基准
    benchmark_price = _fetch_benchmark_price()
    if benchmark_price > 0:
        if "benchmark_start_price" not in raw_state or raw_state["benchmark_start_price"] <= 0:
            raw_state["benchmark_start_price"] = benchmark_price
            logger.info(f"[基准] 记录沪深300起始价: {benchmark_price:.2f}")
        raw_state["benchmark_current_price"] = benchmark_price
        logger.info(f"[基准] 沪深300当前: {benchmark_price:.2f}")

    raw_state["price_updated_at"] = datetime.now().isoformat()
    save_raw(raw_state)
    return raw_state


def _compute_portfolio_metrics(raw_state: dict) -> Dict[str, Any]:
    """计算组合级别的收益指标。"""
    positions = raw_state.get("positions", {})
    cash = raw_state.get("cash", 0)
    initial_capital = raw_state.get("initial_capital", 1000000)
    realized_pnl = raw_state.get("realized_pnl", 0)
    total_fees = raw_state.get("total_fees", 0)

    unrealized_pnl = 0.0
    total_market_value = 0.0

    for pos in positions.values():
        price = pos.get("current_price", pos.get("avg_cost", 0))
        shares = pos.get("shares", 0)
        avg_cost = pos.get("avg_cost", 0)
        mv = shares * price
        total_market_value += mv
        unrealized_pnl += (price - avg_cost) * shares

    total_value = total_market_value + cash
    total_return = realized_pnl + unrealized_pnl
    net_return = total_value - initial_capital
    total_return_pct = net_return / initial_capital * 100 if initial_capital > 0 else 0

    benchmark_start = raw_state.get("benchmark_start_price", 0)
    benchmark_current = raw_state.get("benchmark_current_price", 0)
    benchmark_return_pct = 0.0
    if benchmark_start > 0 and benchmark_current > 0:
        benchmark_return_pct = (benchmark_current / benchmark_start - 1) * 100

    return {
        "total_value": round(total_value, 2),
        "total_market_value": round(total_market_value, 2),
        "cash": round(cash, 2),
        "initial_capital": initial_capital,
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_return": round(net_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_fees": round(total_fees, 2),
        "benchmark_return_pct": round(benchmark_return_pct, 2),
        "benchmark_start_price": benchmark_start,
        "benchmark_current_price": benchmark_current,
    }


def raw_to_graph_state(raw_state: dict) -> Dict[str, Any]:
    """将 portfolio_state.json 格式转换为 PortfolioState 需要的字段。"""
    positions = raw_state.get("positions", {})
    cash = raw_state.get("cash", 0)

    graph_positions: Dict[str, Any] = {}
    total_market_value = 0.0

    for code, pos in positions.items():
        price = pos.get("current_price", pos.get("avg_cost", 0))
        shares = pos.get("shares", 0)
        avg_cost = pos.get("avg_cost", 0)
        mv = shares * price
        total_market_value += mv

        pnl_pct = (price / avg_cost - 1) * 100 if avg_cost > 0 else 0
        pnl_amount = (price - avg_cost) * shares

        today = datetime.now().strftime("%Y-%m-%d")
        if pos.get("last_buy_date") == today:
            today_bought = pos.get("today_bought_shares", shares)
        elif pos.get("buy_date") == today and "last_buy_date" not in pos:
            today_bought = shares
        else:
            today_bought = 0
        today_bought = min(today_bought, shares)
        sellable = max(shares - today_bought, 0)

        graph_positions[code] = {
            "stock_code": code,
            "stock_name": pos.get("name", code),
            "sector": pos.get("sector", ""),
            "shares": shares,
            "sellable_shares": sellable,
            "avg_cost": avg_cost,
            "current_price": price,
            "market_value": round(mv, 2),
            "unrealized_pnl": round(pnl_amount, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
            "buy_date": pos.get("buy_date", ""),
        }

    total_value = total_market_value + cash

    for code, gp in graph_positions.items():
        gp["weight"] = round(gp["market_value"] / total_value * 100, 2) if total_value > 0 else 0

    return {
        "current_positions": graph_positions,
        "portfolio_value": round(total_value, 2),
        "cash_balance": round(cash, 2),
    }


def load_for_graph(do_refresh: bool = True) -> Dict[str, Any]:
    """完整加载 → 迁移 → 刷新价格 → 转换为 graph state 字段。"""
    raw = load_raw()
    if not raw:
        return {"current_positions": {}, "portfolio_value": 0, "cash_balance": 0}

    raw = _migrate_state(raw)

    if do_refresh and raw.get("positions"):
        logger.info("[持仓加载] 正在用实时行情刷新持仓价格...")
        raw = refresh_prices(raw)

    state_fields = raw_to_graph_state(raw)
    metrics = _compute_portfolio_metrics(raw)

    n = len(state_fields["current_positions"])
    pv = state_fields["portfolio_value"]
    cash = state_fields["cash_balance"]

    if pv > 0:
        logger.info(
            f"[持仓加载] 持仓 {n} 只, 总资产 {pv:,.0f}元, "
            f"现金 {cash:,.0f}元, 仓位 {(pv - cash) / pv * 100:.1f}%"
        )
        logger.info(
            f"[收益统计] 已实现: {metrics['realized_pnl']:+,.0f}元, "
            f"未实现: {metrics['unrealized_pnl']:+,.0f}元, "
            f"累计收益: {metrics['total_return']:+,.0f}元 ({metrics['total_return_pct']:+.2f}%), "
            f"沪深300: {metrics['benchmark_return_pct']:+.2f}%"
        )
    else:
        logger.info(f"[持仓加载] 空仓, 现金 {cash:,.0f}元")

    closed_count = len(raw.get("closed_positions", []))
    if closed_count > 0:
        logger.info(f"[历史持仓] 共 {closed_count} 条已平仓记录")

    return state_fields


def sync_after_trades() -> Dict[str, Any]:
    """交易执行后重新加载 JSON 并转换为 graph state 字段。"""
    raw = load_raw()
    if not raw:
        return {"current_positions": {}, "portfolio_value": 0, "cash_balance": 0}
    return raw_to_graph_state(raw)


def format_positions_summary(state_fields: Dict[str, Any], raw_state: dict | None = None) -> str:
    """生成持仓的文本摘要，供 LLM prompt 使用。包含收益统计和基准对比。

    Args:
        state_fields: graph state 字段（current_positions, cash_balance, portfolio_value）
        raw_state: 可选的原始 JSON 数据，避免重复读取文件
    """
    positions = state_fields.get("current_positions", {})
    cash = state_fields.get("cash_balance", 0)
    total_value = state_fields.get("portfolio_value", 0)

    if not positions:
        return f"当前无持仓\n现金余额: {cash:,.0f}元"

    raw = raw_state if raw_state is not None else load_raw()
    metrics = _compute_portfolio_metrics(raw)

    # 区分可卖和T+1锁定的持仓
    t1_locked = []
    sellable_positions = []
    for code in sorted(positions.keys()):
        p = positions[code]
        sellable = p.get("sellable_shares", p["shares"])
        if sellable < p["shares"]:
            t1_locked.append((code, p, sellable))
        else:
            sellable_positions.append((code, p))

    lines = ["当前持仓:"]
    for code, p, *rest in sellable_positions + [(c, p, s) for c, p, s in t1_locked]:
        sellable = rest[0] if rest else p["shares"]
        t1_tag = " ⛔T+1不可卖" if sellable == 0 else (f" ⚠可卖{sellable}股" if sellable < p["shares"] else "")
        lines.append(
            f"  {p['stock_name']}({code}): "
            f"{p['shares']}股{t1_tag}, 成本{p['avg_cost']:.2f}, "
            f"现价{p['current_price']:.2f}, "
            f"市值{p['market_value']:,.0f}元, "
            f"盈亏{p['unrealized_pnl_pct']:+.1f}%({p['unrealized_pnl']:+,.0f}元), "
            f"权重{p['weight']:.1f}%"
        )

    # T+1 警告段落
    if t1_locked:
        locked_names = [f"{p['stock_name']}({code})" for code, p, _ in t1_locked]
        lines.append(
            f"\n⛔ T+1限制: 以下{len(t1_locked)}只股票今日买入，今日不可卖出/减持: "
            + "、".join(locked_names)
        )

    stock_value = total_value - cash
    lines.append(f"\n现金: {cash:,.0f}元")
    lines.append(f"总资产: {total_value:,.0f}元")
    lines.append(f"股票仓位: {stock_value / total_value * 100:.1f}%" if total_value > 0 else "仓位: 0%")

    lines.append("\n--- 收益统计 ---")
    lines.append(f"已实现盈亏: {metrics['realized_pnl']:+,.0f}元")
    lines.append(f"未实现盈亏: {metrics['unrealized_pnl']:+,.0f}元")
    lines.append(f"累计收益: {metrics['total_return']:+,.0f}元 ({metrics['total_return_pct']:+.2f}%)")
    lines.append(f"累计费用: {metrics['total_fees']:,.0f}元")
    if metrics["benchmark_start_price"] > 0:
        lines.append(
            f"沪深300收益: {metrics['benchmark_return_pct']:+.2f}% "
            f"({metrics['benchmark_start_price']:.2f} → {metrics['benchmark_current_price']:.2f})"
        )
        excess = metrics["total_return_pct"] - metrics["benchmark_return_pct"]
        lines.append(f"超额收益: {excess:+.2f}%")

    closed_count = len(raw.get("closed_positions", []))
    if closed_count > 0:
        lines.append(f"\n历史已平仓: {closed_count} 笔")

    return "\n".join(lines)


TARGET_FILE = os.path.join(DATA_DIR, "target_portfolio.json")


def save_target_portfolio(target: dict):
    """保存目标组合配置（由组合策略节点生成，供再平衡参考）。

    target 格式: {code: {name, sector, target_weight, target_amount}, ...}
    """
    _ensure_dir()
    payload = {
        "updated_at": datetime.now().isoformat(),
        "targets": target,
    }
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[目标组合] 已保存 {len(target)} 只标的到 {TARGET_FILE}")


def load_target_portfolio() -> dict:
    """加载上次保存的目标组合配置。"""
    if os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        targets = payload.get("targets", {})
        updated = payload.get("updated_at", "未知")
        if targets:
            logger.info(f"[目标组合] 已加载 {len(targets)} 只标的 (更新于 {updated})")
        return targets
    return {}


def compute_position_deviations(
    current_positions: dict,
    target_portfolio: dict,
    total_value: float,
) -> list:
    """计算当前持仓与目标组合的偏离度。

    返回: [{code, name, current_weight, target_weight, deviation, action_hint}, ...]
    """
    deviations = []

    all_codes = set(list(current_positions.keys()) + list(target_portfolio.keys()))

    for code in sorted(all_codes):
        cur = current_positions.get(code, {})
        tgt = target_portfolio.get(code, {})

        cur_weight = cur.get("weight", 0) if cur else 0
        tgt_weight = tgt.get("target_weight", 0) if tgt else 0
        deviation = cur_weight - tgt_weight
        name = cur.get("stock_name", tgt.get("name", code))

        if abs(deviation) < 0.1:
            hint = "持仓达标"
        elif deviation > 0:
            hint = f"超配{deviation:.1f}%，考虑减持"
        else:
            hint = f"低配{abs(deviation):.1f}%，考虑增持"

        if code not in target_portfolio and code in current_positions:
            hint = "目标外持仓，考虑清仓"
        elif code in target_portfolio and code not in current_positions:
            hint = f"目标新增，考虑建仓{tgt_weight:.1f}%"

        deviations.append({
            "code": code,
            "name": name,
            "current_weight": round(cur_weight, 2),
            "target_weight": round(tgt_weight, 2),
            "deviation": round(deviation, 2),
            "action_hint": hint,
        })

    deviations.sort(key=lambda x: abs(x["deviation"]), reverse=True)
    return deviations


def format_deviation_summary(deviations: list, threshold: float = 5.0) -> str:
    """格式化偏离度摘要。只展示偏离超过阈值的和非正常状态的。"""
    if not deviations:
        return "无偏离度数据（无目标组合或无当前持仓）"

    lines = ["持仓偏离度分析:"]
    significant = [d for d in deviations if abs(d["deviation"]) >= threshold or "目标外" in d["action_hint"] or "目标新增" in d["action_hint"]]
    normal = [d for d in deviations if d not in significant]

    if significant:
        lines.append(f"\n需要关注的偏离（阈值 ≥{threshold}%）:")
        for d in significant:
            lines.append(
                f"  {d['name']}({d['code']}): "
                f"当前{d['current_weight']:.1f}% → 目标{d['target_weight']:.1f}%, "
                f"偏离{d['deviation']:+.1f}%, {d['action_hint']}"
            )
    else:
        lines.append("  所有持仓均在目标范围内，偏离度均低于阈值。")

    if normal:
        lines.append(f"\n正常范围内（{len(normal)}只）:")
        for d in normal:
            lines.append(f"  {d['name']}({d['code']}): {d['current_weight']:.1f}% → {d['target_weight']:.1f}%, 偏离{d['deviation']:+.1f}%")

    max_dev = max(abs(d["deviation"]) for d in deviations) if deviations else 0
    needs_rebalance = max_dev >= threshold or any("目标外" in d["action_hint"] for d in deviations)
    lines.append(f"\n最大偏离: {max_dev:.1f}%, 建议调仓: {'是' if needs_rebalance else '否'}")

    return "\n".join(lines)


def get_trade_history_summary(last_n: int = 20) -> str:
    """获取交易历史摘要文本。"""
    raw = load_raw()
    history = raw.get("trade_history", [])
    if not history:
        return "无历史交易记录"

    lines = [f"最近 {min(last_n, len(history))} 条交易记录:"]
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
    lines.append(f"\n统计: 买入{buy_count}次, 卖出{sell_count}次, 总已实现盈亏{total_pnl:+,.0f}元")
    return "\n".join(lines)


def get_closed_positions_summary(last_n: int = 20) -> str:
    """获取历史持仓摘要。"""
    raw = load_raw()
    closed = raw.get("closed_positions", [])
    if not closed:
        return "无历史持仓记录"

    lines = [f"历史持仓（最近 {min(last_n, len(closed))} 笔）:"]
    for cp in closed[-last_n:]:
        lines.append(
            f"  {cp['name']}({cp['code']}): "
            f"{cp['shares']}股, "
            f"成本{cp['avg_cost']:.2f} → 卖出{cp['sell_price']:.2f}, "
            f"盈亏{cp['realized_pnl_pct']:+.1f}%({cp['realized_pnl']:+,.0f}元), "
            f"{cp.get('buy_date', '')} → {cp.get('sell_date', '')}"
        )

    total_pnl = sum(cp.get("realized_pnl", 0) for cp in closed)
    win_count = sum(1 for cp in closed if cp.get("realized_pnl", 0) > 0)
    win_rate = win_count / len(closed) * 100 if closed else 0
    lines.append(f"\n合计: {len(closed)}笔, 胜率{win_rate:.0f}%, 总盈亏{total_pnl:+,.0f}元")

    return "\n".join(lines)
