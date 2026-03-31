"""技术与基本面分析工具

提供技术指标计算、行业相对估值评分、相对强度(RPS)、多因子选股等分析能力。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.tools.data_provider import (
    fetch_all_stocks,
    fetch_stock_history,
    fetch_stock_industry,
    SW_SECTOR_KEYWORDS,
)

logger = logging.getLogger("portfolio")


# ====================================================================
#  行业分类（优先 akshare API，回退关键词匹配）
# ====================================================================

_SECTOR_KEYWORDS = SW_SECTOR_KEYWORDS

# 申万二级/三级行业 → 申万一级行业映射
# 数据来源: Chinalin getStockDetail API 的 secondIndustryName 字段
_SW_SUB_TO_L1_MAP: dict[str, str] = {
    # -- 食品饮料 --
    "白酒Ⅱ": "食品饮料", "白酒": "食品饮料", "白酒Ⅲ": "食品饮料",
    "啤酒": "食品饮料", "啤酒Ⅱ": "食品饮料",
    "乳品": "食品饮料", "乳品Ⅱ": "食品饮料",
    "饮料制造": "食品饮料", "食品加工": "食品饮料", "食品加工Ⅱ": "食品饮料",
    "调味发酵品": "食品饮料", "调味发酵品Ⅱ": "食品饮料",
    "休闲食品": "食品饮料", "休闲食品Ⅱ": "食品饮料",
    "预加工食品": "食品饮料", "保健品": "食品饮料",
    # -- 电力设备 --
    "光伏设备": "电力设备", "风电设备": "电力设备", "电池": "电力设备",
    "电网设备": "电力设备", "储能设备": "电力设备", "充电桩": "电力设备",
    "电机Ⅱ": "电力设备", "电工仪器仪表": "电力设备",
    # -- 电子 --
    "半导体": "电子", "消费电子": "电子", "光学光电子": "电子",
    "元件": "电子", "电子化学品": "电子", "被动元件": "电子",
    "集成电路": "电子", "分立器件": "电子", "印制电路板": "电子",
    "LED": "电子", "面板": "电子", "品牌消费电子": "电子",
    # -- 计算机 --
    "软件开发": "计算机", "IT服务": "计算机", "IT服务Ⅱ": "计算机",
    "计算机设备": "计算机", "计算机设备Ⅱ": "计算机",
    "网络安全": "计算机", "云计算": "计算机",
    # -- 通信 --
    "通信设备": "通信", "通信服务": "通信",
    "通信设备Ⅱ": "通信", "通信服务Ⅱ": "通信",
    # -- 传媒 --
    "游戏": "传媒", "游戏Ⅱ": "传媒", "广告营销": "传媒",
    "影视院线": "传媒", "出版": "传媒", "数字媒体": "传媒",
    # -- 基础化工 --
    "化学制品": "基础化工", "化学原料": "基础化工", "化学纤维": "基础化工",
    "塑料": "基础化工", "橡胶": "基础化工", "农化制品": "基础化工",
    "膜材料": "基础化工", "涂料油墨": "基础化工", "日用化学品": "基础化工",
    "氟化工": "基础化工", "磷化工": "基础化工", "纯碱": "基础化工",
    "钛白粉": "基础化工", "粘胶": "基础化工", "碳纤维": "基础化工",
    # -- 钢铁 --
    "普钢": "钢铁", "特钢": "钢铁", "钢铁Ⅱ": "钢铁",
    # -- 有色金属 --
    "铜": "有色金属", "铝": "有色金属", "铅锌": "有色金属",
    "黄金": "有色金属", "稀土": "有色金属", "锂": "有色金属",
    "小金属": "有色金属", "贵金属": "有色金属", "工业金属": "有色金属",
    "钨": "有色金属", "钴": "有色金属", "镍": "有色金属",
    "钼": "有色金属", "锡": "有色金属", "锑": "有色金属",
    "能源金属": "有色金属", "金属新材料": "有色金属",
    # -- 煤炭 --
    "煤炭开采": "煤炭", "焦炭Ⅱ": "煤炭", "焦炭": "煤炭", "煤炭Ⅱ": "煤炭",
    "冶金煤": "煤炭", "动力煤": "煤炭",
    # -- 石油石化 --
    "石油开采": "石油石化", "油服工程": "石油石化", "炼化及贸易": "石油石化",
    "油气开采Ⅱ": "石油石化", "油气开采Ⅲ": "石油石化",
    # -- 建筑材料 --
    "水泥": "建筑材料", "玻璃玻纤": "建筑材料", "装修建材": "建筑材料",
    "水泥Ⅱ": "建筑材料", "玻璃": "建筑材料", "玻纤": "建筑材料",
    "耐火材料": "建筑材料", "管材": "建筑材料",
    # -- 建筑装饰 --
    "房屋建设": "建筑装饰", "基础建设": "建筑装饰", "装修装饰": "建筑装饰",
    "专业工程": "建筑装饰", "工程咨询服务": "建筑装饰",
    "房屋建设Ⅱ": "建筑装饰", "基础建设Ⅱ": "建筑装饰",
    # -- 国防军工 --
    "航空装备": "国防军工", "地面兵装": "国防军工", "航海装备": "国防军工",
    "航天装备": "国防军工", "军工电子": "国防军工",
    "航空装备Ⅱ": "国防军工", "地面兵装Ⅱ": "国防军工", "地面兵装Ⅲ": "国防军工",
    "航海装备Ⅱ": "国防军工", "航天装备Ⅱ": "国防军工",
    # -- 汽车 --
    "汽车整车": "汽车", "汽车零部件": "汽车", "汽车服务": "汽车",
    "乘用车": "汽车", "商用车": "汽车", "摩托车": "汽车",
    "乘用车Ⅱ": "汽车", "商用车Ⅱ": "汽车",
    # -- 机械设备 --
    "通用设备": "机械设备", "专用设备": "机械设备", "工程机械": "机械设备",
    "自动化设备": "机械设备", "仪器仪表": "机械设备",
    "轨交设备": "机械设备", "农机": "机械设备",
    # -- 交通运输 --
    "航空机场": "交通运输", "航运港口": "交通运输", "物流": "交通运输",
    "铁路公路": "交通运输", "快递": "交通运输",
    "航空": "交通运输", "机场": "交通运输", "航运": "交通运输",
    "港口": "交通运输", "高速公路": "交通运输", "铁路": "交通运输",
    # -- 医药生物 --
    "化学制药": "医药生物", "中药": "医药生物", "中药Ⅱ": "医药生物",
    "生物制品": "医药生物", "生物制品Ⅱ": "医药生物",
    "医疗器械": "医药生物", "医疗服务": "医药生物", "医药商业": "医药生物",
    "原料药": "医药生物", "化学制剂": "医药生物", "疫苗": "医药生物",
    "血液制品": "医药生物", "CXO": "医药生物", "线下药店": "医药生物",
    # -- 房地产 --
    "房地产开发": "房地产", "房地产服务": "房地产",
    "房地产开发Ⅱ": "房地产", "房地产服务Ⅱ": "房地产", "物业管理": "房地产",
    # -- 银行 --
    "银行": "银行", "银行Ⅱ": "银行",
    "国有大型银行Ⅱ": "银行", "股份制银行Ⅱ": "银行",
    "城商行Ⅱ": "银行", "农商行Ⅱ": "银行",
    # -- 非银金融 --
    "证券": "非银金融", "保险": "非银金融", "多元金融": "非银金融",
    "证券Ⅱ": "非银金融", "保险Ⅱ": "非银金融", "多元金融Ⅱ": "非银金融",
    "信托": "非银金融", "期货": "非银金融",
    # -- 家用电器 --
    "白色家电": "家用电器", "黑色家电": "家用电器", "小家电": "家用电器",
    "厨卫电器": "家用电器", "照明设备": "家用电器",
    "白色家电Ⅱ": "家用电器", "白色家电Ⅲ": "家用电器",
    # -- 商贸零售 --
    "商业百货": "商贸零售", "专业连锁": "商贸零售", "互联网电商": "商贸零售",
    "超市": "商贸零售", "一般零售": "商贸零售",
    # -- 社会服务 --
    "酒店餐饮": "社会服务", "旅游及景区": "社会服务", "教育": "社会服务",
    "专业服务": "社会服务", "体育": "社会服务",
    # -- 纺织服饰 --
    "纺织制造": "纺织服饰", "服装家纺": "纺织服饰", "饰品": "纺织服饰",
    "品牌服饰": "纺织服饰", "鞋帽": "纺织服饰",
    # -- 美容护理 --
    "个护用品": "美容护理", "化妆品": "美容护理", "医美": "美容护理",
    # -- 轻工制造 --
    "造纸": "轻工制造", "包装印刷": "轻工制造", "家具用品": "轻工制造",
    "文娱用品": "轻工制造", "家居用品": "轻工制造",
    # -- 农林牧渔 --
    "种植业": "农林牧渔", "养殖业": "农林牧渔", "饲料": "农林牧渔",
    "动物保健": "农林牧渔", "林业": "农林牧渔",
    "种子": "农林牧渔", "养鸡": "农林牧渔", "养猪": "农林牧渔",
    "生猪养殖": "农林牧渔", "渔业": "农林牧渔",
    # -- 公用事业 --
    "电力": "公用事业", "燃气": "公用事业", "水务": "公用事业",
    "环保": "公用事业", "火电": "公用事业", "水电": "公用事业",
    "核电": "公用事业", "热力": "公用事业", "新能源发电": "公用事业",
    # -- 综合 --
    "综合": "综合", "综合Ⅱ": "综合",
}


def _classify_sector(name: str, stock_code: str = "") -> str:
    """行业分类：优先 Chinalin API 精确分类，回退关键词匹配。

    Args:
        name: 股票名称
        stock_code: 股票代码（可选，提供后调用 Chinalin 内部 API 获取申万行业）
    """
    if stock_code:
        sw_sub = fetch_stock_industry(stock_code)
        if sw_sub:
            if sw_sub in _SW_SUB_TO_L1_MAP:
                return _SW_SUB_TO_L1_MAP[sw_sub]
            for sw_name in SW_SECTOR_KEYWORDS:
                if sw_name in sw_sub:
                    return sw_name
            return sw_sub

    for sector, keywords in _SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return sector
    return "其他"


def _get_industry_stats() -> dict[str, dict]:
    """计算各行业的 PE/PB 分位数统计，用于行业相对估值。

    Returns:
        {行业: {pe_median, pe_25, pe_75, pb_median, pb_25, pb_75, count}}
    """
    df = fetch_all_stocks()
    if df.empty:
        return {}

    df = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] < 500) & (df["市净率"] > 0)].copy()
    df["行业"] = df["名称"].apply(_classify_sector)

    stats = {}
    for sector, group in df.groupby("行业"):
        if len(group) < 5:
            continue
        stats[sector] = {
            "pe_median": float(group["市盈率-动态"].median()),
            "pe_25": float(group["市盈率-动态"].quantile(0.25)),
            "pe_75": float(group["市盈率-动态"].quantile(0.75)),
            "pb_median": float(group["市净率"].median()),
            "pb_25": float(group["市净率"].quantile(0.25)),
            "pb_75": float(group["市净率"].quantile(0.75)),
            "count": len(group),
        }
    return stats


# ====================================================================
#  技术指标
# ====================================================================

@tool
def calculate_technical_indicators(stock_code: str, days: int = 120) -> str:
    """计算股票技术指标（MA、MACD、RSI、布林带等）。

    Args:
        stock_code: 股票代码
        days: 计算周期天数
    """
    try:
        df = fetch_stock_history(stock_code, days=days + 60)
        if df.empty or len(df) < 30:
            return f"{stock_code}: 数据不足，无法计算技术指标"

        close = df["收盘"].values
        high = df["最高"].values
        low = df["最低"].values
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

        current = close[-1]
        ma_trend = "多头排列" if ma5 > ma10 > ma20 else ("空头排列" if ma5 < ma10 < ma20 else "震荡")

        return (
            f"{stock_code} 技术指标:\n"
            f"  当前价: {current:.2f}\n"
            f"  MA5={ma5:.2f}, MA10={ma10:.2f}, MA20={ma20:.2f}"
            f"{f', MA60={ma60:.2f}' if ma60 else ''}\n"
            f"  均线趋势: {ma_trend}\n"
            f"  MACD: DIF={dif.iloc[-1]:.3f}, DEA={dea.iloc[-1]:.3f}, MACD柱={macd_hist.iloc[-1]:.3f}\n"
            f"  RSI(14): {rsi.iloc[-1]:.1f} ({'超买' if rsi.iloc[-1] > 70 else '超卖' if rsi.iloc[-1] < 30 else '中性'})\n"
            f"  布林带: 上轨={bb_upper.iloc[-1]:.2f}, 中轨={bb_mid.iloc[-1]:.2f}, 下轨={bb_lower.iloc[-1]:.2f}\n"
            f"  量比: {vol_ratio:.2f} ({'放量' if vol_ratio > 1.5 else '缩量' if vol_ratio < 0.7 else '正常'})"
        )
    except Exception as e:
        return f"计算 {stock_code} 技术指标失败: {e}"


# ====================================================================
#  基本面评分（行业相对估值）
# ====================================================================

@tool
def calculate_stock_score(
    stock_code: str,
    pe_ratio: float,
    pb_ratio: float,
    roe: float,
    revenue_growth: float,
    profit_growth: float,
    debt_ratio: float,
    dividend_yield: float = 0,
    sector: str = "",
) -> str:
    """基于基本面指标计算股票综合评分（百分制），支持行业相对估值。

    与绝对阈值打分不同，估值维度会参考同行业 PE/PB 分位数：
    - PE 低于行业 25 分位 → 估值便宜
    - PE 高于行业 75 分位 → 估值偏贵
    这避免了「银行 PE=5 永远高分、科技股 PE=40 永远低分」的问题。

    Args:
        stock_code: 股票代码
        pe_ratio: 市盈率
        pb_ratio: 市净率
        roe: 净资产收益率(%)
        revenue_growth: 营收增长率(%)
        profit_growth: 净利润增长率(%)
        debt_ratio: 资产负债率(%)
        dividend_yield: 股息率(%)
        sector: 所属行业（用于行业相对估值，留空则使用绝对阈值）
    """
    score = 50.0
    valuation_detail = ""

    industry_stats = None
    if sector:
        all_stats = _get_industry_stats()
        industry_stats = all_stats.get(sector)

    # --- 估值维度 (满分25) ---
    if industry_stats and pe_ratio > 0:
        pe_med = industry_stats["pe_median"]
        pe_25 = industry_stats["pe_25"]
        pe_75 = industry_stats["pe_75"]
        if pe_ratio < pe_25:
            score += 20
            valuation_detail = f"PE={pe_ratio:.1f} < 行业25分位{pe_25:.1f}(便宜)"
        elif pe_ratio < pe_med:
            score += 15
            valuation_detail = f"PE={pe_ratio:.1f} < 行业中位{pe_med:.1f}(合理偏低)"
        elif pe_ratio < pe_75:
            score += 8
            valuation_detail = f"PE={pe_ratio:.1f} < 行业75分位{pe_75:.1f}(合理)"
        else:
            score -= 3
            valuation_detail = f"PE={pe_ratio:.1f} > 行业75分位{pe_75:.1f}(偏贵)"

        pb_med = industry_stats["pb_median"]
        if pb_ratio > 0:
            if pb_ratio < industry_stats["pb_25"]:
                score += 5
            elif pb_ratio < pb_med:
                score += 3
            elif pb_ratio > industry_stats["pb_75"]:
                score -= 3
    else:
        if pe_ratio > 0:
            if pe_ratio < 15:
                score += 20
            elif pe_ratio < 25:
                score += 15
            elif pe_ratio < 40:
                score += 8
            else:
                score -= 5
            valuation_detail = f"PE={pe_ratio:.1f}(绝对估值)"

        if pb_ratio > 0:
            if pb_ratio < 1.5:
                score += 5
            elif pb_ratio < 3:
                score += 3
            elif pb_ratio > 8:
                score -= 5

    # --- 盈利质量 (满分20) ---
    if roe > 20:
        score += 20
    elif roe > 15:
        score += 15
    elif roe > 10:
        score += 10
    elif roe > 5:
        score += 5

    if dividend_yield > 3:
        score += 5
    elif dividend_yield > 1.5:
        score += 3

    # --- 成长性 (满分25) ---
    if revenue_growth > 30:
        score += 12
    elif revenue_growth > 15:
        score += 8
    elif revenue_growth > 5:
        score += 4
    elif revenue_growth < -10:
        score -= 5

    if profit_growth > 30:
        score += 13
    elif profit_growth > 15:
        score += 8
    elif profit_growth > 0:
        score += 4
    elif profit_growth < -20:
        score -= 8

    # --- 财务健康 (满分10) ---
    if debt_ratio < 40:
        score += 5
    elif debt_ratio > 70:
        score -= 10
    elif debt_ratio > 60:
        score -= 5

    score = max(0, min(100, score))

    sector_info = f"  行业: {sector} ({industry_stats['count']}只同业)" if industry_stats else "  行业: 未指定(使用绝对估值)"

    return (
        f"{stock_code} 综合评分: {score:.0f}/100\n"
        f"{sector_info}\n"
        f"  估值: {valuation_detail}, PB={pb_ratio}\n"
        f"  盈利: ROE={roe}%, 股息率={dividend_yield}%\n"
        f"  成长: 营收增长={revenue_growth}%, 利润增长={profit_growth}%\n"
        f"  健康: 资产负债率={debt_ratio}%\n"
        f"  评级: {'优秀' if score >= 80 else '良好' if score >= 65 else '一般' if score >= 50 else '较差'}"
    )


# ====================================================================
#  相对强度 (RPS)
# ====================================================================

@tool
def calculate_relative_strength(stock_codes: list[str], period_days: int = 20) -> str:
    """计算股票相对强度排名（RPS），衡量个股跑赢/跑输大盘的程度。

    RPS = 该股票在全市场中的涨幅百分位排名（0-100）。
    RPS > 80 表示强于80%的股票（强势），RPS < 20 表示弱于80%的股票（弱势）。
    专业选股时通常要求 RPS20 > 80 且 RPS60 > 70（动量选股）。

    Args:
        stock_codes: 要查询的股票代码列表
        period_days: RPS 计算周期（20日/60日）
    """
    try:
        df = fetch_all_stocks()
        if df.empty:
            return "无法获取全市场数据"

        target_quotes = {code: None for code in stock_codes}
        for code in stock_codes[:20]:
            try:
                hist = fetch_stock_history(code, days=period_days + 10)
                if not hist.empty and len(hist) > period_days:
                    start_price = hist["收盘"].iloc[-period_days - 1]
                    end_price = hist["收盘"].iloc[-1]
                    target_quotes[code] = (end_price / start_price - 1) * 100
            except Exception:
                continue

        all_changes = df["涨跌幅"].dropna().values
        n_total = len(all_changes)

        results = []
        for code in stock_codes[:20]:
            change = target_quotes.get(code)
            if change is None:
                results.append(f"  {code}: 数据不足")
                continue

            rps = float(np.searchsorted(np.sort(all_changes), change) / n_total * 100)
            strength = "极强" if rps > 90 else ("强势" if rps > 80 else ("中等" if rps > 50 else ("偏弱" if rps > 20 else "极弱")))
            results.append(
                f"  {code}: {period_days}日涨幅{change:+.2f}%, RPS={rps:.1f} ({strength})"
            )

        return (
            f"相对强度排名 (RPS{period_days}, 全市场{n_total}只股票):\n"
            + "\n".join(results)
            + f"\n\n  参考: RPS>80=强势(适合趋势跟踪), RPS<20=弱势(可能反转机会)"
        )
    except Exception as e:
        return f"RPS计算失败: {e}"


# ====================================================================
#  多因子选股评分
# ====================================================================

@tool
def calculate_multi_factor_score(stock_codes: list[str]) -> str:
    """多因子综合评分：基于价值、动量、质量、流动性四大因子对股票打分排名。

    因子定义：
    - 价值因子(25%): PE/PB 在行业内的百分位排名（越低越好）
    - 动量因子(25%): 近20日涨幅的全市场排名（越高越好）
    - 质量因子(25%): ROE 代理（PE>0时用1/PE近似，越高越好）
    - 流动性因子(25%): 换手率适中性（过高过低都扣分）

    每个因子得分 0-100，综合加权后排序。

    Args:
        stock_codes: 股票代码列表（最多20只）
    """
    try:
        df = fetch_all_stocks()
        if df.empty:
            return "无法获取全市场数据"

        valid = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] < 500) & (df["市净率"] > 0)].copy()
        valid["行业"] = valid["名称"].apply(_classify_sector)

        pe_ranks = valid.groupby("行业")["市盈率-动态"].rank(pct=True)
        pb_ranks = valid.groupby("行业")["市净率"].rank(pct=True)
        valid["pe_rank"] = pe_ranks
        valid["pb_rank"] = pb_ranks
        valid["value_score"] = ((1 - valid["pe_rank"]) * 60 + (1 - valid["pb_rank"]) * 40)

        valid["momentum_score"] = valid["涨跌幅"].rank(pct=True) * 100

        valid["quality_score"] = 50.0
        mask_pe = valid["市盈率-动态"] > 0
        valid.loc[mask_pe, "quality_score"] = (1 / valid.loc[mask_pe, "市盈率-动态"]).rank(pct=True) * 100

        turn = valid["换手率"]
        turn_med = turn.median()
        valid["liquidity_score"] = 100 - (((turn - turn_med).abs() / turn_med).clip(0, 2) * 50)

        valid["composite"] = (
            valid["value_score"] * 0.25
            + valid["momentum_score"] * 0.25
            + valid["quality_score"] * 0.25
            + valid["liquidity_score"] * 0.25
        )

        results = []
        for code in stock_codes[:20]:
            row = valid[valid["代码"] == code]
            if row.empty:
                results.append(f"  {code}: 未找到或数据不足")
                continue
            r = row.iloc[0]
            sector = r["行业"]
            composite = r["composite"]
            rank_in_all = float((valid["composite"] <= composite).sum() / len(valid) * 100)

            results.append(
                f"  {r['名称']}({code}) [{sector}]:\n"
                f"    综合: {composite:.1f}分 (全市场前{100 - rank_in_all:.1f}%)\n"
                f"    价值={r['value_score']:.0f} 动量={r['momentum_score']:.0f} "
                f"质量={r['quality_score']:.0f} 流动性={r['liquidity_score']:.0f}\n"
                f"    PE={r['市盈率-动态']:.1f}(行业{r['pe_rank']*100:.0f}%分位) "
                f"PB={r['市净率']:.1f} 涨跌幅={r['涨跌幅']:.2f}% 换手率={r['换手率']:.2f}%"
            )

        return (
            f"多因子评分({len(stock_codes)}只, 权重: 价值25%+动量25%+质量25%+流动性25%):\n"
            + "\n".join(results)
        )
    except Exception as e:
        return f"多因子评分失败: {e}"


# ====================================================================
#  量价分析 & 横向对比
# ====================================================================

@tool
def analyze_price_volume_pattern(stock_code: str) -> str:
    """分析股票的量价关系和形态特征。

    Args:
        stock_code: 股票代码
    """
    try:
        df = fetch_stock_history(stock_code, days=180)
        if df.empty or len(df) < 20:
            return f"{stock_code}: 数据不足"

        close = df["收盘"]
        volume = df["成交量"]

        pct_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
        pct_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0
        pct_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 60 else 0

        returns = close.pct_change().dropna()
        vol_20d = returns.tail(20).std() * np.sqrt(252) * 100

        recent_high = df["最高"].tail(20).max()
        recent_low = df["最低"].tail(20).min()
        current = close.iloc[-1]

        vol_5 = volume.tail(5).mean()
        vol_20 = volume.tail(20).mean()
        vol_trend = "放量" if vol_5 > vol_20 * 1.3 else ("缩量" if vol_5 < vol_20 * 0.7 else "平稳")

        return (
            f"{stock_code} 量价分析:\n"
            f"  当前价: {current:.2f}\n"
            f"  近5日涨跌: {pct_5d:+.2f}%\n"
            f"  近20日涨跌: {pct_20d:+.2f}%\n"
            f"  近60日涨跌: {pct_60d:+.2f}%\n"
            f"  20日波动率(年化): {vol_20d:.1f}%\n"
            f"  20日高点: {recent_high:.2f}, 低点: {recent_low:.2f}\n"
            f"  量能趋势: {vol_trend} (5日均量/20日均量={vol_5 / vol_20:.2f})\n"
            f"  位置: 距高点{(current / recent_high - 1) * 100:.1f}%, 距低点{(current / recent_low - 1) * 100:.1f}%"
        )
    except Exception as e:
        return f"分析 {stock_code} 失败: {e}"


@tool
def compare_stocks(stock_codes: list[str]) -> str:
    """横向对比多只股票的关键指标。

    Args:
        stock_codes: 股票代码列表（最多10只）
    """
    try:
        from src.tools.data_provider import fetch_stock_quote
        quotes = fetch_stock_quote(stock_codes[:10])
        if not quotes:
            return "未找到任何股票数据"

        results = []
        for code in stock_codes[:10]:
            q = quotes.get(code)
            if not q:
                continue
            results.append({
                "代码": code,
                "名称": q["name"],
                "现价": q["price"],
                "涨跌幅%": q["change_pct"],
                "成交额(亿)": round(q["amount"] / 1e8, 2),
                "换手率%": q["turnover"],
                "PE": q["pe"] or "N/A",
                "PB": q["pb"] or "N/A",
            })
        if not results:
            return "未找到任何股票数据"
        return f"股票对比:\n{pd.DataFrame(results).to_string(index=False)}"
    except Exception as e:
        return f"股票对比失败: {e}"


@tool
def batch_technical_indicators(stock_codes: list[str], days: int = 120) -> str:
    """批量计算多只股票的技术指标（MA、MACD、RSI、布林带、量比），一次最多20只。

    相比 calculate_technical_indicators 逐只调用，此工具可一轮获取全部持仓的技术面数据，
    大幅节省迭代轮次。持仓审查时务必使用此工具。

    Args:
        stock_codes: 股票代码列表（最多20只）
        days: 计算周期天数
    """
    results = []
    for code in stock_codes[:20]:
        try:
            df = fetch_stock_history(code, days=days + 60)
            if df.empty or len(df) < 30:
                results.append(f"{code}: 数据不足")
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
            ma_trend = "多头排列" if ma5 > ma10 > ma20 else ("空头排列" if ma5 < ma10 < ma20 else "震荡")

            support = bb_lower.iloc[-1]
            resistance = bb_upper.iloc[-1]
            dist_support = (current / support - 1) * 100 if support > 0 else 0
            dist_resistance = (current / resistance - 1) * 100 if resistance > 0 else 0

            results.append(
                f"【{code}】价={current:.2f} | "
                f"MA趋势={ma_trend} "
                f"MA5={ma5:.2f}/MA10={ma10:.2f}/MA20={ma20:.2f}"
                f"{f'/MA60={ma60:.2f}' if ma60 else ''} | "
                f"MACD柱={macd_hist.iloc[-1]:.3f} DIF={dif.iloc[-1]:.3f} | "
                f"RSI={rsi.iloc[-1]:.1f} | "
                f"布林={bb_lower.iloc[-1]:.2f}~{bb_upper.iloc[-1]:.2f} "
                f"距支撑{dist_support:+.1f}%/距压力{dist_resistance:+.1f}% | "
                f"量比={vol_ratio:.2f}({'放量' if vol_ratio > 1.5 else '缩量' if vol_ratio < 0.7 else '正常'}) | "
                f"5日{pct_5d:+.1f}% 20日{pct_20d:+.1f}%"
            )
        except Exception as e:
            results.append(f"{code}: 计算失败({e})")

    return f"批量技术指标({len(stock_codes)}只):\n" + "\n".join(results)


# ====================================================================
#  工具导出
# ====================================================================

@tool
def detect_market_regime(index_code: str = "000300", days: int = 120) -> str:
    """检测当前市场状态（趋势+波动率regime），为投资风格切换提供依据。

    通过均线关系判定趋势、波动率百分位判定波动环境、RSI和成交额判定情绪，
    综合给出 regime 标签和建议的风格倾向。

    Args:
        index_code: 基准指数代码（默认沪深300）
        days: 分析数据天数
    """
    try:
        from src.tools.data_provider import fetch_index_history
        df = fetch_index_history(index_code, days=days + 60)
        if df.empty or len(df) < 60:
            return f"数据不足（{len(df)}天），无法判定市场状态"

        close = df["收盘"].values
        volume = df.get("成交额", df.get("成交量", pd.Series([0] * len(df)))).values

        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values

        c, m20, m60 = close[-1], ma20[-1], ma60[-1]
        if np.isnan(m60):
            return "数据不足60天，无法计算MA60"

        if c > m20 > m60:
            trend = "上升趋势"
        elif c < m20 < m60:
            trend = "下降趋势"
        elif c > m60:
            trend = "震荡偏强"
        else:
            trend = "震荡偏弱"

        ret = pd.Series(close).pct_change().dropna()
        vol_20 = float(ret.iloc[-20:].std() * np.sqrt(252))
        vol_60 = float(ret.iloc[-60:].std() * np.sqrt(252))
        vol_full = float(ret.std() * np.sqrt(252))

        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
        vol_pctile = float((ret.rolling(20).std().rank(pct=True)).iloc[-1])

        if vol_pctile > 0.8:
            vol_regime = "高波动"
        elif vol_pctile < 0.3:
            vol_regime = "低波动"
        else:
            vol_regime = "中等波动"

        # RSI(14)
        delta = ret.values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().iloc[-1]
        avg_loss = pd.Series(loss).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

        vol_recent = volume[-20:]
        vol_avg = volume[-60:].mean()
        vol_trend = vol_recent.mean() / vol_avg if vol_avg > 0 else 1.0

        if rsi > 70 and vol_trend > 1.3:
            sentiment = "过热"
        elif rsi < 30 and vol_trend < 0.7:
            sentiment = "恐慌"
        elif rsi > 60:
            sentiment = "偏乐观"
        elif rsi < 40:
            sentiment = "偏悲观"
        else:
            sentiment = "中性"

        # regime → style 建议
        regime_label = f"{trend} + {vol_regime} + {sentiment}"
        if "上升" in trend and "低波动" in vol_regime:
            style_hint = "aggressive（趋势向上+波动率低，适合进攻）"
        elif "下降" in trend or "恐慌" in sentiment:
            style_hint = "conservative（趋势向下或市场恐慌，防御优先）"
        elif "高波动" in vol_regime:
            style_hint = "balanced偏保守（高波动环境，控制仓位和回撤）"
        else:
            style_hint = "balanced（震荡环境，攻守兼备）"

        return (
            f"市场状态检测 ({index_code}):\n"
            f"  Regime: {regime_label}\n\n"
            f"  趋势: {trend} (现价{c:.2f}, MA20={m20:.2f}, MA60={m60:.2f})\n"
            f"  波动率: {vol_regime}\n"
            f"    近20日年化波动率: {vol_20 * 100:.1f}%\n"
            f"    近60日年化波动率: {vol_60 * 100:.1f}%\n"
            f"    波动率百分位: {vol_pctile * 100:.0f}%\n"
            f"    波动率变化比: {vol_ratio:.2f} ({'波动率扩张' if vol_ratio > 1.2 else '波动率收缩' if vol_ratio < 0.8 else '稳定'})\n"
            f"  情绪: {sentiment} (RSI={rsi:.1f}, 量能比={vol_trend:.2f})\n\n"
            f"  风格建议: {style_hint}"
        )
    except Exception as e:
        return f"市场状态检测失败: {e}"


@tool
def check_event_calendar(stock_codes: list[str]) -> str:
    """检查股票的近期重要事件（财报发布、除权除息、解禁等），避免在关键事件前后做出错误交易决策。
    多源回退: Baostock → akshare。

    Args:
        stock_codes: 股票代码列表（最多20只）
    """
    from src.tools.data_provider import fetch_dividend_data

    results = []
    for code in stock_codes[:20]:
        events = []
        try:
            div_df = fetch_dividend_data(code, years=2)
            if div_df is not None and not div_df.empty:
                for _, row in div_df.head(3).iterrows():
                    ex_date = str(row.get("除权除息日", ""))
                    if ex_date and ex_date != "nan":
                        plan = row.get("分红方案", "")
                        label = f"除权除息: {ex_date}"
                        if plan:
                            label += f" ({plan})"
                        events.append(label)
        except Exception:
            pass

        if events:
            results.append(f"  {code}: " + "; ".join(events))
        else:
            results.append(f"  {code}: 未查询到近期事件")

    if not results:
        return "无法获取事件日历数据"

    return "近期重要事件日历:\n" + "\n".join(results) + (
        "\n\n⚠ 建议: 财报发布前后3天避免大额调仓，除权除息日前考虑是否参与分红"
    )


ANALYSIS_TOOLS = [
    calculate_technical_indicators,
    batch_technical_indicators,
    calculate_stock_score,
    analyze_price_volume_pattern,
    compare_stocks,
    calculate_relative_strength,
    calculate_multi_factor_score,
    detect_market_regime,
    check_event_calendar,
]
