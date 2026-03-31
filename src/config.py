"""配置管理模块

从 YAML 文件加载配置，支持环境变量替换和默认值。
提供统一的配置访问接口和共享模型工厂。
"""

import os
import re
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    research: str = "openai:ep-20251219111602-7nqgk"
    research_max_tokens: int = 16000
    research_temperature: float = 0.4
    decision: str = "openai:ep-20251219111602-7nqgk"
    decision_max_tokens: int = 16000
    decision_temperature: float = 0.1
    compression: str = "openai:ep-20251219111602-7nqgk"
    compression_max_tokens: int = 10000
    compression_temperature: float = 0
    report: str = "openai:ep-20251219111602-7nqgk"
    report_max_tokens: int = 32000
    report_temperature: float = 0.25
    base_url: str = "http://ark.cn-beijing.volces.com/api/v3"
    temperature: float = 0.2


@dataclass
class NodeSwitches:
    """节点开关"""
    enable_market_data: bool = True
    enable_market_analysis: bool = True
    enable_stock_screening: bool = True
    enable_portfolio_strategy: bool = True
    enable_risk_assessment: bool = True
    enable_trade_execution: bool = True
    enable_rebalance_monitor: bool = False
    enable_analysis_compression: bool = False
    enable_strategy_compression: bool = False


@dataclass
class ReactConfig:
    """ReAct 参数"""
    max_iterations: int = 8
    screening_max_iterations: int = 12
    strategy_max_iterations: int = 10
    trade_max_iterations: int = 20


@dataclass
class PortfolioConfig:
    """组合策略配置"""
    initial_capital: float = 1_000_000
    max_positions: int = 10
    max_single_weight: float = 0.15
    max_sector_weight: float = 0.35
    min_trade_amount: float = 5000
    style: str = "balanced"
    rebalance_threshold: float = 5.0
    benchmark: str = "000300"
    min_cash_ratio: float = 0.05
    max_correlation: float = 0.85
    max_turnover_rate: float = 0.80
    max_sector_stocks: int = 3
    min_rebalance_deviation: float = 2.0


@dataclass
class RiskConfig:
    """风险控制配置"""
    stop_loss_pct: float = -8.0
    take_profit_pct: float = 30.0
    max_drawdown_pct: float = -15.0
    var_limit_pct: float = 2.0
    volatility_target: float = 15.0
    max_leverage: float = 1.0
    min_daily_turnover: float = 1000
    exclude_st: bool = True
    require_profit: bool = True
    min_market_cap_billion: float = 50
    trailing_stop_pct: float = 8.0
    trailing_stop_min_hold_days: int = 5


@dataclass
class TradingConfig:
    """交易配置"""
    mode: str = "simulation"
    broker_api: str = ""
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.0005
    slippage_pct: float = 0.1


@dataclass
class ScheduleConfig:
    """调度配置"""
    market_open: str = "09:30"
    market_close: str = "15:00"
    analysis_time: str = "08:30"
    summary_time: str = "15:30"
    rebalance_frequency: str = "weekly"


@dataclass
class SearchConfig:
    """搜索工具配置

    通过 engine 字段切换搜索引擎：
      - "chinalin": 使用 Chinalin API（默认）
      - "tavily":   使用 Tavily Search API

    搜索结果由 LLM 生成结构化摘要（enable_summarization 控制开关）。
    """
    engine: str = "chinalin"
    max_results: int = 5
    timeout: int = 30
    # Chinalin 引擎配置
    chinalin_base_url: str = "http://api-test.chinalions.cn"
    chinalin_api_key: str = ""
    # Tavily 引擎配置
    tavily_api_key: str = ""
    tavily_topic: str = "general"
    tavily_include_raw_content: bool = False
    # 搜索结果 LLM 摘要配置
    enable_summarization: bool = True
    summarization_model: str = ""
    summarization_model_base_url: str = ""
    summarization_model_max_tokens: int = 8192
    max_content_length: int = 50000
    max_structured_output_retries: int = 1


@dataclass
class RebalanceTriggersConfig:
    """调仓触发条件"""
    pnl_threshold: float = 15.0
    deviation_threshold: float = 8.0
    max_hold_days: int = 30
    market_shock_pct: float = 3.0
    min_cash_alert: float = 0.03


@dataclass
class DataSourceConfig:
    """数据源配置（ChinaLin 内部行情 API）"""
    chinalin_base_url: str = "https://chinalintest.wenxingonline.com"
    http_timeout: int = 10
    max_response_chars: int = 50000


@dataclass
class LogDetailConfig:
    """日志配置"""
    detail_level: str = "normal"
    file_mode: str = "a"


@dataclass
class AppConfig:
    """系统总配置"""
    models: ModelConfig = field(default_factory=ModelConfig)
    nodes: NodeSwitches = field(default_factory=NodeSwitches)
    react: ReactConfig = field(default_factory=ReactConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    rebalance_triggers: RebalanceTriggersConfig = field(default_factory=RebalanceTriggersConfig)
    logging: LogDetailConfig = field(default_factory=LogDetailConfig)
    api_keys: Dict[str, str] = field(default_factory=dict)
    get_api_keys_from_config: bool = True


def _resolve_env_vars(value: str) -> str:
    """替换 ${VAR_NAME} 格式的环境变量"""
    if not isinstance(value, str):
        return value
    pattern = re.compile(r'\$\{(\w+)\}')
    def replacer(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    return pattern.sub(replacer, value)


def _resolve_dict_env_vars(d: dict) -> dict:
    """递归替换字典中所有环境变量"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _resolve_dict_env_vars(v)
        elif isinstance(v, str):
            result[k] = _resolve_env_vars(v)
        elif isinstance(v, list):
            result[k] = [_resolve_env_vars(i) if isinstance(i, str) else i for i in v]
        else:
            result[k] = v
    return result


def _load_section(raw: dict, key: str, cls, defaults=None):
    """通用加载逻辑：从 raw dict 构建 dataclass"""
    if key not in raw:
        return defaults or cls()
    section = raw[key]
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in section.items() if k in valid_fields}
    return cls(**filtered)


def load_config(config_path: str = "portfolio_config.yaml") -> AppConfig:
    """从 YAML 文件加载配置"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return AppConfig()

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f) or {}

    raw_config = _resolve_dict_env_vars(raw_config)

    config = AppConfig()
    config.models = _load_section(raw_config, 'models', ModelConfig, config.models)
    config.nodes = _load_section(raw_config, 'nodes', NodeSwitches, config.nodes)
    config.react = _load_section(raw_config, 'react', ReactConfig, config.react)
    config.portfolio = _load_section(raw_config, 'portfolio', PortfolioConfig, config.portfolio)
    config.risk = _load_section(raw_config, 'risk', RiskConfig, config.risk)
    config.trading = _load_section(raw_config, 'trading', TradingConfig, config.trading)
    config.schedule = _load_section(raw_config, 'schedule', ScheduleConfig, config.schedule)
    config.search = _load_section(raw_config, 'search', SearchConfig, config.search)
    config.data_source = _load_section(raw_config, 'data_source', DataSourceConfig, config.data_source)
    config.rebalance_triggers = _load_section(raw_config, 'rebalance_triggers', RebalanceTriggersConfig, config.rebalance_triggers)
    config.logging = _load_section(raw_config, 'logging', LogDetailConfig, config.logging)
    config.api_keys = raw_config.get('api_keys', {})
    config.get_api_keys_from_config = raw_config.get('get_api_keys_from_config', True)

    logger.info(f"配置加载完成 | 模型: {config.models.research} | 策略风格: {config.portfolio.style}")
    return config


# ============================================================
# 统一配置访问接口
# ============================================================

def get_app_config(config: RunnableConfig) -> AppConfig:
    """从 RunnableConfig 中安全提取 AppConfig"""
    configurable = config.get("configurable", {})
    return configurable.get("app_config", AppConfig())


# ============================================================
# 共享模型工厂
# ============================================================

_configurable_model: Optional[Any] = None


def _get_configurable_model() -> Any:
    """Lazy 初始化 configurable model"""
    global _configurable_model
    if _configurable_model is None:
        _configurable_model = init_chat_model(
            configurable_fields=("model", "max_tokens", "api_key", "base_url", "temperature"),
        )
    return _configurable_model


def get_configured_model(
    app_config: AppConfig,
    role: str = "research",
    bind_tools: list | None = None,
) -> Any:
    """根据角色返回正确配置的模型实例"""
    role_config_map = {
        "research": {
            "model": app_config.models.research,
            "max_tokens": app_config.models.research_max_tokens,
            "temperature": app_config.models.research_temperature,
        },
        "decision": {
            "model": app_config.models.decision,
            "max_tokens": app_config.models.decision_max_tokens,
            "temperature": app_config.models.decision_temperature,
        },
        "compression": {
            "model": app_config.models.compression,
            "max_tokens": app_config.models.compression_max_tokens,
            "temperature": app_config.models.compression_temperature,
        },
        "report": {
            "model": app_config.models.report,
            "max_tokens": app_config.models.report_max_tokens,
            "temperature": app_config.models.report_temperature,
        },
    }

    role_cfg = role_config_map.get(role, role_config_map["research"])
    model_name = role_cfg["model"]

    model_config = {
        "model": model_name,
        "max_tokens": role_cfg["max_tokens"],
        "api_key": get_api_key(app_config, model_name),
        "base_url": app_config.models.base_url,
        "temperature": role_cfg["temperature"],
        "tags": ["langsmith:nostream"],
    }

    model = _get_configurable_model()
    if bind_tools:
        model = model.bind_tools(bind_tools)
    model = model.with_retry(stop_after_attempt=3).with_config(model_config)
    return model


def get_api_key(config: AppConfig, model_name: str = "") -> Optional[str]:
    """获取模型对应的 API Key

    根据 base_url 和 model_name 前缀自动匹配：
      - volces.com base_url → VOLCANO_API_KEY
      - anthropic: 前缀 → ANTHROPIC_API_KEY
      - openai: 前缀 → OPENAI_API_KEY
      - 兜底按优先级尝试
    """
    if config.models.base_url and "volces.com" in config.models.base_url:
        if config.get_api_keys_from_config:
            return config.api_keys.get("VOLCANO_API_KEY")
        return os.getenv("VOLCANO_API_KEY")

    if model_name.startswith("anthropic:") or "claude" in model_name.lower():
        if config.get_api_keys_from_config:
            return config.api_keys.get("ANTHROPIC_API_KEY")
        return os.getenv("ANTHROPIC_API_KEY")

    if model_name.startswith("openai:"):
        if config.get_api_keys_from_config:
            return config.api_keys.get("OPENAI_API_KEY")
        return os.getenv("OPENAI_API_KEY")

    if config.get_api_keys_from_config:
        return (config.api_keys.get("VOLCANO_API_KEY")
                or config.api_keys.get("OPENAI_API_KEY")
                or config.api_keys.get("ANTHROPIC_API_KEY"))
    return os.getenv("VOLCANO_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
