"""网络搜索工具模块

支持 Tavily 和 Chinalin 两种搜索引擎，通过配置文件 search.engine 字段切换。
搜索结果通过 LLM 进行摘要（不截断原始内容），输出结构化的 summary + key_excerpts。

引擎切换方式：在 portfolio_config.yaml 中设置 search.engine 为 "chinalin" 或 "tavily"。
"""

import aiohttp
import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from src.config import get_app_config, get_api_key

logger = logging.getLogger(__name__)


# ============================================================
# 结构化输出：摘要模型
# ============================================================
class Summary(BaseModel):
    """Research summary with key findings."""

    summary: str
    key_excerpts: str


# ============================================================
# 摘要 Prompt（与 open_deep_research 保持一致）
# ============================================================
SUMMARIZE_WEBPAGE_PROMPT = """You are tasked with summarizing the raw content of a webpage retrieved from a web search. Your goal is to create a summary that preserves the most important information from the original web page. This summary will be used by a downstream research agent, so it's crucial to maintain the key details without losing essential information.

Here is the raw content of the webpage:

<webpage_content>
{webpage_content}
</webpage_content>

Please follow these guidelines to create your summary:

1. Identify and preserve the main topic or purpose of the webpage.
2. Retain key facts, statistics, and data points that are central to the content's message.
3. Keep important quotes from credible sources or experts.
4. Maintain the chronological order of events if the content is time-sensitive or historical.
5. Preserve any lists or step-by-step instructions if present.
6. Include relevant dates, names, and locations that are crucial to understanding the content.
7. Summarize lengthy explanations while keeping the core message intact.

When handling different types of content:

- For news articles: Focus on the who, what, when, where, why, and how.
- For scientific content: Preserve methodology, results, and conclusions.
- For opinion pieces: Maintain the main arguments and supporting points.
- For product pages: Keep key features, specifications, and unique selling points.

Your summary should be significantly shorter than the original content but comprehensive enough to stand alone as a source of information. Aim for about 25-30 percent of the original length, unless the content is already concise.

Present your summary in the following format:

```
{{
   "summary": "Your summary here, structured with appropriate paragraphs or bullet points as needed",
   "key_excerpts": "First important quote or excerpt, Second important quote or excerpt, Third important quote or excerpt, ...Add more excerpts as needed, up to a maximum of 5"
}}
```

Here are two examples of good summaries:

Example 1 (for a news article):
```json
{{
   "summary": "On July 15, 2023, NASA successfully launched the Artemis II mission from Kennedy Space Center. This marks the first crewed mission to the Moon since Apollo 17 in 1972. The four-person crew, led by Commander Jane Smith, will orbit the Moon for 10 days before returning to Earth. This mission is a crucial step in NASA's plans to establish a permanent human presence on the Moon by 2030.",
   "key_excerpts": "Artemis II represents a new era in space exploration, said NASA Administrator John Doe. The mission will test critical systems for future long-duration stays on the Moon, explained Lead Engineer Sarah Johnson. We're not just going back to the Moon, we're going forward to the Moon, Commander Jane Smith stated during the pre-launch press conference."
}}
```

Example 2 (for a scientific article):
```json
{{
   "summary": "A new study published in Nature Climate Change reveals that global sea levels are rising faster than previously thought. Researchers analyzed satellite data from 1993 to 2022 and found that the rate of sea-level rise has accelerated by 0.08 mm/year² over the past three decades. This acceleration is primarily attributed to melting ice sheets in Greenland and Antarctica. The study projects that if current trends continue, global sea levels could rise by up to 2 meters by 2100, posing significant risks to coastal communities worldwide.",
   "key_excerpts": "Our findings indicate a clear acceleration in sea-level rise, which has significant implications for coastal planning and adaptation strategies, lead author Dr. Emily Brown stated. The rate of ice sheet melt in Greenland and Antarctica has tripled since the 1990s, the study reports. Without immediate and substantial reductions in greenhouse gas emissions, we are looking at potentially catastrophic sea-level rise by the end of this century, warned co-author Professor Michael Green."  
}}
```

Remember, your goal is to create a summary that can be easily understood and utilized by a downstream research agent while preserving the most critical information from the original webpage.

Today's date is {date}.
"""


# ============================================================
# LRU 缓存（摘要模型实例复用）
# ============================================================
class _LRUCache:
    """带有过期时间的 LRU 缓存，用于复用摘要模型实例。"""

    def __init__(self, max_size: int = 5, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl_seconds:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (value, time.time())
        while len(self._cache) > self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    def clear(self):
        self._cache.clear()


_summarization_model_cache = _LRUCache(max_size=5, ttl_seconds=3600)


# ============================================================
# 全局 aiohttp 会话管理（连接池复用）
# ============================================================
_global_session: Optional[aiohttp.ClientSession] = None


async def _get_global_session() -> aiohttp.ClientSession:
    """获取或创建全局 aiohttp 会话，实现连接池复用。"""
    global _global_session
    if _global_session is None or _global_session.closed:
        _global_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=20, limit_per_host=10),
        )
    return _global_session


async def close_global_session():
    """关闭全局 aiohttp 会话。应在程序退出时调用。"""
    global _global_session
    if _global_session is not None and not _global_session.closed:
        await _global_session.close()
        _global_session = None


def _get_today_str() -> str:
    """获取当前日期字符串，用于 prompt 中的日期上下文。"""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


# ============================================================
# 网页摘要函数
# ============================================================
async def summarize_webpage(model, webpage_content: str) -> str:
    """使用 LLM 对网页内容生成结构化摘要，带超时保护。

    Args:
        model: 配置好的 chat model（已绑定 structured output）
        webpage_content: 原始网页内容

    Returns:
        格式化的 <summary> + <key_excerpts> 字符串；
        超时或异常时返回原始内容
    """
    try:
        prompt_content = SUMMARIZE_WEBPAGE_PROMPT.format(
            webpage_content=webpage_content,
            date=_get_today_str(),
        )

        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0,
        )

        return (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

    except asyncio.TimeoutError:
        logger.warning("摘要生成超时(60s)，返回原始内容")
        return webpage_content
    except Exception as e:
        logger.warning(f"摘要生成失败: {e}，返回原始内容")
        return webpage_content


# ============================================================
# Chinalin 搜索客户端
# ============================================================
class AsyncChinalinSearchClient:
    """异步 Chinalin 网络搜索客户端。

    封装 Chinalin 搜索 API，返回统一格式的搜索结果。
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "http://api-test.chinalions.cn",
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> Dict:
        """执行搜索查询，返回统一格式结果。"""
        session = await _get_global_session()

        url = f"{self.base_url}/aiassistant/getWebSearchData"
        payload = {
            "query": query,
            "rewrite": "false",
            "count": str(max_results),
            "SortByTime": True,
            "IsCheckWebSearch": False,
            "ExcludeUrl": ["cfi.cn"],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return self._parse_response(query, data)

        except aiohttp.ClientError as e:
            logger.error(f"Chinalin搜索请求失败 [query={query}]: {e}")
            return {"query": query, "results": []}
        except asyncio.TimeoutError:
            logger.error(f"Chinalin搜索请求超时 [query={query}]")
            return {"query": query, "results": []}
        except Exception as e:
            logger.error(f"Chinalin搜索未知错误 [query={query}]: {e}")
            return {"query": query, "results": []}

    def _parse_response(self, query: str, data: Dict) -> Dict:
        """将 Chinalin API 响应解析为统一格式。"""
        results = []

        code = data.get("code") if isinstance(data, dict) else None
        if code not in ("0", 0):
            logger.warning(
                f"Chinalin API 返回错误: code={code}, msg={data.get('msg', 'Unknown')} [query={query}]"
            )
            return {"query": query, "results": []}

        inner_data = data.get("data", {})
        raw_results = inner_data.get("webSearchResults", []) if isinstance(inner_data, dict) else []

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "content": item.get("snippet", ""),
                "raw_content": item.get("snippet", ""),
                "score": item.get("score", 0),
                "published_date": item.get("formattedTime", ""),
            })

        return {"query": query, "results": results}


# ============================================================
# Tavily 搜索客户端
# ============================================================
class AsyncTavilySearchClient:
    """异步 Tavily 搜索客户端。

    通过 Tavily REST API 执行搜索，返回统一格式结果。
    支持 general / news / finance 三种 topic 过滤。
    """

    TAVILY_API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        topic: str = "general",
        include_raw_content: bool = False,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.topic = topic
        self.include_raw_content = include_raw_content
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> Dict:
        """执行搜索查询，返回统一格式结果。"""
        session = await _get_global_session()

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "topic": self.topic,
            "include_raw_content": self.include_raw_content,
        }

        try:
            async with session.post(
                self.TAVILY_API_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                return {
                    "query": data.get("query", query),
                    "results": [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "content": r.get("content", ""),
                            "raw_content": r.get("raw_content", ""),
                            "score": r.get("score", 0),
                        }
                        for r in data.get("results", [])
                    ],
                }

        except aiohttp.ClientError as e:
            logger.error(f"Tavily搜索请求失败 [query={query}]: {e}")
            return {"query": query, "results": []}
        except asyncio.TimeoutError:
            logger.error(f"Tavily搜索请求超时 [query={query}]")
            return {"query": query, "results": []}
        except Exception as e:
            logger.error(f"Tavily搜索未知错误 [query={query}]: {e}")
            return {"query": query, "results": []}


# ============================================================
# 搜索客户端工厂
# ============================================================
def _create_search_client(search_config):
    """根据配置创建对应的搜索客户端实例。"""
    engine = search_config.engine.lower()

    if engine == "tavily":
        api_key = search_config.tavily_api_key
        if not api_key:
            raise ValueError(
                "Tavily 搜索引擎需要配置 search.tavily_api_key，"
                "请在 portfolio_config.yaml 中设置或通过 ${TAVILY_API_KEY} 环境变量注入"
            )
        return AsyncTavilySearchClient(
            api_key=api_key,
            topic=search_config.tavily_topic,
            include_raw_content=search_config.tavily_include_raw_content,
            timeout=search_config.timeout,
        )

    return AsyncChinalinSearchClient(
        api_key=search_config.chinalin_api_key or None,
        base_url=search_config.chinalin_base_url,
        timeout=search_config.timeout,
    )


def _get_summarization_model(app_config):
    """获取或创建摘要模型实例（带 LRU 缓存）。

    优先使用 search 配置中的 summarization_model，
    未配置时回退到 models.compression。
    """
    search_cfg = app_config.search
    models_cfg = app_config.models

    model_name = search_cfg.summarization_model or models_cfg.compression
    base_url = search_cfg.summarization_model_base_url or models_cfg.base_url
    max_tokens = search_cfg.summarization_model_max_tokens

    cache_key = f"{model_name}:{base_url}:{max_tokens}"
    cached = _summarization_model_cache.get(cache_key)
    if cached is not None:
        return cached

    api_key = get_api_key(app_config, model_name)
    model = init_chat_model(
        model=model_name,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        tags=["langsmith:nostream"],
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=search_cfg.max_structured_output_retries,
    )

    _summarization_model_cache.set(cache_key, model)
    logger.debug(f"创建新的摘要模型实例: {cache_key}")
    return model


# ============================================================
# 统一搜索工具
# ============================================================
@tool
async def web_search(
    queries: List[str],
    config: RunnableConfig = None,
) -> str:
    """网络搜索工具，支持同时执行多个搜索查询。

    并行获取结果后自动按 URL 去重，并使用 LLM 对每条结果生成结构化摘要。
    适用于检索公司公告、财务数据、行业信息、宏观政策、市场事件等公开信息。
    搜索引擎（Tavily / Chinalin）通过配置文件切换。

    Args:
        queries: 搜索查询关键词列表，支持多个查询并行执行
        config: LangChain 运行时配置

    Returns:
        格式化的搜索结果字符串，每条结果包含 LLM 生成的摘要和关键摘录
    """
    search_cfg = None
    app_cfg = None
    engine_name = "Chinalin"
    max_results = 5

    if config:
        app_cfg = get_app_config(config)
        search_cfg = app_cfg.search
        max_results = search_cfg.max_results
        engine_name = search_cfg.engine.capitalize()

    try:
        if search_cfg:
            client = _create_search_client(search_cfg)
        else:
            client = AsyncChinalinSearchClient()
    except ValueError as e:
        return f"搜索引擎初始化失败: {e}"

    logger.info(f"[{engine_name}] 执行搜索: {queries}")

    # Step 1: 并行执行所有搜索查询
    tasks = [client.search(query=q, max_results=max_results) for q in queries]
    all_responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Step 2: 按 URL 去重
    unique_results: Dict[str, Dict] = {}
    for response in all_responses:
        if isinstance(response, Exception):
            logger.error(f"搜索查询执行异常: {response}")
            continue
        if not isinstance(response, dict):
            continue
        for result in response.get("results", []):
            url = result.get("url", "")
            if url and url not in unique_results:
                unique_results[url] = {**result, "query": response.get("query", "")}
            elif not url:
                key = f"_no_url_{len(unique_results)}"
                unique_results[key] = {**result, "query": response.get("query", "")}

    if not unique_results:
        return f"未找到与以下查询相关的搜索结果：{', '.join(queries)}"

    # Step 3: LLM 摘要
    enable_summarization = True
    max_content_length = 50000

    if search_cfg:
        enable_summarization = search_cfg.enable_summarization
        max_content_length = search_cfg.max_content_length

    if enable_summarization and app_cfg:
        summarization_model = _get_summarization_model(app_cfg)

        async def _noop():
            return None

        summarization_tasks = [
            _noop() if not result.get("raw_content")
            else summarize_webpage(
                summarization_model,
                result["raw_content"][:max_content_length],
            )
            for result in unique_results.values()
        ]

        summaries = await asyncio.gather(*summarization_tasks)

        summarized_results = {
            url: {
                "title": result["title"],
                "content": result["content"] if summary is None else summary,
            }
            for url, result, summary in zip(
                unique_results.keys(),
                unique_results.values(),
                summaries,
            )
        }
    else:
        summarized_results = {
            url: {"title": result["title"], "content": result.get("content", "")}
            for url, result in unique_results.items()
        }

    # Step 4: 格式化输出（与 open_deep_research 格式一致）
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output
