import os
import asyncio
import logging
import httpx
from functools import lru_cache
from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# 使用标准日志模块替代自定义导入
logger = logging.getLogger(__name__)

# 全局缓存客户端实例
_cached_llamafactory_client: Optional[httpx.AsyncClient] = None

def initialize_llamafactory_client(
    base_url: str = None,
    api_key: str = None
) -> httpx.AsyncClient:
    """
    初始化并缓存 LLaMA Factory API 客户端
    
    参数:
        base_url: API 基础 URL
        api_key: API 密钥
        
    返回:
        配置好的异步 HTTP 客户端
    """
    global _cached_llamafactory_client
    if _cached_llamafactory_client is None:
        base_url = base_url or os.getenv("LLAMA_FACTORY_BASE_URL", "http://localhost:8000")
        api_key = api_key or os.getenv("LLAMA_FACTORY_API_KEY", "")
        
        logger.info(f"初始化 LLaMA Factory 客户端: {base_url}")
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # 创建带超时和连接限制的客户端
        _cached_llamafactory_client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            transport=httpx.AsyncHTTPTransport(retries=3)  # 添加自动重试
        )
    
    return _cached_llamafactory_client

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def llamafactory_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    **generation_params
) -> str:
    """
    LLaMA Factory 聊天补全 API 的异步封装
    
    参数:
        model_name: 使用的模型名称
        prompt: 用户提示
        system_prompt: 系统级提示（可选）
        history_messages: 对话历史（可选）
        generation_params: 生成参数
        
    返回:
        生成的文本响应
    """
    # 从参数中获取基础URL和API密钥
    base_url = generation_params.pop("base_url", None)
    api_key = generation_params.pop("api_key", None)
    
    # 初始化客户端
    client = initialize_llamafactory_client(base_url, api_key)
    
    # 准备消息列表
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # 准备请求体
    request_body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_params.get("max_tokens", 4096),
        "temperature": generation_params.get("temperature", 0.0),
        "top_p": generation_params.get("top_p", 0.9),
        "stream": False
    }
    
    # 添加可选参数
    for key in ["frequency_penalty", "presence_penalty", "stop"]:
        if key in generation_params:
            request_body[key] = generation_params[key]
    
    try:
        # 调用 API
        response = await client.post(
            "/v1/chat/completions",
            json=request_body
        )
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"API 错误: {e.response.status_code} - {e.response.text}")
        raise
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error(f"网络错误: {str(e)}")
        raise
    except (KeyError, TypeError) as e:
        logger.error(f"无效的响应格式: {str(e)}")
        raise ValueError("无效的 API 响应格式")
    except json.JSONDecodeError:
        logger.error(f"JSON 解析失败: {response.text}")
        raise ValueError("API 响应不是有效的 JSON")

async def llamafactory_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    model_name: Optional[str] = None,
    **kwargs
) -> Union[str, Dict]:
    """
    LLaMA Factory 补全的统一接口
    
    参数:
        prompt: 用户提示
        system_prompt: 系统级提示（可选）
        history_messages: 对话历史（可选）
        model_name: 使用的模型名称（可选）
        kwargs: 其他参数
        
    返回:
        生成的文本或提取的关键词
    """
    # 确定模型名称
    if not model_name:
        if "hashing_kv" in kwargs and "global_config" in kwargs["hashing_kv"]:
            model_name = kwargs["hashing_kv"]["global_config"]["llm_model_name"]
        else:
            raise ValueError("未提供模型名称且无法从配置中获取")
    
    # 调用生成函数
    return await llamafactory_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )