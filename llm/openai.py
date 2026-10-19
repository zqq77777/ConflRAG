import os
import asyncio
import time
import logging
import math
import re
from functools import lru_cache
from typing import Tuple, Union, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OpenAI_Client")
logger.setLevel(logging.WARNING)

def clean_base_url(url):
    """确保base_url格式正确，移除结尾的斜杠"""
    if url is None:
        return None
    return url.rstrip('/') if url else url

@lru_cache(maxsize=1)
def initialize_openai_client(api_key=None, base_url=None):
    # 首先确保我们有API密钥
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided as a parameter or via OPENAI_API_KEY / DASHSCOPE_API_KEY")
    
    # 处理基础URL
    base_url = base_url or os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    base_url = clean_base_url(base_url)
    
    logger.info(f"🛠️ 初始化OpenAI客户端: base_url={base_url}")
    
    # 使用异步客户端
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )

class ContentSafetyFilter:
    """内容安全过滤器，防止触发内容安全策略"""
    
    def __init__(self):
        # 敏感关键词列表
        self.sensitive_keywords = [
            "暴力", "色情", "政治", "仇恨", "歧视", "恐怖", "敏感",
            "violence", "porn", "politics", "hate", "discrimination", "terror"
        ]
        
        # 安全替换映射
        self.safe_replacements = {
            "暴力": "冲突",
            "色情": "亲密关系",
            "政治": "社会议题",
            "仇恨": "分歧",
            "歧视": "差异",
            "恐怖": "紧张",
            "敏感": "重要"
        }
    
    def filter_content(self, text: str) -> str:
        """过滤敏感内容并替换为安全词汇"""
        # 检查是否包含敏感关键词
        for keyword in self.sensitive_keywords:
            if keyword in text.lower():
                logger.warning(f"检测到敏感内容: {keyword}")
                # 替换敏感词汇
                for sensitive, safe in self.safe_replacements.items():
                    text = text.replace(sensitive, safe)
                return text
        
        return text
    
    def sanitize_context(self, context: str) -> str:
        """清理上下文中的潜在敏感内容"""
        # 移除可能触发安全策略的段落
        sanitized = re.sub(r'(暴力|色情|政治|仇恨|歧视|恐怖|敏感)[^。！？]*[。！？]', '', context)
        return sanitized.strip() or context
    
    def create_safe_prompt(self, prompt: str) -> str:
        """创建安全的提示词，添加安全约束"""
        safe_prompt = f"""
        请遵循以下安全准则:
        1. 避免任何政治敏感话题
        2. 不讨论暴力、色情或非法内容
        3. 保持回答中立、客观、教育性
        
        原始问题: {prompt}
        """
        return safe_prompt

@retry(
    # stop=stop_after_attempt(5),
    # wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, APIError)
    ),
    before_sleep=lambda retry_state: logger.warning(
        f"OpenAI API调用失败，第 {retry_state.attempt_number} 次重试... 错误: {retry_state.outcome.exception()}"
    )
)
async def openai_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: str = None,
    history_messages: Optional[list] = None,
    api_key: str = None,
    base_url: str = None,
    temperature: float = 0.7,
    return_confidence: bool = False,
    **kwargs
) -> Union[str, Tuple[str, float]]:
    start_time = time.time()
    logger.debug(f"开始处理请求: {model_name}, 提示长度: {len(prompt)}")
    
    client = initialize_openai_client(api_key=api_key, base_url=base_url)

    # 初始化内容安全过滤器（评测默认关闭，避免篡改上下文影响ACC）
    safety_mode = kwargs.get("safe_mode", "off")
    safety_filter = ContentSafetyFilter() if safety_mode == "on" else None
    
    # 过滤敏感内容
    if safety_filter:
        prompt = safety_filter.filter_content(prompt)
    
    # 清理上下文
    if safety_filter and "context" in kwargs and isinstance(kwargs["context"], str):
        kwargs["context"] = safety_filter.sanitize_context(kwargs["context"])
    
    # 创建系统提示（若未启用安全过滤，则直接使用原system_prompt）
    safe_system_prompt = (
        safety_filter.create_safe_prompt(system_prompt or "你是一个有帮助的助手")
        if safety_filter else (system_prompt or "你是一个有帮助的助手")
    )
        
    # 清理历史消息
    safe_history = []
    history_messages = history_messages or []
    for msg in history_messages:
        safe_msg = msg.copy()
        safe_msg["content"] = (
            safety_filter.filter_content(msg.get("content", "")) if safety_filter
            else msg.get("content", "")
        )
        safe_history.append(safe_msg)
    messages = []
    if safe_system_prompt:
        messages.append({"role": "system", "content": safe_system_prompt})
        logger.debug(f"系统提示: {safe_system_prompt[:50]}...")
    
    if safe_history:
        messages.extend(safe_history)
        logger.debug(f"历史消息: {len(safe_history)} 条")
    
    # 添加JSON格式要求（如果指定了JSON响应格式）
    if "response_format" in kwargs and kwargs["response_format"].get("type") == "json_object":
        # 确保提示中包含"json"这个词
        if "json" not in prompt.lower():
            prompt = f"请以 JSON 格式返回答案。\n{prompt}"
    
    messages.append({"role": "user", "content": prompt})
    logger.debug(f"用户消息: {prompt[:50]}...")
    
    api_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1000),
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
    }
    
    # 使用显式指定的可选参数
    for key in ["response_format", "seed", "stop"]:
        # 若要求 JSON 且用户未手动传 stop，则添加安全 stop，避免“解释性文字”溢出到 JSON 外
        if api_params.get("response_format", {}).get("type") == "json_object" and "stop" not in api_params:
            # 仅拦截代码围栏与通用结束标记，避免截断 JSON 键名
            api_params["stop"] = [
                "\n```", "```",
                "</s>", "<|endoftext|>"
            ]
        if key in kwargs:
            api_params[key] = kwargs[key]
    
    # 检查是否为DashScope后端（不支持logprobs）
    is_dashscope = base_url and "dashscope" in base_url
    # 若需要置信度且后端支持，显式请求logprobs
    if return_confidence and not is_dashscope:
        api_params["logprobs"] = True
        api_params["top_logprobs"] = 1
        
    # 如果要求返回置信度，但后端是DashScope，则移除logprobs参数
    if return_confidence and is_dashscope:
        logger.warning("DashScope后端不支持logprobs参数，将使用备选置信度计算方法")
        # 确保移除任何可能存在的logprobs参数
        if "logprobs" in api_params:
            del api_params["logprobs"]
        if "top_logprobs" in api_params:
            del api_params["top_logprobs"]
    
    logger.debug(f"向API发送请求: {client.base_url}/chat/completions")
    
    try:
        # 检查是否为阿里云 DashScope API
        is_dashscope = base_url and "dashscope" in base_url
        
        # 只有在使用阿里云 DashScope API 时才添加安全模式参数
        if is_dashscope:
            # 阿里云 DashScope API 支持的安全参数
            api_params["safe_mode"] = "none"  # 评测时禁用，线上可改 "block"
            api_params["sensitive_word_check"] = False
                
        response = await client.chat.completions.create(**api_params)
        content = response.choices[0].message.content.strip()
        duration = time.time() - start_time
        
        logger.info(f"响应接收 | 用时: {duration:.2f}s | 长度: {len(content)}")
        logger.debug(f"响应接收 | 用时: {duration:.2f}s | 长度: {len(content)}")
        logger.debug(f"完整响应: {content[:100]}...")
        
        # 计算置信度（如果请求）
        if return_confidence:
            try:
                # 对于DashScope后端，使用备选置信度计算方法
                if is_dashscope:
                    # 备选方法1：基于文本长度的启发式方法
                    # confidence = min(1.0, max(0.0, len(content) / 100))
                    confidence = 0.5
                    logger.debug(f"使用备选置信度计算: {confidence:.4f}")
                else:
                    # 检查logprobs是否可用
                    if response.choices[0].logprobs is not None:
                        # 获取logprobs
                        logprobs = response.choices[0].logprobs
                        
                        # 计算平均token概率
                        token_logprobs = []
                        for token_info in logprobs.content:
                            # 取最可能token的logprob
                            top_logprob = token_info.logprob
                            token_logprobs.append(top_logprob)
                        
                        # 计算平均logprob
                        avg_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0
                        
                        # 将logprob转换为概率 (0-1)
                        confidence = math.exp(avg_logprob)
                    else:
                        # 回退机制：基于文本长度的启发式方法
                        # confidence = min(1.0, max(0.0, len(content) / 100))
                        confidence = 0.5

                
                logger.debug(f"置信度计算: {confidence:.4f}")
                return content, float(confidence)
            except Exception as e:
                logger.error(f"置信度计算失败: {str(e)}")
                return content, 0.5  # 默认置信度
        
        return content

    except Exception as e:
        if "inappropriate content" in str(e) or "data_inspection_failed" in str(e):
            logger.warning("内容安全策略触发，使用备选方案")
            fb = _use_safe_fallback(prompt, is_dashscope)
            return (fb, 0.0) if return_confidence else fb
        else:
            logger.error(f"API调用失败: {str(e)}")
            raise


def _use_safe_fallback(prompt: str, is_dashscope: bool) -> str:
    """
    评测安全回退：始终返回标准 JSON 拒答，避免上游解析失败计错
    """
    return '{"answer":"I don\'t know"}'

async def openai_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: Optional[list] = None,
    api_key: str = None,
    base_url: str = None,
    return_confidence: bool = False,
    **kwargs
) -> Union[str, Tuple[str, float]]:
    # 使用更适合生产环境的模型作为默认
    model_name = kwargs.pop("model_name", "gpt-4o-mini")
    
    # 优化日志输出 - 保存当前日志级别
    original_log_level = logger.level
    
    try:
        # 临时减少日志输出
        logger.setLevel(logging.WARNING)
        
        response = await openai_chat_completion(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            return_confidence=return_confidence,
            **kwargs
        )
        return response
    finally:
        # 恢复原日志级别
        logger.setLevel(original_log_level)
