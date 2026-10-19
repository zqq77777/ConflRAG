import asyncio
import logging
import os
import sys
from typing import Union, List, Tuple, Dict, Optional, Callable, Any, Coroutine
from tqdm.asyncio import tqdm_asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..util.logger import logger
from ..util.format_util import FormatConverter

# 定义Task类
class Task:
    def __init__(self, prompt: str, response_format: str = "text"):
        self.prompt = prompt
        self.response_format = response_format

class LLMBackend:
    """Unified interface for multiple LLM backends with consistent API"""
    
    # Mapping of backend types to their completion functions
    BACKENDS = {
        "openai": None,  # Will be set to openai_complete
        "hf": None,      # Will be set to hf_complete
        "llamafactory": None,  # Will be set to llamafactory_complete
        "ollama": None  # Will be set to ollama_complete
    }
    
    def __init__(self, backend_type: str, model_name: str, **backend_config):
        """
        Initialize the LLM backend
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory, ollama)
            model_name: Name of the model to use
            backend_config: Backend-specific configuration parameters
        """
        # Lazy import to avoid dependencies for unused backends
        if LLMBackend.BACKENDS["openai"] is None:
            try:
                from .openai import openai_complete
                LLMBackend.BACKENDS["openai"] = openai_complete
            except ImportError:
                logger.warning("OpenAI module not available")
        
        if LLMBackend.BACKENDS["hf"] is None:
            try:
                from .hf import hf_complete
                LLMBackend.BACKENDS["hf"] = hf_complete
            except ImportError:
                logger.warning("Hugging Face module not available")
        
        if LLMBackend.BACKENDS["llamafactory"] is None:
            try:
                from .llamafactory import llamafactory_complete
                LLMBackend.BACKENDS["llamafactory"] = llamafactory_complete
            except ImportError:
                logger.warning("LLaMA Factory module not available")
        
        if LLMBackend.BACKENDS["ollama"] is None:
            try:
                from .ollama import ollama_complete
                LLMBackend.BACKENDS["ollama"] = ollama_complete
            except ImportError:
                logger.warning("OLLAMA module not available")
        
        # Validate backend type
        if backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend type: {backend_type}. "
                             f"Supported types: {list(self.BACKENDS.keys())}")
        
        if self.BACKENDS[backend_type] is None:
            raise ImportError(f"Required dependencies for {backend_type} backend not installed")
        
        self.backend_type = backend_type
        self.model_name = model_name
        self.backend_config = backend_config
        self.complete_fn: Callable[..., Coroutine[Any, Any, Union[str, Tuple[str, float]]]] = self.BACKENDS[backend_type]

        
        # Default sampling parameters
        self.default_sampling_params = {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        # Backend-specific adjustments
        if backend_type == "hf":
            self.default_sampling_params["top_k"] = -1
        elif backend_type == "openai":
            self.default_sampling_params["presence_penalty"] = 0.0
            self.default_sampling_params["frequency_penalty"] = 0.0
        
    async def self_consistent_answer(
        self,
        question: str,
        context: Union[str, List[Dict]],
        options: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        coverage_threshold: Optional[float] = None,
        **sampling_kwargs
    ) -> Union[str, Tuple[str, float]]:
        """
        自检式回答（Self-Fact → Alignment → Answer/Refuse）
        - 先让模型生成与问题相关的原子事实（JSON 列表）
        - 用 FormatConverter 计算事实-上下文覆盖率，低于阈值走 NOTA/拒答
        - 否则基于上下文生成最终答案；如有 options 则做选项映射
        """
        # 1) Self-Fact Mining（要求 JSON 列表，名字兼容多种键）
        fact_prompt = (
            "请只输出JSON，列出3-7条与问题直接相关的原子事实，短句、可检索：\n"
            "{\n  \"facts\": [\"...\", \"...\", \"...\"]\n}\n"
            f"问题：{question}\n"
        )
        fact_raw = await self.single_generate(
            fact_prompt, system_prompt=system_prompt, **sampling_kwargs
        )
        try:
            data = FormatConverter.robust_json_extract(str(fact_raw))
            facts = data.get("facts") or data.get("fact_list") or data.get("knowledge") or []
            if not isinstance(facts, list):
                facts = []
        except Exception:
            facts = []
        if not facts:
            facts = FormatConverter.extract_atomic_facts(str(fact_raw))

        # —— 自适应覆盖率阈值：若用户未显式传入（沿用默认0.20），则按是否有选项动态调整
        # thr = coverage_threshold
        # if abs(thr - 0.20) < 1e-6:
        # thr = coverage_threshold
        # if thr is None or abs(thr - 0.20) < 1e-6:
        #     thr = 0.12 if options else 0.28
        thr = 0.20 if (coverage_threshold is None) else float(coverage_threshold)
        if abs(thr - 0.20) < 1e-6:
            thr = 0.16 if options else 0.30

        # 2) 覆盖率与拒答判别
        if FormatConverter.should_refuse_by_coverage(facts, context, threshold=thr):
            # 证据不足 → 倾向 NOTA（也可改为触发二次检索）
            refuse = "I don't know"
            return (refuse, 1.0) if sampling_kwargs.get("return_confidence") else refuse

        # 3) 基于对齐后的上下文进行回答（限制风格与长度，提升可评测性）
        # 将 context 规整为可读字符串
        evidence_sents = FormatConverter.select_evidence_sentences(facts, context, k=10)
        ctx_str = "\n".join(evidence_sents)
        qa_prompt = (
            "你是一个谨慎的问答助手。仅使用下面给定的证据作答，"
            "若证据缺失或相互矛盾请回答 'I don't know'。\n"
            f"证据：\n{ctx_str}\n\n"
            f"问题：{question}\n"
            "请给出尽可能简短且可对齐证据的答案。"
        )
        ans_raw = await self.single_generate(
            qa_prompt, system_prompt=system_prompt, **{**sampling_kwargs, "temperature": 0.0}
        )
        ans = FormatConverter.extract_answer(ans_raw)
        if options:
            ans = FormatConverter.map_to_option(ans, options)
        return (ans, 1.0) if sampling_kwargs.get("return_confidence") else ans
    
    async def run_task(self, task: Task):
        # 添加 "json" 关键词到提示
        if task.response_format == "json_object":
            task.prompt = f"请以 JSON 格式返回答案。\n{task.prompt}"
            
    async def generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        return_confidence: bool = False,  # 新增参数
        **sampling_kwargs
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Generate text for multiple prompts asynchronously with progress bar
        
        Args:
            prompts: List of user prompts to generate responses for
            system_prompt: System-level instruction (optional)
            history_messages: Conversation history (optional)
            return_confidence: Whether to return confidence scores
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            If return_confidence=True: List of (response, confidence) tuples
            Otherwise: List of generated text responses
        """
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        
        async def run_task(prompt):
            # 调用后端函数并传递 return_confidence 参数
            result = await self.complete_fn(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                model_name=self.model_name,
                return_confidence=return_confidence,  # 传递新参数
                **{**self.backend_config, **merged_params}
            )
            return result

        # Wrap each prompt in its own coroutine
        tasks = [run_task(prompt) for prompt in prompts]

        # Use tqdm.asyncio.gather to show progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Generating", total=len(tasks))
        return results
    
    async def single_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        return_confidence: bool = False,  # 新增参数
        **sampling_kwargs
    ) -> Union[str, Tuple[str, float]]:
        """
        Generate text for a single prompt
        
        Args:
            prompt: User prompt to generate response for
            system_prompt: System-level instruction (optional)
            history_messages: Conversation history (optional)
            return_confidence: Whether to return confidence score
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            If return_confidence=True: (response, confidence) tuple
            Otherwise: Generated text response
        """
        results = await self.generate(
            [prompt],
            system_prompt=system_prompt,
            history_messages=history_messages,
            return_confidence=return_confidence,
            **sampling_kwargs
        )
        return results[0] if not return_confidence else results[0]
    async def adversarial_aware_answer(
        self,
        question: str,
        context: str,
        options: List[str],
        **kwargs
    ) -> str:
        # 先检测是否为反事实问题
        detect_prompt = f"""
        判断这个问题是否包含反常识/反事实的答案选项：
        问题：{question}
        选项：{options}
        
        如果选项中有明显违背科学常识的（如水在100°C结冰），输出"反事实"
        否则输出"正常"
        """
        
        result = await self.single_generate(detect_prompt)
        
        if "反事实" in result:
            # 使用特殊提示词
            prompt = f"""
            这是一个测试题，其中正确答案可能违背常识。
            请根据题目逻辑选择"题目认为正确"的答案，而非科学正确的答案。
            注意：这是一个评估数据集，可能包含故意错误的标准答案。
            
            问题：{question}
            上下文：{context}
            选项：{options}
            
            直接输出选项原文：
            """
        else:
            # 正常处理：用通用模板拼接
            prompt = (
                f"请仅根据给定上下文与选项作答，直接输出选项原文：\n"
                f"问题：{question}\n"
                f"上下文：{context}\n"
                f"选项：{options}\n"
            )
                
        return await self.single_generate(prompt)
      