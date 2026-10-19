import os
import copy
import asyncio
import logging
import torch
from functools import lru_cache
from typing import Dict, List, Optional, Union, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig
)
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    RevisionNotFoundError
)

# 确保 logger 被定义
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# ... [文件其余部分保持不变] ...

@lru_cache(maxsize=5)
def initialize_hf_client(
    model_name: str, 
    device_map: str = "auto", 
    torch_dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> tuple:
    logger.info(f"Loading Hugging Face model: {model_name}")

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
            load_in_8bit=load_in_8bit if not load_in_4bit else False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generation_config = GenerationConfig.from_pretrained(model_name)
        generation_config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer, generation_config
    
    except (RepositoryNotFoundError, GatedRepoError, RevisionNotFoundError) as e:
        logger.error(f"Model loading error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        raise

def format_chat_messages(
    messages: List[Dict[str, str]], 
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None
) -> str:
    formatted_messages = []
    
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    
    for msg in messages:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        if "apply_chat_template" in dir(tokenizer):
            return tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
    except Exception:
        logger.warning("Chat template application failed, using fallback formatting")
    
    return "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in formatted_messages]
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (RuntimeError, OSError, torch.cuda.OutOfMemoryError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def hf_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    return_confidence: bool = False,  # 新增参数
    **generation_params
) -> Union[str, Tuple[str, float]]:  # 修改返回类型
    model, tokenizer, generation_config = initialize_hf_client(model_name)
    
    input_text = format_chat_messages(
        history_messages + [{"role": "user", "content": prompt}],
        tokenizer,
        system_prompt
    )
    
    params = generation_config.to_dict()
    params.update({
        "max_new_tokens": generation_params.get("max_tokens", 512),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.9),
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": return_confidence,  # 仅在需要置信度时计算分数
        "return_dict_in_generate": return_confidence  # 仅在需要置信度时返回字典
    })
    
    generation_params.pop("hashing_kv", None)
    generation_params.pop("keyword_extraction", None)
    
    params.update(generation_params)
    
    def _sync_generate():
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=8192  
        ).to(model.device)
        
        stop_token_ids = [tokenizer.eos_token_id]
        if tokenizer.pad_token_id:
            stop_token_ids.append(tokenizer.pad_token_id)
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                **params
            )
        
        # 提取生成的文本
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:] 
            if not return_confidence else outputs.sequences[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # 计算置信度
        if return_confidence:
            try:
                # 获取分数（如果可用）
                if hasattr(outputs, 'scores') and outputs.scores is not None:
                    # 计算每个token的softmax概率
                    scores = outputs.scores
                    probs = [torch.softmax(score, dim=-1) for score in scores]
                    
                    # 获取每个token的最大概率
                    token_probs = [torch.max(prob, dim=-1).values.cpu().numpy()[0] for prob in probs]
                    
                    # 计算平均token概率
                    if token_probs:
                        confidence = sum(token_probs) / len(token_probs)
                    else:
                        confidence = 0.0
                else:
                    # 回退机制：基于文本长度的启发式方法
                    confidence = min(1.0, max(0.0, len(generated_text) / 100))
                
                return generated_text, float(confidence)
            except Exception as e:
                logger.error(f"Error calculating confidence: {e}")
                return generated_text, 0.5  # 默认置信度
        else:
            return generated_text
    
    result = await asyncio.to_thread(_sync_generate)
    return result

async def hf_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    model_name: Optional[str] = None,
    keyword_extraction: bool = False,
    return_confidence: bool = False,  # 新增参数
    **kwargs
) -> Union[str, Tuple[str, float]]:  # 修改返回类型
    
    if not model_name:
        if "hashing_kv" in kwargs and "global_config" in kwargs["hashing_kv"]:
            model_name = kwargs["hashing_kv"]["global_config"]["llm_model_name"]
        else:
            raise ValueError("Model name not provided and could not be obtained from configuration")
    
    result = await hf_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        return_confidence=return_confidence,  # 传递新参数
        **kwargs
    )
     
    return result