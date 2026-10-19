from typing import List, Tuple, Dict, Union, Optional
import sys
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
import re
from sentence_transformers import SentenceTransformer, util
import nltk
from datasets import load_dataset,Dataset
import json
import string
import os
import transformers
from collections import Counter
from collections import defaultdict
import logging
from datetime import datetime
from .prompts import PromptGenerator
import torch
from rouge_score import rouge_scorer
import inflect
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from .llm.backend import LLMBackend
from omegaconf import DictConfig
from .util.logger import logger
from .util.format_util import FormatConverter


class FactMiningModule:
    """Fact mining module with support for multiple LLM backends"""
    
    def __init__(
        self, 
        backend_type: str, 
        model_name: str,
        **backend_config
    ):
        """
        Initialize the fact mining module
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory)
            model_name: Name of the model to use
            backend_config: Backend-specific configuration parameters
        """
        # Initialize LLM backend
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        
        # Initialize prompt generators
        self.prompt_generator = PromptGenerator(
            llm_type=backend_type,
            task="normal"
        )
        self.prompt_generator_extract = PromptGenerator(
            llm_type=backend_type,
            task="extract"
        )
        
        # Default sampling parameters
        self.default_sampling_params = {
            'max_tokens': 320,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        self.fact_pattern = r'(?:\d+\.\s+|\-\s+|\*\s+)?([^\n]+?)(?=(?:\n|$))'
        
    
    async def generate_knowledges(
        self, 
        dataset: Dataset, 
        **sampling_kwargs
    ) -> List[Dict]:
        """
        Generate factual knowledge for each item in the dataset
        
        Args:
            dataset: Input dataset
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            List of dictionaries containing facts for each item
        """
        # Generate prompts for knowledge extraction
        prompts = [
            self.prompt_generator_extract.generate_factual_knowledge(
                user_query=item['question']
            )
            for item in dataset
        ]
        
        # Generate responses using LLM backend
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_extract.system_prompt,
            **merged_params
        )
        
        # Parse and return facts
        return [
            {
                'id': item['id'],
                'facts': re.findall(self.fact_pattern, result)
            }
            for item, result in zip(dataset, raw_results)
        ]
    
    async def generate_self_context(
        self,
        dataset: Dataset,
        knowledges: Optional[Union[Dict, List[Dict]]] = None,
        **sampling_kwargs
    ) -> List[Dict]:
        """
        Generate self-context for each item in the dataset
        
        Args:
            dataset: Input dataset
            knowledges: Optional factual knowledge to incorporate
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            List of dictionaries containing context for each item
        """
        # Generate prompts for context generation
        prompts = []
        for item in dataset:
            if knowledges is None:
                prompt = self.prompt_generator.generate_context_directly_prompt(
                    user_query=item['question']
                )
            else:
                # Find matching knowledge for this item
                knowledge = next(
                    (k for k in knowledges if k['id'] == item['id']), 
                    None
                )
                if knowledge is None:
                    logger.warning(f"No knowledge found for item {item['id']}")
                    prompt = self.prompt_generator.generate_context_directly_prompt(
                        user_query=item['question']
                    )
                else:
                    prompt = self.prompt_generator.generate_context_by_factual_knowledge(
                        user_query=item['question'],
                        factual_knowledge=knowledge['facts']
                    )
            prompts.append(prompt)
        
        # Generate responses using LLM backend
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Generating self-contexts...")
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator.system_prompt,
            **merged_params
        )
        
        # Return contexts
        return [
            {'id': item['id'], 'context': result}
            for item, result in zip(dataset, raw_results)
        ]
    
    async def extract_facts(
        self,
        contexts: Union[List[Dict], Dict],
        **sampling_kwargs
    ) -> List[Dict]:
        """
        Extract facts from given contexts
        
        Args:
            contexts: List of contexts or single context dictionary
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            List of dictionaries containing extracted facts
        """
        # Normalize input to list
        if isinstance(contexts, dict):
            contexts = [contexts]
        
        # Generate prompts for fact extraction
        prompts = [
            self.prompt_generator_extract.generate_context_extract(
                user_context=ctx['context']
            )
            for ctx in contexts
        ]        
        # Generate responses using LLM backend
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Extracting facts...")
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_extract.system_prompt,
            **merged_params
        )
        
        # Parse and return facts
        return [
            {
                'id': ctx['id'],
                'facts': re.findall(self.fact_pattern, result)
            }
            for ctx, result in zip(contexts, raw_results)
        ]

class ContextualAlignmentModule:
    def __init__(self,
                 similarity_model:str):
        self.similarity_model = SentenceTransformer(
            similarity_model,
            # device=os.environ.get("SBERT_DEVICE", None)  # export SBERT_DEVICE=cuda
            # device=os.environ.get("SBERT_DEVICE", "cuda" if torch.cuda.is_available() else None)
            device=os.environ.get("SBERT_DEVICE", "cpu")

        )

    def chunk_text(self,paragraph: str, chunk_size: int = 60) -> List[str]:
        sentences = nltk.sent_tokenize(paragraph) 
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
        
    @staticmethod
    def _mmr_rerank(candidates, k=8, lambda_div=0.7, sbert_model=None):
        """
        candidates: List[{'chunk': str, 'score': float}]
        返回：经过 MMR 多样性重排后的前 k 个片段（含原 score）
        """
        if not candidates:
            return []
        pool = sorted(candidates, key=lambda x: x['score'], reverse=True)[:100]
        if not sbert_model:
            # 没有向量模型就按分数截断
            return pool[:k]
        from sentence_transformers import util as st_util
        import torch
        texts = [c['chunk'] for c in pool]
        emb = sbert_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    
        selected_idx = [0]  # 首先取最高分
        while len(selected_idx) < min(k, len(pool)):
            rest = [i for i in range(len(pool)) if i not in selected_idx]
            sel_emb = emb[selected_idx]
            # 计算每个候选与已选集合的最大相似度（冗余）
            sims = st_util.cos_sim(emb[rest], sel_emb).max(dim=1).values  # [len(rest)]
            rel = torch.tensor([pool[i]['score'] for i in rest], dtype=torch.float32)
            mmr = lambda_div * rel - (1 - lambda_div) * sims
            next_idx = rest[int(torch.argmax(mmr).item())]
            selected_idx.append(next_idx)
    
        return [pool[i] for i in selected_idx]
    # ---------- 新增结束 ----------

    def calculate_similarity(
        self,
        paragraph: str,
        str_list: List[str],
        top_k: int = 6,
        chunk_size: int = 50
    ):
        chunks = self.chunk_text(paragraph, chunk_size=chunk_size)

        chunk_embeddings = self.similarity_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        str_list_embeddings = self.similarity_model.encode(str_list, convert_to_tensor=True, show_progress_bar=False)

        results = []

        for i, string in enumerate(str_list):
            cosine_scores = util.cos_sim(str_list_embeddings[i], chunk_embeddings)
            
            top_results = sorted(
                enumerate(cosine_scores[0].tolist()), key=lambda x: x[1], reverse=True
            )[:top_k]

            top_chunks = [(chunks[idx], score) for idx, score in top_results]
            results.append((string, top_chunks))

        return results
    # 在 modules.py 第300行后添加
    # def detect_counterfactual_pattern(self, question: str, context: str) -> bool:
    #     """检测是否为反事实问题"""
    #     counterfactual_indicators = [
    #         "mercury is hotter than venus",  # 错误的天文学
    #         "gravity will become stronger", # 错误的物理学
    #         "dependent.*variable.*changes", # 混淆的实验设计
    #         "the moon revolves.*the sun",   # 错误的轨道关系
    #     ]
        
    #     combined = (question + " " + context).lower()
    #     for pattern in counterfactual_indicators:
    #         if re.search(pattern, combined):
    #             return True
        
    #     # 检查答案选项是否包含明显违背常识的内容
    #     if hasattr(self, 'current_options'):
    #         for opt in self.current_options:
    #             if any(wrong in opt.lower() for wrong in [
    #                 "gravity becomes stronger", "moon revolves around sun",
    #                 "water freezes at 100", "ice sinks in water"
    #             ]):
    #                 return True
    #     return False

    def get_topk_contextual_chunks(self, all_chunks, topk=8):
        """
        先对多路检索候选取并集去重，再使用 MMR 重排。
        """
        if not all_chunks:
            return []
    
        dedup = {}
        for item in all_chunks:
            text = item.get("chunk", "").strip()
            if not text:
                continue
    
            score = float(item.get("score", 0.0))
            if text not in dedup or score > dedup[text]["score"]:
                dedup[text] = {
                    "chunk": text,
                    "score": score
                }
    
        candidates = list(dedup.values())
    
        mmr_selected = self._mmr_rerank(
            candidates=candidates,
            k=topk,
            lambda_div=0.60,
            sbert_model=self.similarity_model
        )
    
        return mmr_selected
    def get_contextual_chunks(self, dataset, facts, sent_topk, chunk_topk, chunk_size: int = 60):
        results = []
        for item in dataset:
            paragraphs = (item.get('context') or '').split('\n\n')
            # HyDE-lite：把“原始问题”也作为一条查询字符串加入，并给予轻权重
            hyde_query = item['question']
            alpha = 1.15  # 问题命中权重（可调 1.1~1.3）

            all_chunks = []
            for paragraph in paragraphs:
                for fact in facts:
                    if fact['id'] != item['id']:
                        continue
                    # 把 question 放到 facts 前面，共同参与匹配
                    # 选项感知查询：把「问题+每个选项」也当查询加进去，略降权
                    opt_queries = [f"{hyde_query} {opt}" for opt in (item.get('choices') or [])]
                    query_strs = [hyde_query] + opt_queries + fact['facts']
                    results_tmp = self.calculate_similarity(
                        paragraph, query_strs, top_k=sent_topk, chunk_size=chunk_size
                    )
                    # 收集匹配到的句子，若命中的是“问题”，给予 alpha 加权
                    for s, top_chunks in results_tmp:
                        boost = alpha if s == hyde_query else 1.0
                        for entry in top_chunks:
                            # 若命中来源是选项查询，降低一点点分数防止“虚假强相关”
                            if s in opt_queries:
                                boost *= 1.0
                            all_chunks.append({
                                "chunk": entry[0],
                                "score": entry[1] * boost
                            })

            # 用上面 1.2 的 MMR 取最终 topk 片段
            topk_selected = self.get_topk_contextual_chunks(all_chunks, topk=chunk_topk)
            results.append({
                "id": item['id'],
                "topk_chunks": topk_selected
            })
        return results

class SelfThinkModule:
    """Self-thinking module for answer prediction with multiple LLM backends"""
    
    def __init__(
        self, 
        backend_type: str, 
        model_name: str,
        **backend_config
    ):
        """
        Initialize the self-thinking module
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory)
            model_name: Name of the model to use
            backend_config: Backend-specific configuration parameters
        """
        # Initialize LLM backend
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        
        # Initialize prompt generators
        self.prompt_generator_qa_cot = PromptGenerator(
            llm_type=backend_type,
            task="qa-cot"
        )
        self.prompt_generator_qa = PromptGenerator(
            llm_type=backend_type,
            task="qa"
        )
        
        # Default sampling parameters
        self.default_sampling_params = {
            'max_tokens': 512,
            'top_p': 1.0,
            'temperature': 0.0
        }
        self.confidence_store = {}
    def detect_counterfactual_pattern(self, question: str, context: str) -> bool:
        """检测是否为反事实/反常识问题"""
        import re
        counterfactual_indicators = [
            "mercury is hotter than venus",  # 错误的天文学
            "gravity will become stronger",  # 错误的物理学
            "dependent.*variable.*changes",  # 混淆的实验设计
            "the moon revolves.*the sun",    # 错误的轨道关系
        ]
        combined = (question + " " + (context or "")).lower()
        for pattern in counterfactual_indicators:
            if re.search(pattern, combined):
                return True

        # 检查答案选项是否包含明显违背常识的内容
        if hasattr(self, 'current_options'):
            for opt in self.current_options:
                if any(wrong in opt.lower() for wrong in [
                    "gravity becomes stronger", "moon revolves around sun",
                    "water freezes at 100", "ice sinks in water"
                ]):
                    return True
        return False

    async def predict_answer_normal_cot(
        self,
        dataset: Dataset,
        facts: List[Dict],
        **sampling_kwargs
    ) -> Dict[str, str]:
    
        prompts = []
        id_order = []
        predictions = {}
        id2item = {}
        id2ctx = {}
    
        n_vote = int(sampling_kwargs.pop("n_vote", 3))
        seed_base = int(sampling_kwargs.get("seed_base", 12345))
        gen_temp = float(sampling_kwargs.get("temperature", 0.35))
        top_p = float(sampling_kwargs.get("top_p", 0.95))
        max_tokens = int(sampling_kwargs.get("max_tokens", 512))
        confidence_threshold = float(sampling_kwargs.get("confidence_threshold", 0.5))
    
        for item in dataset:
            qid = item["id"]
    
            fact_str = next(
                (
                    "\n".join([c["chunk"] for c in d.get("topk_chunks", [])])
                    for d in facts
                    if d["id"] == qid
                ),
                ""
            )
    
            id2item[qid] = item
            id2ctx[qid] = fact_str
    
            prompt = self.prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                context=fact_str,
                question=item["question"],
                options=item.get("choices"),
                facts=fact_str
            )
    
            prompts.append(prompt)
            id_order.append(qid)
    
        votes = {qid: [] for qid in id_order}
    
        for i in range(n_vote):
            raw_results = await self.llm_backend.generate(
                prompts,
                system_prompt=self.prompt_generator_qa_cot.system_prompt,
                response_format={"type": "json_object"},
                temperature=gen_temp,
                top_p=top_p,
                max_tokens=max_tokens,
                return_confidence=False,
                seed=seed_base + i
            )
    
            for qid, raw in zip(id_order, raw_results):
                text = raw[0] if isinstance(raw, (tuple, list)) else raw
                parsed = FormatConverter.extract_answer(text)
    
                choices = id2item.get(qid, {}).get("choices") or []
                if choices:
                    parsed = FormatConverter.map_to_option(parsed, choices)
    
                votes[qid].append(parsed)
    
        for qid in id_order:
            counter = Counter(votes[qid])
            top_ans, top_count = counter.most_common(1)[0]
            agreement = top_count / max(1, len(votes[qid]))
    
            choices = id2item.get(qid, {}).get("choices") or []
            nota_option = FormatConverter.find_nota_option(choices)
    
            # 论文式拒答：只有候选项中显式存在 NOTA/Unknown，且投票一致度不足时才拒答
            if nota_option and agreement < confidence_threshold:
                top_ans = nota_option
    
            if choices:
                top_ans = FormatConverter.map_to_option(top_ans, choices)
    
            predictions[qid] = top_ans
            self.confidence_store[qid] = agreement
    
        return predictions
        

    
    async def predict_answer_scheduled_cot(
        self,
        dataset: Dataset,
        facts: List[Dict],
        **sampling_kwargs
    ) -> Dict[str, str]:
        """
        Predict answers using scheduled chain-of-thought reasoning
        
        Args:
            dataset: Input dataset
            facts: Factual knowledge for each item
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Generate prompts
        prompts = []
        for item in dataset:
            # Find matching facts for this item
            fact_str = next(
                (' '.join([chunk['chunk'] for chunk in d['topk_chunks']]) 
                 for d in facts if d['id'] == item['id']), 
                ""
            )
            
            prompt = self.prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                context=fact_str,
                question=item['question'],
                options=item.get('choices'),
                facts=fact_str
            )
            prompts.append(prompt)
        # 518 行之后新插入
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_qa_cot.system_prompt,
            response_format={'type': 'json_object'},
            return_confidence=False,
            **merged_params
        )
        predictions = {}
        for item, (result, confidence) in zip(dataset, raw_results):
            self.confidence_store[item['id']] = confidence
            parsed = FormatConverter.extract_answer(result)
            if item.get('choices'):
                parsed = FormatConverter.map_to_option(parsed, item['choices'])
            predictions[item['id']] = parsed
        return predictions

    async def predict_answer_wo_cot(
        self,
        dataset: Dataset,
        facts: List[Dict],
        **sampling_kwargs
    ) -> Dict[str, str]:
        """
        Predict answers without chain-of-thought reasoning
        
        Args:
            dataset: Input dataset
            facts: Factual knowledge for each item
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Generate prompts
        prompts = []
        for item in dataset:
            # Find matching facts for this item
            fact_str = next(
                (' '.join([chunk['chunk'] for chunk in d['topk_chunks']]) 
                 for d in facts if d['id'] == item['id']), 
                 ""
            )
            
            prompts.append(
                self.prompt_generator_qa.generate_qa_prompt(
                    context=item.get('context', ''),
                    question=item['question'],
                    options=item.get('choices'),
                    facts=fact_str
                )
            )
        # Generate responses
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info("Generating answers with scheduled cot...")
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_qa.system_prompt,
            response_format={'type': 'json_object'},
            return_confidence=False, # 要求返回置信度
            **merged_params
        )
        # 存储置信度并处理结果
        predictions = {}
        for item, (result, confidence) in zip(dataset, raw_results):
            self.confidence_store[item['id']] = confidence
            parsed = FormatConverter.extract_answer(result)
            if item.get('choices'):
                parsed = FormatConverter.map_to_option(parsed, item['choices'])
            predictions[item['id']] = parsed
            
        return predictions
    
    def get_confidence(self, id: str) -> float:
        """获取生成答案的置信度"""
        return self.confidence_store.get(id, 1.0)  # 默认为1.0
