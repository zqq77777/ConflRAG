import asyncio
import os
from typing import Dict, List, Optional
from datasets import Dataset
from tqdm import tqdm
from .evaluate import (
    exact_match_score, 
    acc_score, 
    f1_score, 
    metric_max_over_ground_truths
)
from .modules import (
    FactMiningModule,
    ContextualAlignmentModule,
    SelfThinkModule
)
from .util.format_util import FormatConverter

# def detect_dataset_type(dataset: Dataset) -> str:
#     """自动检测数据集类型"""
#     counterfactual_count = 0
#     sample_size = min(100, len(dataset))
    
#     for item in dataset.select(range(sample_size)):
#         question = item.get('question', '').lower()
#         answer = item.get('answer', '').lower()
        
#         # 检查是否包含反科学答案
#         if any(pattern in answer for pattern in [
#             "gravity.*stronger", "moon.*sun", "dependent.*variable",
#             "100.*freeze", "0.*boil"
#         ]):
#             counterfactual_count += 1
    
#     # 如果超过20%是反事实问题，认为是对抗性数据集
#     if counterfactual_count / sample_size > 0.2:
#         return "adversarial"
#     return "normal" 
def detect_dataset_type(dataset: Dataset) -> str:
    return "normal"
class ConflRAG:
    """Faithful Retrieval-Augmented Generation pipeline"""
    
    def __init__(
        self,
        backend_type: str,
        model_name: str,
        similarity_model: str,
        mining_sampling_params: Optional[Dict] = None,
        generation_sampling_params: Optional[Dict] = None,
        confidence_threshold: float = 0.35, 
        dataset_type="normal",
        **backend_config
    ):
        """
        Initialize the ConflRAG pipeline
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory)
            model_name: Name of the model to use
            similarity_model: Sentence similarity model name
            mining_sampling_params: Parameters for fact mining generation
            generation_sampling_params: Parameters for answer generation
            backend_config: Backend-specific configuration parameters
        """
        self.backend_type = backend_type
        self.dataset_type = dataset_type  # "normal" 或 "adversarial"
        self.model_name = model_name
        self.similarity_model = similarity_model
        self.confidence_threshold = confidence_threshold
        # Set default sampling parameters if not provided
        self.mining_sampling_params = mining_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        self.generation_sampling_params = generation_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        # Initialize modules
        self.fact_mining_module = FactMiningModule(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        
        self.contextual_alignment_module = ContextualAlignmentModule(
            similarity_model=similarity_model
        )
        
        self.self_think_module = SelfThinkModule(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
    
    async def get_self_facts(
        self, 
        dataset: Dataset, 
        fact_mining_type: str = "default",
        **mining_params
    ) -> List[Dict]:
        """
        Generate self-consistent facts for the dataset
        
        Args:
            dataset: Input dataset
            fact_mining_type: Type of fact mining ("default")
            mining_params: Override parameters for fact mining
            
        Returns:
            List of self-consistent facts dictionaries
        """
        # Use provided parameters or defaults
        params = {**self.mining_sampling_params, **mining_params}
        
        if fact_mining_type == "default":
            # Generate initial knowledge facts
            knowledges = await self.fact_mining_module.generate_knowledges(
                dataset, **params
            )
            
            # Generate self-context using the knowledge
            self_context = await self.fact_mining_module.generate_self_context(
                dataset, knowledges=knowledges, **params
            )
            
            # Extract self-consistent facts from the context
            return await self.fact_mining_module.extract_facts(
                self_context, **params
            )
        else:
            raise ValueError(f"Unsupported fact mining type: {fact_mining_type}")
    
    def get_topk_chunks(
        self, 
        dataset: Dataset, 
        self_facts: List[Dict],
        sent_topk: int = 8,
        chunk_topk: int = 8,
        chunk_size: int = 60
    ) -> List[Dict]:
        """
        Retrieve top-k contextual chunks for each fact
        
        Args:
            dataset: Input dataset
            self_facts: Self-consistent facts
            sent_topk: Number of top sentences to retrieve
            chunk_topk: Number of top chunks to return
            chunk_size: Size of context chunks
            
        Returns:
            List of dictionaries with top-k chunks
        """
        contextual_chunks = self.contextual_alignment_module.get_contextual_chunks(
            dataset, self_facts, sent_topk, chunk_topk, chunk_size
        )
        return contextual_chunks
            
    async def get_predictions(
        self,
        dataset: Dataset, 
        facts: List[Dict],
        generation_type: str = "normal_cot",
        **generation_params
    ) -> Dict[str, str]:
        """
        Generate predictions for the dataset
        """
        # 修复：移除错误的system_prompt引用
        params = {**self.generation_sampling_params, **generation_params}      
        # 如果是对抗性数据集，通过参数传递特殊指令
        if self.dataset_type == "adversarial":
            params['adversarial_mode'] = True
                                               
        raw_predictions = {}
        if generation_type == "normal_cot":
            params['response_format'] = {"type": "json_object"}
            raw_predictions = await self.self_think_module.predict_answer_normal_cot(
                dataset, facts, **params
            )            
        elif generation_type == "scheduled_cot":
            params['response_format'] = {"type": "json_object"}
            raw_predictions = await self.self_think_module.predict_answer_scheduled_cot(
                dataset, facts, **params
            )
        elif generation_type == "wo_cot":
            raw_predictions = await self.self_think_module.predict_answer_wo_cot(
                dataset, facts, **params
            )
        else:
            raise ValueError(f"Unsupported generation type: {generation_type}")
            
        # ===== 新增后处理和置信度过滤 =====
        processed_predictions = {}
        # 选项映射需要用到数据集中的 choices
        id2options = {item['id']: item.get('choices') for item in dataset}
        
        for id_, pred in raw_predictions.items():
            # 1. 提取答案（如果是JSON格式）
            if generation_type in ["normal_cot", "scheduled_cot", "wo_cot"]:
                try:
                    pred = FormatConverter.extract_answer(pred)
                except:
                    pred = "I don't know"
            
            # 2. 置信度过滤（如果后端支持）
            # if hasattr(self.self_think_module, "get_confidence"):
            #     confidence = self.self_think_module.get_confidence(id_)
            #     if confidence < self.confidence_threshold:
            #         pred = "I don't know"
            # # 2.5 选项映射（将预测强制对齐到原始选项文本）
            # options = id2options.get(id_)
            # if options and pred != "I don't know":
            #     pred = FormatConverter.map_to_option(pred, options)
            # 2. 置信度过滤（如果后端支持）
            # if hasattr(self.self_think_module, "get_confidence"):
            #     confidence = self.self_think_module.get_confidence(id_)
            #     # 对于无NOTA数据集，禁用或降低阈值
            #     if os.environ.get("DISABLE_NOTA") == "true":
            #         pass  # 不做拒答处理
            #     elif confidence < 0.05:  # 极低置信度才拒答
            #         pred = "I don't know"
            # if hasattr(self.self_think_module, "get_confidence"):
            #     confidence = self.self_think_module.get_confidence(id_)
            #      # 仅当选项中确实包含 NOTA/未知类选项时才考虑拒答
            #     opts = id2options.get(id_) or []
            #     nota_signals = {"i don't know","unknown","no answer","nota","not given","无法判断","不知道","不确定","未提及","以上皆非"}
            #     has_nota = any(FormatConverter.normalize_answer(o) in nota_signals for o in opts)
            #     if has_nota and confidence < 0.05:
            #         pred = "I don't know"
            # 2.5 选项映射（最终对齐到原始选项文本）
            # options = id2options.get(id_)
            # if options and pred != "I don't know":
            #     pred = FormatConverter.map_to_option(pred, options)
            options = id2options.get(id_)
            if options:
                pred = FormatConverter.map_to_option(pred, options)
            # 3. 不再二次标准化，避免破坏与 GT 的精确包含关系（评测端会统一 normalize）
            processed_predictions[id_] = pred

        return processed_predictions    
        
    def evaluate(
        self, 
        dataset: Dataset, 
        predictions: Dict[str, str],
        cot_format: bool = False,
        detailed_output: bool = False,
        analyze_errors: bool = False
    ) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Args:
            dataset: Input dataset with ground truth
            predictions: Generated predictions
            detailed_output: Whether to include per-item details
            
        Returns:
            Evaluation results dictionary
        """
        prediction_details = []
        total_em = total_acc = total_f1 = 0
        num_items = 0
        
        for item in tqdm(dataset, desc="Evaluating"):
            prediction = predictions.get(item['id'], "")
            # if prediction is in JSON format, extract the 'answer' field
            if cot_format:
                prediction = FormatConverter.extract_answer(prediction)
            ground_truth = item['answer']
            
            # Calculate metrics
            em_score = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truth)
            acc_score_val = metric_max_over_ground_truths(
                acc_score, prediction, ground_truth)
            f1_score_val = metric_max_over_ground_truths(
                f1_score, prediction, ground_truth)
            
            # Accumulate totals
            total_em += em_score
            total_acc += acc_score_val
            total_f1 += f1_score_val
            num_items += 1
            
            # Store details if requested
            if detailed_output:
                prediction_details.append({
                    "id": item['id'],
                    "question": item['question'],
                    "answer": ground_truth,
                    "prediction": prediction,
                    "exact_match": em_score,
                    "acc": acc_score_val,
                    "f1": f1_score_val
                })
        
        # Calculate averages
        avg_em = 100.0 * total_em / num_items if num_items > 0 else 0
        avg_acc = 100.0 * total_acc / num_items if num_items > 0 else 0
        avg_f1 = 100.0 * total_f1 / num_items if num_items > 0 else 0
        
        # 准备结果
        result = {
            "num_items": num_items,
            "exact_match": avg_em,
            "acc": avg_acc,
            "f1": avg_f1
        }
        
        if detailed_output:
            error_report = self._analyze_errors(dataset, prediction_details)
            result["details"] = prediction_details
            if error_report:
                result["error_analysis"] = error_report            
        return result
    def _analyze_errors(self, dataset: Dataset, details: List[Dict]) -> Dict:
        """错误分析闭环"""
        errors = {
            'incomplete': [],    # 不完整回答
            'confidence': [],    # 低置信度回答
            'format_mismatch': [] # 格式不匹配
        }
        
        for detail in details:
            pred = detail['prediction']
            truth = detail['answer']
            
            # 1. 检测不完整回答
            if "I don" in pred and "I don't know" not in pred:
                errors['incomplete'].append(detail['id'])
            
            # 2. 检测格式不匹配（数值/货币）
            num_chars = ['$', ',', 'K', 'M']
            if any(c in pred for c in num_chars) and not any(c in truth for c in num_chars):
                errors['format_mismatch'].append(detail['id'])
            
            # 3. 检测低置信度（需要后端支持）
            if hasattr(self.self_think_module, "get_confidence"):
                conf = self.self_think_module.get_confidence(detail['id'])
                if conf < self.confidence_threshold and detail['f1'] == 0:
                    errors['confidence'].append(detail['id'])
        
        return errors