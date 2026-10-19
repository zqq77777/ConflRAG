import json
import re
import os
import inflect
import string
import difflib
import logging
from typing import Optional, Dict,Any, List, Union
from collections import Counter
# --- 在第8行之后新增 ---
from functools import lru_cache
from typing import Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FormatConverter")

# ---- SBERT 单例：避免重复加载并固定设备 ----
_SBERT_SINGLETON = None
def _get_sbert():
    global _SBERT_SINGLETON
    if _SBERT_SINGLETON is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get("SBERT_MODEL_PATH", "all-MiniLM-L6-v2")
            device = os.environ.get("SBERT_DEVICE", None)  # 如需强制GPU: export SBERT_DEVICE=cuda
            _SBERT_SINGLETON = SentenceTransformer(model_name, device=device)
        except Exception:
            _SBERT_SINGLETON = None
    return _SBERT_SINGLETON

class FormatConverter:

    @staticmethod
    def robust_json_extract(text: str) -> dict:
        """
        增强版JSON解析方法，解决常见的输出解析失败问题：
        - 不完整JSON（如被截断）
        - 语法错误（引号缺失、未关闭括号等）
        - 代码块包装（```json ...```）
        - 文本前缀/后缀干扰
        
        支持多种常见错误模式的自动修复
        """
        # 步骤1：预处理 - 清理常见包装格式
        cleaned = FormatConverter._preprocess_json(text)
        
        # 步骤2：定位最可能的JSON起始和结束位置
        json_candidate = FormatConverter._locate_json_candidate(cleaned)
        
        # 步骤3：尝试标准解析
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            # 步骤4：常见错误模式修复
            repaired = FormatConverter._autofix_json(json_candidate, str(e))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                # 步骤5：最终回退 - 结构化信息提取
                return FormatConverter._fallback_extraction(text)
    
    @staticmethod
    def _preprocess_json(raw_text: str) -> str:
        """清理JSON周围的非结构化内容"""
        # 移除代码块标记（```json...```）
        cleaned = re.sub(r'```json\s*([\s\S]*?)\s*```', r'\1', raw_text)
        
        # 移除常见的无关前缀
        patterns = [
            r'^\s*(Here is the (JSON|answer)|Result:|Output:)',
            r'^\s*[{$$\s]*',  # 行首多余的空格和括号
            r'[\s}$$]+\s*$'  # 行尾多余的空格和括号
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    
    @staticmethod
    def extract_atomic_facts(text: str, max_facts: int = 7) -> List[str]:
        """
        轻量级原子事实抽取：按句切分 -> 去连接词/冗余前缀 -> 截断
        无外部依赖，作为 Self-Fact Mining 的兜底
        """
        if not text:
            return []
        # 复用已有清洗，见本文件的 clean_output/normalize_prediction
        try:
            text = FormatConverter.clean_output(text)
        except Exception:
            text = str(text)
        # 简单分句
        segs = re.split(r'[。.!?；;]\s*', text)
        facts = []
        for s in segs:
            s = s.strip()
            if not s or len(s) < 4:
                continue
            # 去掉无信息前缀
            s = re.sub(r'^(因此|所以|综上|结论是|我们可以推断|Based on|Therefore|Hence)[：:，,]?\s*', '', s, flags=re.IGNORECASE)
            facts.append(' '.join(s.split()[:32]))
            if len(facts) >= max_facts:
                break
        return facts

    @staticmethod
    def context_to_sentences(context: Union[str, List[Dict]]) -> List[str]:
        """
        将检索上下文（字符串或 [{'chunk': ...}, ...]）规范为句子列表
        """
        sents = []
        if isinstance(context, str):
            pool = [context]
        elif isinstance(context, list):
            # 兼容 topk_chunks 结构： [{'chunk': '...'}, ...]
            pool = []
            for x in context:
                if isinstance(x, dict) and 'chunk' in x:
                    pool.append(x['chunk'])
                else:
                    pool.append(str(x))
        else:
            pool = [str(context)]
        for block in pool:
            for s in re.split(r'[。.!?；;]\s*', str(block)):
                s = s.strip()
                if s:
                    sents.append(s)
        return sents[:256]  # 防止过长
    @staticmethod
    def select_evidence_sentences(facts: List[str], context: Union[str, List[Dict]], k: int = 10) -> List[str]:
        """按与自检事实的最大相似度选 Top-K 证据句"""
        sents = FormatConverter.context_to_sentences(context)
        if not sents:
            return []
        if not facts:
            return sents[:k]
        try:
            model = _get_sbert()
            if model:
                from sentence_transformers import util as st_util
                e_f = model.encode(facts, convert_to_tensor=True, show_progress_bar=False)
                e_s = model.encode(sents, convert_to_tensor=True, show_progress_bar=False)
                sims = st_util.cos_sim(e_f, e_s).max(dim=0).values  # 每句对所有fact的最大相似度
                topk = min(k, len(sents))
                idx = sims.topk(topk).indices.tolist()
                return [sents[i] for i in idx]
        except Exception:
            pass
        return sents[:k]

    @staticmethod
    def fact_context_coverage(facts: List[str], context: Union[str, List[Dict]], use_semantic: bool = True) -> float:
        """
        计算事实与上下文的覆盖率（每个fact在上下文句子上的最大相似度的平均值，0~1）
        - 优先用 all-MiniLM-L6-v2/BGE 等语义相似度
        - 失败则回退到 token-F1
        """
        sents = FormatConverter.context_to_sentences(context)
        if not facts or not sents:
            return 0.0

        def token_f1(a: str, b: str) -> float:
            at = FormatConverter.normalize_prediction(a).split()
            bt = FormatConverter.normalize_prediction(b).split()
            inter = (Counter(at) & Counter(bt))
            if not inter:
                return 0.0
            p = sum(inter.values()) / (len(at) or 1)
            r = sum(inter.values()) / (len(bt) or 1)
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if use_semantic:
            try:
                from sentence_transformers import SentenceTransformer, util as st_util
                model_name = os.environ.get("SBERT_MODEL_PATH", "all-MiniLM-L6-v2")
                _sbert = _get_sbert()
                if _sbert is None:
                    raise RuntimeError("SBERT model not available")
                emb_f = _sbert.encode(facts, convert_to_tensor=True, show_progress_bar=False)
                emb_s = _sbert.encode(sents, convert_to_tensor=True, show_progress_bar=False)
                sims = st_util.cos_sim(emb_f, emb_s).cpu().numpy()  # [nf, ns]
                best = sims.max(axis=1).tolist()
                return float(sum(best) / len(best))
            except Exception:
                pass

        # 回退：token-F1
        best = []
        for f in facts:
            best.append(max((token_f1(f, s) for s in sents), default=0.0))
        return float(sum(best) / len(best))

    @staticmethod
    def should_refuse_by_coverage(facts: List[str], context: Union[str, List[Dict]], threshold: float = 0.20) -> bool:
        """
        如果覆盖率低于阈值，则建议“NOTA/拒答”
        阈值经验取自 ConflRAG 思路（先挖自事实、再与段落对齐，低相似即不答）与 Self-RAG 的自反检查思想。
        """
        score = FormatConverter.fact_context_coverage(facts, context)
        return score < threshold
    
    @staticmethod
    def _locate_json_candidate(text: str) -> str:
        """在文本中找到最可能的JSON字符串"""
        # 尝试找到完整的JSON对象
        if re.search(r'^\s*{.*}\s*$', text, re.DOTALL):
            return text
        
        # 查找JSON起始位置
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        # 处理未闭合的JSON
        if start_idx != -1 and end_idx == -1:
            # 在行末添加闭合大括号
            return text[start_idx:] + '}'
        elif start_idx == -1 and end_idx != -1:
            # 在行首添加起始大括号
            return '{' + text[:end_idx+1]
        elif start_idx == -1 and end_idx == -1:
            # 完全找不到括号 - 尝试包装为JSON
            return FormatConverter._wrap_as_json(text)
        else:
            # 提取完整的JSON片段
            return text[start_idx:end_idx+1]
    
    @staticmethod
    def _wrap_as_json(text: str) -> str:
        """
        将无结构文本包装为JSON格式：
        "answer": {text}
        """
        # 尝试检测已有键值
        if re.search(r'["\']?\w+["\']?\s*:', text):
            return '{' + text + '}'
        
        # 如果包含自然语言答案
        if ':' in text:
            parts = text.split(':', 1)
            key = parts[0].strip().strip('"\'')
            value = parts[1].strip().strip('"\'')
            
            return json.dumps({key: value}, ensure_ascii=False)
        
        # 默认返回整个文本作为answer值
        return json.dumps({"answer": text}, ensure_ascii=False)
    @staticmethod
    def is_nota_text(text: str) -> bool:
        t = FormatConverter.normalize_answer(str(text))
        nota_set = {
            "i don t know",
            "i dont know",
            "unknown",
            "no answer",
            "none",
            "nota",
            "not applicable",
            "n a",
            "no information",
            "cannot answer",
            "cannot determine",
            "not enough information",
            "not given",
            "none of above",
            "none of the above",
            "无法判断",
            "不知道",
            "不确定",
            "未提及",
            "以上皆非"
        }
        return t in nota_set
    
    
    @staticmethod
    def find_nota_option(options):
        if not options:
            return None
    
        for opt in options:
            if FormatConverter.is_nota_text(opt):
                return opt
    
        return None    
    @staticmethod
    def _autofix_json(json_str: str, error_message: str) -> str:
        """基于错误类型的自动化修复"""
        # 处理键值缺少引号
        if 'Expecting property name enclosed in double quotes' in error_message:
            return re.sub(
                r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', 
                r'\1"\2"\3', 
                json_str
            )
        
        # 处理单引号问题
        if 'Single quotes' in error_message:
            return json_str.replace("'", '"')
        
        # 处理尾部逗号问题
        if "Expecting property name" in error_message and "line 1 column" in error_message:
            return re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # 处理布尔值问题
        if "true" in json_str.lower() or "false" in json_str.lower():
            return re.sub(
                r':\s*(True|TRUE|true)\b', 
                ': true', 
                re.sub(r':\s*(False|FALSE|false)\b', ': false', json_str)
            )
        
        # 处理未闭合大括号
        if "Expecting value" in error_message and not json_str.endswith('}'):
            return json_str + '}'
        
        return json_str
    
    @staticmethod
    def _fallback_extraction(text: str) -> dict:
        """
        当所有JSON解析尝试失败时的最终回退策略
        使用启发式规则提取键值对
        """
        # 尝试提取键值对模式
        pattern = r'["\']?([^"\':]+)["\']?\s*:\s*["\']?([^"\']*)["\']?'
        pairs = re.findall(pattern, text)
        
        # 构建结果字典
        result = {}
        for key, value in pairs:
            normalized_key = key.strip().lower().replace(' ', '_')
            result[normalized_key] = value.strip()
        
        # 如果没有找到键值对
        if not result:
            # 尝试搜索答案提示词
            answer_match = re.search(
                r'(?:answer|答案|response)[:\s]\s*([^\n]+)', 
                text, 
                re.IGNORECASE
            )
            if answer_match:
                result['answer'] = answer_match.group(1).strip()
            else:
                # 最终回退：整个文本作为答案
                result['extracted_content'] = text[:300].strip() + ('...' if len(text) > 300 else '')
        
        return result

    @staticmethod
    def is_knowledge_boundary_question(question: str) -> bool:
        """检测是否超出知识边界的问题"""
        boundary_keywords = {
            "this week", "recent", "currently", "current", "today", "now",
            "2022", "2023", "2024", "2025", "current year",
            "latest", "updated", "recently", "newest", "announced"
        }
        return any(kw in question.lower() for kw in boundary_keywords)

    @staticmethod
    def convert_text2fact(text: str, output_file: Optional[str] = None) -> Dict[str, str]:
        """将包含<seq>标签的文本转换为键值对字典"""
        matches = re.findall(r'<seq>(.*?)</seq>', text, re.DOTALL)
        result = {f"seq_{i+1}": match.strip() for i, match in enumerate(matches)}
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as file:
                for key, value in result.items():
                    json.dump({key: value}, file, ensure_ascii=False)
                    file.write('\n')
        
        return result

    @staticmethod
    def remove_brackets_and_content(input_str: str) -> str:
        """移除所有<...>和[...]格式的内容"""
        cleaned_str = re.sub(r'\<.*?\>', '', input_str)
        cleaned_str = re.sub(r'\$\$.*?\$\$', '', cleaned_str)
        return ' '.join(cleaned_str.split())
    @staticmethod
    def normalize_answer(s: str) -> str:
        """文本规范化处理：小写、去标点、移除冠词等"""
        p = inflect.engine()
        
        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def handle_punc(text: str) -> str:
            exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text: str) -> str:
            return text.lower()

        def replace_underscore(text: str) -> str:
            return text.replace('_', ' ')
        
        def convert_numbers_to_words(text: str) -> str:
            words = text.split()
            result = []
            for word in words:
                if word.isdigit() and int(word) < 100:
                    try:
                        word_in_words = p.number_to_words(int(word))
                        result.append(word_in_words)
                    except:
                        result.append(word)
                else:
                    result.append(word)
            return ' '.join(result)
        
        # 应用所有规范化步骤
        s = replace_underscore(s)
        s = lower(s)
        # s = convert_numbers_to_words(s)
        s = handle_punc(s)
        s = remove_articles(s)
        s = white_space_fix(s)
        
        return s.strip()

    @staticmethod
    def convert_context(context: str, chunks: List[str]) -> str:
        """在上下文中标记重要事实"""
        for chunk in chunks:
            context = re.sub(re.escape(chunk), f"[important facts: {chunk}]", context)
        return context
        
    @staticmethod
    def normalize_prediction(text: Union[str, None]) -> str:
        """统一答案格式并修复常见问题"""
        if text is None:
            return "I don't know"
        
        text = str(text).strip()
        
        # 特殊值处理
        if text.lower() in ["null", "none", "no information", "unknown", "nan", ""]:
            return "I don't know"
        
        # 修复不完整回答
        if text.lower().startswith("i don") and "i don't know" not in text.lower():
            return "I don't know"
        
        # 处理带引号的答案（如“Ted Lasso”）
        if text.startswith('“') and text.endswith('”'):
            text = text[1:-1]
        elif text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # 统一百分比格式
        if '%' in text:
            # 移除空格和多余字符
            text = re.sub(r'[^\d.%]', '', text)
            # 确保百分比格式一致
            if '.' in text:
                text = f"{float(text.replace('%', '')):.1f}%"
            else:
                text = f"{int(text.replace('%', ''))}%"
                
        # 统一货币格式
        if '$' in text:
            # 提取数字部分
            amount = re.search(r'\d[\d,.]*', text)
            if amount:
                # 移除逗号，转换为整数
                amount_str = amount.group(0).replace(',', '')
                if '.' in amount_str:
                    text = f"${float(amount_str):.2f}"
                else:
                    text = f"${int(amount_str)}"
                    
        # 统一年份格式
        if re.match(r'^\d{4}$', text) and 1000 <= int(text) <= 2100:
            return text
            
        # 统一温度格式
        if '°C' in text or '°F' in text:
            # 提取温度值
            temp_match = re.search(r'(\d+\.?\d*)\s*°?[CF]', text)
            if temp_match:
                temp_value = float(temp_match.group(1))
                return f"{temp_value:.1f}°C" if 'C' in text else f"{temp_value:.1f}°F"
                
        return text
    @staticmethod
    def clean_output(text: str) -> str:
        """
        新增方法：清洗模型输出中的无关字符和格式污染
        不影响原有功能，仅作为新增功能
        """
        if not text:
            return text
        
        # 1. 移除JSON包装格式
        patterns_to_remove = [
            r'```json\s*',          # JSON代码块开始
            r'\s*```',               # 代码块结束
            r'^{\s*"extracted_content":\s*"',  # 提取内容包装
            r'{\s*"prediction":\s*"', # 预测键包装
            r'{\s*"output":\s*"'     # 输出键包装
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            text = re.sub(r'["\']?\s*}\s*$', '', text)  # 移除尾部JSON标记
        
        # 2. 移除常见污染字符
        text = re.sub(r'[\n}\\\"\{\}]', ' ', text)
        
        # 3. 移除API提示词前缀
        api_prefixes = [
            "Answer:", "答案：", "Response:", "Output:", 
            "Final Answer:", "Prediction:", "Result:"
        ]
        for prefix in api_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 4. 处理特殊转义字符
        text = re.sub(r'\\n', ' ', text)
        text = re.sub(r'\\"', '"', text)
        text = re.sub(r'\\t', ' ', text)
        
        # 5. 移除多余空格和标点
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[.,;:!?]+$', '', text)  # 移除尾部标点
        
        return text
    @staticmethod
    def extract_answer(prediction: Union[str, dict]) -> str:
        """
        增强版答案提取方法，支持多种格式：
        1. 标准JSON格式
        2. 类似JSON格式但缺少引号
        3. 自然语言格式（包含"Answer:"或"答案:"）
        4. 其他格式的回退策略
        """
        # 特殊值处理
        if prediction in ["null", "None", "no information", "unknown"]:
            return "I don't know"
        
        # 尝试修复常见格式问题
        cleaned_pred = str(prediction)
        
        # 优先检测"I don't know"模式
        if "i don't know" in cleaned_pred.lower():
            return "I don't know"
        
        # 修复常见JSON格式问题
        cleaned_pred = cleaned_pred.replace("：", ":")  # 中文冒号转英文
        cleaned_pred = cleaned_pred.replace("'", "\"")  # 单引号转双引号
        cleaned_pred = re.sub(r'(\w+)\s*:\s*', r'"\1":', cleaned_pred)  # 添加键的引号
        
        # 1. 尝试解析为标准JSON
        try:
            # 使用 robust_json_extract 方法
            data = FormatConverter.robust_json_extract(cleaned_pred)
            # 搜索各种可能的键名
            for key in ["answer", "Answer", "ANSWER", "回答", "答案", "prediction", "output"]:
                if key in data:
                    ans = str(data[key])
                    # 清理答案尾部标点
                    return FormatConverter.normalize_prediction(
                        re.sub(r'[.,;:!?"]+$', '', ans).strip()
                    )
            # 如果没有显式答案键，尝试提取整个文本
            return FormatConverter.normalize_prediction(json.dumps(data))
        except (json.JSONDecodeError, TypeError):
            pass  # 继续尝试其他方法
        
        # 2. 尝试从类似JSON格式中提取
        answer_match = re.search(
            r'["]?(Answer|答案)["]?\s*:\s*["]?([^"\',\}]+)', 
            cleaned_pred, 
            re.IGNORECASE
        )
        if answer_match:
            return FormatConverter.normalize_prediction(answer_match.group(2).strip())
        
        # 3. 尝试提取自然语言答案
        natural_match = re.search(
            r'(?:answer|答案)[:：\s]\s*(.*?)(?:[.,;!?\n]|$)', 
            cleaned_pred, 
            re.IGNORECASE
        )
        if natural_match:
            return FormatConverter.normalize_prediction(natural_match.group(1).strip())
        
        # 4. 尝试提取最后包含大写字母的短语（可能是答案）
        capitalized = re.findall(r'\b([A-Z][\w\s]+)\b', cleaned_pred)
        if capitalized:
            return FormatConverter.normalize_prediction(capitalized[-1].strip())        
        
        # 在返回前添加清洗步骤（兜底：把清洗后的整体文本再规范化一次）
        if isinstance(cleaned_pred, str):
            cleaned = FormatConverter.clean_output(cleaned_pred)
            return FormatConverter.normalize_prediction(cleaned)
        else:
            return FormatConverter.normalize_prediction(str(cleaned_pred))
            
    @staticmethod
    def map_to_option(pred: str, options: List[str]) -> str:
        """
        更鲁棒的选项映射：
        1) 原文精确匹配 → 2) 规范化匹配 → 3) 布尔极性归一（是/否/True/False/正确/错误）
        4) NOTA/未知归一 → 5) 数字/单位“压扁”匹配 → 6) SBERT 语义兜底 → 7) 编辑距离兜底
        """
        import re
        from difflib import SequenceMatcher

        if not options:
            return FormatConverter.normalize_prediction(pred)

        p_raw = (pred or "").strip()

        # 1) 原文大小写无关匹配
        for o in options:
            if p_raw.lower() == o.strip().lower():
                return o

        # 2) 规范化匹配
        norm = FormatConverter.normalize_prediction
        p = norm(p_raw)
        for o in options:
            if norm(o) == p:
                return o

        # 3) 布尔/极性归一
        YES = {"yes","true","正确","是","对","affirmative","y"}
        NO  = {"no","false","错误","否","不","negative","nope"}
        if p in YES | NO:
            target = YES if p in YES else NO
            for o in options:
                on = norm(o)
                if on in target or on in {"是","否","正确","错误","true","false","yes","no"}:
                    return o

        # 4) NOTA/未知归一
        NOTA = {"i don t know","i dont know","unknown","no answer","nota","none of the above",
                "无法判断","不知道","不确定","未提及","以上皆非","not given"}
        if any(x in p for x in NOTA):
            for o in options:
                if any(x in norm(o) for x in NOTA):
                    return o

        # 5) 数字/单位压扁匹配
        squash = lambda s: re.sub(r"[^0-9a-z]", "", norm(s))
        p_num = squash(p_raw)
        if p_num:
            for o in options:
                if squash(o) == p_num:
                    return o

        # 6) 语义相似度兜底（有 SBERT 才用）
        try:
            model = _get_sbert()
            if model:
                from sentence_transformers import util as st_util
                emb = model.encode([p_raw] + list(options), convert_to_tensor=True, show_progress_bar=False)
                sim = st_util.cos_sim(emb[0:1], emb[1:]).cpu().numpy()[0]
                best = int(sim.argmax())
                if float(sim[best]) >= 0.50:
                    return options[best]
        except Exception:
            pass

        # 7) 编辑距离兜底
        scores = [(o, SequenceMatcher(None, norm(o), p).ratio()) for o in options]
        return max(scores, key=lambda x: x[1])[0]

    @staticmethod
    def map_to_choice_letter(pred_text: str, choices) -> str:
        """
        pred_text: 模型原文输出
        choices: 形如 [{"label":"A","text":"..."}, ...]
        return: "A"/"B"/"C"/"D" 或 ""
        """
        import re
        t = pred_text.strip()
        # 1) 直接抓 'Answer": "B"' 或 '选项: C'
        m = re.search(r'["\']?Answer["\']?\s*[:：]\s*["\']?([A-D])["\']?', t, re.I)
        if m: return m.group(1).upper()
    
        # 2) 若输出给出完整选项文本, 做相似度/包含匹配
        norm = lambda s: re.sub(r'\W+', ' ', s).strip().lower()
        tnorm = norm(t)
        best = ("", 0)
        for ch in choices:
            label, text = ch["label"], norm(ch["text"])
            score = (text in tnorm) or (tnorm in text)
            if score and best[1] == 0:
                best = (label, 1)
        if best[1] == 1: return best[0]
    
        # 3) 兜底: 抓 'the answer is C' 类固定短语
        m = re.search(r'\b([A-D])\b', t.upper())
        return m.group(1) if m else ""
