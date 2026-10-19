from collections import Counter
import inflect
import string
import re

ENGINE = inflect.engine()

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def acc_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if type(ground_truths) != list:
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)

        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def normalize_answer(s,p=ENGINE):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())
        
    # 新增：统一千位分隔符格式
    def unify_thousands(text):
        # 处理空格分隔的千位表示法 (10 000 → 10000)
        text = re.sub(r'(\d)\s+(\d{3}\b)', r'\1\2', text)
        # 处理逗号分隔的千位表示法 (10,000 → 10000)
        text = re.sub(r'(\d),(\d{3}\b)', r'\1\2', text)
        return text
    
    s = unify_thousands(s)
    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    def convert_numbers_to_words(text):
        words = text.split()
        result = []
        for word in words:
            if word.isdigit() and int(word) < 100:
                word_in_words = p.number_to_words(int(word))
                result.append(word_in_words)
            else:
                result.append(word)
        return ' '.join(result)

    return white_space_fix(remove_articles(handle_punc(convert_numbers_to_words(lower(replace_underscore(s)))))).strip()

def get_ground_truths(answer):
    # 兼容字符串或列表
    if isinstance(answer, list):
        return [normalize_answer(a) for a in answer]
    return [normalize_answer(answer)]

def is_nota(text: str) -> bool:
    t = normalize_answer(text)
    NOTA_SET = {
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
    return t in NOTA_SET

def nota_aware_em(prediction, ground_truths):
    """
    EM 的 NOTA 友好版：
    - 预测为 NOTA 时，仅当标准答案也属 NOTA/未知集合才记 1，否则 0
    - 否则回到普通 EM
    """
    p = normalize_answer(prediction)
    gts = ground_truths if isinstance(ground_truths, list) else [ground_truths]
    gts = [normalize_answer(x) for x in gts]
    if is_nota(p):
        return 1 if any(is_nota(gt) for gt in gts) else 0
    return 1 if any(p == gt for gt in gts) else 0

def nota_aware_f1(prediction, ground_truths):
    """
    F1 的 NOTA 友好版：
    - 预测为 NOTA 且 GT 为 NOTA → 1
    - 预测为 NOTA 且 GT 为实体/文本 → 0
    - 其余与原 f1_score 一致，对每个 GT 取最大值
    """
    p = normalize_answer(prediction)
    gts = ground_truths if isinstance(ground_truths, list) else [ground_truths]
    gts = [normalize_answer(x) for x in gts]
    if is_nota(p):
        return 1.0 if any(is_nota(gt) for gt in gts) else 0.0
    return max(f1_score(p, gt) for gt in gts)
