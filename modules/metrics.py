"""
Metrics module for CT2Rep.
Computes NLG metrics for report generation evaluation.
"""

import numpy as np
from collections import Counter


def compute_scores(gts, res):
    """
    Compute NLG metrics between ground truth and generated reports.
    
    Args:
        gts: List of ground truth reports
        res: List of generated reports
        
    Returns:
        Dictionary of metrics
    """
    if len(gts) != len(res):
        min_len = min(len(gts), len(res))
        gts = gts[:min_len]
        res = res[:min_len]
    
    if len(gts) == 0:
        return {
            'BLEU_1': 0.0, 'BLEU_2': 0.0, 'BLEU_3': 0.0, 'BLEU_4': 0.0,
            'METEOR': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0
        }
    
    metrics = {}
    
    # BLEU scores
    bleu = compute_bleu(gts, res)
    metrics.update(bleu)
    
    # METEOR
    metrics['METEOR'] = compute_meteor(gts, res)
    
    # ROUGE-L
    metrics['ROUGE_L'] = compute_rouge_l(gts, res)
    
    # CIDEr
    metrics['CIDEr'] = compute_cider(gts, res)
    
    return metrics


def compute_bleu(references, hypotheses, max_n=4):
    """Compute BLEU-1 through BLEU-4 scores."""
    
    def get_ngrams(text, n):
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = list(text)
        
        if len(tokens) < n:
            return Counter()
        
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    
    def brevity_penalty(ref_len, hyp_len):
        if hyp_len >= ref_len:
            return 1.0
        return np.exp(1 - ref_len / max(hyp_len, 1))
    
    scores = {f'BLEU_{n}': 0.0 for n in range(1, max_n + 1)}
    
    for n in range(1, max_n + 1):
        total_clipped = 0
        total_candidate = 0
        total_ref_len = 0
        total_hyp_len = 0
        
        for ref, hyp in zip(references, hypotheses):
            if not ref or not hyp:
                continue
                
            ref_ngrams = get_ngrams(ref, n)
            hyp_ngrams = get_ngrams(hyp, n)
            
            clipped = {}
            for ngram, count in hyp_ngrams.items():
                clipped[ngram] = min(count, ref_ngrams.get(ngram, 0))
            
            total_clipped += sum(clipped.values())
            total_candidate += sum(hyp_ngrams.values())
            
            ref_tokens = ref.lower().split() if isinstance(ref, str) else list(ref)
            hyp_tokens = hyp.lower().split() if isinstance(hyp, str) else list(hyp)
            total_ref_len += len(ref_tokens)
            total_hyp_len += len(hyp_tokens)
        
        if total_candidate > 0:
            precision = total_clipped / total_candidate
            bp = brevity_penalty(total_ref_len, total_hyp_len)
            scores[f'BLEU_{n}'] = bp * precision
    
    return scores


def compute_meteor(references, hypotheses):
    """Compute simplified METEOR score."""
    
    def get_tokens(text):
        if isinstance(text, str):
            return set(text.lower().split())
        return set(text)
    
    scores = []
    
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            scores.append(0.0)
            continue
            
        ref_tokens = get_tokens(ref)
        hyp_tokens = get_tokens(hyp)
        
        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            scores.append(0.0)
            continue
        
        matches = len(ref_tokens & hyp_tokens)
        
        precision = matches / len(hyp_tokens)
        recall = matches / len(ref_tokens)
        
        if precision + recall > 0:
            f_score = (10 * precision * recall) / (9 * precision + recall)
        else:
            f_score = 0.0
        
        scores.append(f_score)
    
    return np.mean(scores) if scores else 0.0


def compute_rouge_l(references, hypotheses):
    """Compute ROUGE-L score based on Longest Common Subsequence."""
    
    def lcs_length(x, y):
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        
        table = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    table[i][j] = table[i-1][j-1] + 1
                else:
                    table[i][j] = max(table[i-1][j], table[i][j-1])
        
        return table[m][n]
    
    def get_tokens(text):
        if isinstance(text, str):
            return text.lower().split()
        return list(text)
    
    scores = []
    
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            scores.append(0.0)
            continue
            
        ref_tokens = get_tokens(ref)
        hyp_tokens = get_tokens(hyp)
        
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            scores.append(0.0)
            continue
        
        lcs_len = lcs_length(ref_tokens, hyp_tokens)
        
        precision = lcs_len / len(hyp_tokens)
        recall = lcs_len / len(ref_tokens)
        
        if precision + recall > 0:
            f_score = (2 * precision * recall) / (precision + recall)
        else:
            f_score = 0.0
        
        scores.append(f_score)
    
    return np.mean(scores) if scores else 0.0


def compute_cider(references, hypotheses):
    """Compute simplified CIDEr score."""
    
    def get_ngrams(text, n=4):
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = list(text)
        
        ngrams = Counter()
        for i in range(1, n + 1):
            for j in range(len(tokens) - i + 1):
                ngrams[tuple(tokens[j:j+i])] += 1
        
        return ngrams
    
    def cosine_similarity(vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        
        intersection = set(vec1.keys()) & set(vec2.keys())
        
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        
        sum1 = sum(v ** 2 for v in vec1.values())
        sum2 = sum(v ** 2 for v in vec2.values())
        
        denominator = np.sqrt(sum1) * np.sqrt(sum2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    scores = []
    
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            scores.append(0.0)
            continue
            
        ref_ngrams = get_ngrams(ref)
        hyp_ngrams = get_ngrams(hyp)
        
        score = cosine_similarity(ref_ngrams, hyp_ngrams)
        scores.append(score * 10)  # Scale to match standard CIDEr
    
    return np.mean(scores) if scores else 0.0