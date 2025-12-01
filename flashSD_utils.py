import torch
import torch.nn.functional as F
from typing import Tuple

class FlashSDVerifier:
    """
    Flash-SD Core Verifier (High-Performance Vectorized Implementation).
    Eliminates Python loops and redundant computations for maximum throughput.
    """
    def __init__(
        self, 
        base_threshold: float = 0.15,
        entropy_scale: float = 1.0,
        lookahead_confidence: float = 0.85
    ):
        self.base_threshold = base_threshold
        self.entropy_scale = entropy_scale
        self.lookahead_confidence = lookahead_confidence

    def verify_and_sample(
        self,
        candidate_input_ids: torch.Tensor,   # [B, Seq_Len]
        candidate_logits: torch.Tensor,      # [B, K, V]
        target_logits: torch.Tensor,         # [B, K+1, V]
        candidate_length: int,
        is_done_candidate: bool,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, int, torch.Tensor, float, float]:
        
        # 1. Pre-compute Probabilities (Avoid redundant Softmax)
        # Slicing target logits to match candidate length K
        # target_logits shape: [B, K+1, V]
        # p_logits (for verification): [B, K, V]
        p_logits = target_logits[:, :candidate_length, :]
        q_logits = candidate_logits
        
        # Compute probs once
        p_probs = F.softmax(p_logits, dim=-1)
        q_probs = F.softmax(q_logits, dim=-1)
        
        # 2. [Solution 2] Vectorized JS Divergence (Zero-Overhead)
        # JS = 0.5 * (KL(P||M) + KL(Q||M))
        m_probs = 0.5 * (p_probs + q_probs)
        
        # Use Clamp to avoid log(0)
        min_val = 1e-10
        p_probs_safe = p_probs.clamp(min=min_val)
        q_probs_safe = q_probs.clamp(min=min_val)
        m_probs_safe = m_probs.clamp(min=min_val)
        
        # KL(P||M) = sum(p * (log p - log m))
        kl_p = (p_probs * (torch.log(p_probs_safe) - torch.log(m_probs_safe))).sum(dim=-1)
        kl_q = (q_probs * (torch.log(q_probs_safe) - torch.log(m_probs_safe))).sum(dim=-1)
        divs = 0.5 * (kl_p + kl_q) # [B, K]
        
        # 3. [Solution 1] Vectorized Entropy Regulation
        if self.entropy_scale > 0:
            # H(P) = -sum(p * log p)
            entropy = -torch.sum(p_probs * torch.log(p_probs_safe), dim=-1)
            
            # Normalize: H / log(V)
            # Use constant for log(V) to avoid tensor creation overhead if possible, 
            # but here we calculate it safely.
            vocab_size = target_logits.size(-1)
            max_entropy = torch.log(torch.tensor(vocab_size, device=target_logits.device))
            norm_entropy = entropy / max_entropy
            
            # Dynamic Threshold Vector: [B, K]
            dynamic_threshold = self.base_threshold * (1.0 + self.entropy_scale * norm_entropy)
        else:
            dynamic_threshold = self.base_threshold

        # Initial Rejection Mask (True = Rejected)
        rejected_mask = divs > dynamic_threshold
        
        # 4. [Solution 3] Vectorized Parallel Lookahead (No Loops!)
        if self.lookahead_confidence > 0:
            # We need the confidence of the NEXT token for each position i.
            # target_logits has K+1 positions. 
            # For position i in candidate (0 to K-1), the "future" is target_logits[:, i+1, :]
            
            # Get max probs for the whole target sequence (K+1)
            target_all_probs = F.softmax(target_logits, dim=-1)
            target_max_probs = target_all_probs.max(dim=-1).values # [B, K+1]
            
            # Shift left to align "future confidence" with "current position"
            # future_confidence[i] corresponds to confidence of token at i+1
            # We only care about the first K positions
            future_confidence = target_max_probs[:, 1:] # [B, K]
            
            # Create Rescue Mask: True if we should rescue
            # Condition: Currently Rejected AND Future is Confident
            rescue_mask = (rejected_mask) & (future_confidence > self.lookahead_confidence)
            
            # Apply Rescue (Set rejected to False where rescue is True)
            # We must only rescue if i < K-1 because the last token K-1 has no K+1 in logits
            # But the slicing `target_max_probs[:, 1:]` handles dimensions naturally.
            # However, careful: candidate_length could be K. target_logits is K+1.
            # target_max_probs is K+1. future_confidence is K.
            # This aligns perfectly: future_confidence[i] is confidence of step i+1.
            
            rejected_mask = rejected_mask & (~rescue_mask)

        # 5. Enforce Causality (Vectorized Cumprod)
        # Accept = ~Rejected
        accepted_mask = (~rejected_mask).float()
        # Cumprod: 1, 1, 0, 1 -> 1, 1, 0, 0
        valid_mask_cumprod = accepted_mask.cumprod(dim=-1).bool()
        
        n_matches = valid_mask_cumprod.sum().item()
        
        # 6. Construct Output (Standard Logic)
        new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
        correction_term = 0
        
        if is_done_candidate and n_matches == candidate_length:
            n_matches -= 1
            correction_term = 1
            valid_tokens = new_candidate_input_ids[:, :n_matches + 1]
        else:
            # Rejection Sampling
            p_next = target_logits[:, n_matches, :].softmax(dim=-1)
            next_token = torch.multinomial(p_next, num_samples=1)
            
            if n_matches > 0:
                valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], next_token), dim=-1)
            else:
                valid_tokens = next_token

        return valid_tokens, n_matches, target_logits, correction_term, divs, 0.0, 0.0