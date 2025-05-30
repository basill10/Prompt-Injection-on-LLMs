import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm

class UniversalPromptInjectionLLM:
    """
    Implementation of Universal Prompt Injection Attack using Momentum Greedy Coordinate Gradient
    as described in Algorithm 1.
    
    This class encapsulates the functionality to perform gradient-based prompt injection attacks
    on language models such as GPT-2 and T5.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_injection_length: int = 20,
        modifiable_subset: Optional[List[int]] = None,
        momentum_weight: float = 0.9,
        topk: int = 100,
        batch_size: int = 16,
        max_iterations: int = 50,
    ):
        """
        Initialize the Universal Prompt Injection Attack.
        
        Args:
            model_name: The name of the pre-trained model to use ('gpt2' or 't5-small')
            device: The device to run the model on ('cuda' or 'cpu')
            max_injection_length: Maximum length of the injection content
            modifiable_subset: Indices of tokens that can be modified
            momentum_weight: Weight of the momentum term in gradient updates
            topk: Number of top candidates to consider in each iteration
            batch_size: Batch size for evaluation
            max_iterations: Maximum number of iterations for the attack
        """
        self.device = device
        self.model_name = model_name
        self.max_injection_length = max_injection_length
        self.momentum_weight = momentum_weight
        self.topk = topk
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        
        if "gpt2" in model_name:
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "t5" in model_name or "flan-t5" in model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'gpt2', 't5-small', or 'google/flan-t5-small'.")
        
        self.vocab_size = len(self.tokenizer)
        
        if modifiable_subset is None:
            self.modifiable_subset = list(range(max_injection_length))
        else:
            self.modifiable_subset = modifiable_subset
        
        self.previous_gradient = None
    
    def _compute_loss(
        self, 
        injection_tokens: torch.Tensor, 
        user_instructions: List[str], 
        target_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute the loss for the given injection tokens when prepended to user instructions.
        
        Args:
            injection_tokens: Tensor of token IDs representing the injection content
            user_instructions: List of user instructions to attack
            target_texts: List of target texts that the model should generate
            
        Returns:
            Loss value as a torch tensor
        """
        total_loss = 0.0
        n_samples = len(user_instructions)
        
        for i in range(n_samples):
            instruction = user_instructions[i]
            target = target_texts[i]
            
            instruction_tokens = self.tokenizer.encode(instruction, return_tensors="pt").to(self.device)
            target_tokens = self.tokenizer.encode(target, return_tensors="pt").to(self.device)
            
            if "gpt2" in self.model_name:
                input_tokens = torch.cat([
                    injection_tokens.unsqueeze(0), instruction_tokens, target_tokens
                ], dim=1)
                ignore_len = injection_tokens.size(0) + instruction_tokens.size(1)
                labels = torch.cat([
                    torch.full((1, ignore_len), -100, dtype=torch.long, device=self.device),
                    target_tokens
                ], dim=1)
                outputs = self.model(input_tokens, labels=labels)
                loss = outputs.loss
                
            elif "t5" in self.model_name:
                injection_text = self.tokenizer.decode(injection_tokens)
                combined_input = injection_text + instruction
                combined_tokens = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.device)
                
                outputs = self.model(
                    input_ids=combined_tokens,
                    labels=target_tokens
                )
                loss = outputs.loss
            
            total_loss += loss
            
        return total_loss / n_samples
    
    def _compute_gradient(
        self, 
        injection_tokens: torch.Tensor, 
        user_instructions: List[str], 
        target_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute the gradient of the loss with respect to the injection tokens.
        
        Args:
            injection_tokens: Tensor of token IDs representing the injection content
            user_instructions: List of user instructions to attack
            target_texts: List of target texts that the model should generate
            
        Returns:
            Gradient tensor with shape [max_injection_length, vocab_size]
        """
        gradients = torch.zeros(
            injection_tokens.size(0), self.vocab_size, device=self.device
        )
        
        TOKEN_CANDIDATES = 20 
        for pos in range(injection_tokens.size(0)):
            for token_id in range(min(TOKEN_CANDIDATES, self.vocab_size)):
                modified_injection = injection_tokens.clone()
                modified_injection[pos] = token_id
                with torch.no_grad():  
                    loss = self._compute_loss(modified_injection, user_instructions, target_texts)
                gradients[pos, token_id] = loss.item()
        
        for pos in range(gradients.size(0)):
            if torch.max(gradients[pos]) - torch.min(gradients[pos]) > 1e-6:  
                gradients[pos] = (gradients[pos] - torch.min(gradients[pos])) / (torch.max(gradients[pos]) - torch.min(gradients[pos]))
        
        return gradients
    
    def attack(
        self,
        user_instructions: List[str],
        target_texts: List[str],
        initial_injection: Optional[str] = None,
    ) -> str:
        """
        Execute the universal prompt injection attack according to Algorithm 1.
        
        Args:
            user_instructions: List of user instructions to attack
            target_texts: List of target texts that the model should generate
            initial_injection: Initial injection content (optional)
            
        Returns:
            The optimized injection content
        """
        if initial_injection is None:
            injection_tokens = torch.randint(
                0, self.vocab_size, (self.max_injection_length,), device=self.device
            )
        else:
            injection_tokens = torch.tensor(
                self.tokenizer.encode(initial_injection)[:self.max_injection_length],
                device=self.device
            )
            if len(injection_tokens) < self.max_injection_length:
                pad_length = self.max_injection_length - len(injection_tokens)
                padding = torch.tensor([self.tokenizer.pad_token_id] * pad_length, device=self.device)
                injection_tokens = torch.cat([injection_tokens, padding])
        
        self.previous_gradient = torch.zeros(self.max_injection_length, self.vocab_size, device=self.device)
        
        for t in tqdm(range(self.max_iterations), desc="Attack Progress"):
            gradient = self._compute_gradient(injection_tokens, user_instructions, target_texts)
            
            gradient = gradient + self.momentum_weight * self.previous_gradient
            self.previous_gradient = gradient.clone()
            
            candidates = {}
            for idx in self.modifiable_subset:
                
                topk_values, topk_indices = torch.topk(
                    -gradient[idx], k=min(self.topk, self.vocab_size)
                )
                candidates[idx] = topk_indices.tolist()
            
            best_loss = float('inf')
            best_injection = injection_tokens.clone()
            
            for b in range(self.batch_size):
                candidate_injection = injection_tokens.clone()
                
                idx = np.random.choice(self.modifiable_subset)
                
                if candidates[idx]:
                    new_token = np.random.choice(candidates[idx])
                    candidate_injection[idx] = new_token
                
                with torch.no_grad():
                    loss = self._compute_loss(candidate_injection, user_instructions, target_texts)
                
                if loss < best_loss:
                    best_loss = loss
                    best_injection = candidate_injection.clone()
            
            injection_tokens = best_injection
            
            if (t + 1) % 5 == 0 or t == 0:
                current_injection = self.tokenizer.decode(injection_tokens.tolist())
                print(f"\nIteration {t+1}/{self.max_iterations}")
                print(f"Current injection: {current_injection}")
                print(f"Current loss: {best_loss.item():.4f}")
        
        optimized_injection = self.tokenizer.decode(injection_tokens.tolist())
        
        return optimized_injection
    
    def evaluate(
        self,
        injection_content: str,
        test_instructions: List[str],
        target_texts: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the effectiveness of the injection content on test instructions.
        
        Args:
            injection_content: The injection content to evaluate
            test_instructions: List of test instructions
            target_texts: (Optional) List of target texts
            verbose: Whether to print evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        injection_tokens = torch.tensor(
            self.tokenizer.encode(injection_content)[:self.max_injection_length],
            device=self.device
        )
        
        if target_texts is not None:
            with torch.no_grad():
                loss = self._compute_loss(injection_tokens, test_instructions, target_texts)
        
        responses = []
        for instruction in test_instructions:
            combined_input = injection_content + instruction
            
            if "gpt2" in self.model_name:
                input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.device)
                output = self.model.generate(
                    input_ids, 
                    max_length=50, 
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
            elif "t5" in self.model_name:
                input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.device)
                output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
            responses.append(response)
        
        if verbose:
            print(f"Evaluation Results:")
            for i in range(len(test_instructions)):
                print(f"Instruction: {test_instructions[i]}")
                print(f"Injection + Instruction: {injection_content + test_instructions[i]}")
                print(f"Response: {responses[i]}")
                print()
        return {
            "responses": responses
        }