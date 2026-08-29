import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class DPR:
    def __init__(self, model_name, use_cuda=True, is_question_encoder=False, reranker="/T20200133/Model/google-bert/bert-base-uncased"):
        model_class = DPRQuestionEncoder if is_question_encoder else DPRContextEncoder
        tokenizer_class = DPRQuestionEncoderTokenizerFast if is_question_encoder else DPRContextEncoderTokenizerFast
        self.model = model_class.from_pretrained(model_name)
        if use_cuda:
            self.model.to("cuda")
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.tokenizer.model_max_length = 512
        self.is_question_encoder = is_question_encoder
        
        if not is_question_encoder:
            self.reranker = AutoModelForMaskedLM.from_pretrained(reranker).eval().to(self.model.device)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker)
            self.use_convert = self.tokenizer.get_vocab() != self.reranker_tokenizer.get_vocab()
            if self.use_convert:
                print("Using different vocabularies for DPR and Reranker tokenizers.", flush=True)

    def convert_tokens(self, text, mask_indices, max_length=512):
        # Tokenize the text with both tokenizers, asking for offset mappings
        encoding_a = self.tokenizer(text, return_offsets_mapping=True)
        encoding_b = self.reranker_tokenizer(text, return_offsets_mapping=True)

        offsets_a = encoding_a["offset_mapping"]  # List of (start, end) for each token in A
        offsets_b = encoding_b["offset_mapping"]  # List of (start, end) for each token in B

        # This will hold, for each A-index, the list of B-indices that overlap
        a_to_b_indices = []

        # We can keep a pointer `j` into B-tokens to speed things up,
        # but it is not strictly required. We'll do it to avoid re-scanning from 0 each time.
        j = 0
        b_len = len(offsets_b)

        # Ensure the A-indices are in ascending order (so that the pointer approach works best):
        a_indices_sorted = sorted(mask_indices)

        for a_idx in a_indices_sorted:
            # Get start/end character offsets for this A-token
            a_start, a_end = offsets_a[a_idx]

            # Advance `j` if B-tokens end before `a_start` (no overlap).
            while j < b_len and offsets_b[j][1] <= a_start:
                j += 1

            # Now collect all B tokens that overlap [a_start, a_end).
            # Specifically, we want all tokens where b_start < a_end
            # and b_end > a_start, but usually we check b_start < a_end
            # assuming we already moved j so that b_end_j <= a_start no longer holds.
            b_indices_for_this_a = []

            temp_j = j
            while temp_j < b_len:
                b_start, b_end = offsets_b[temp_j]

                # If we've gone past the end of the A token's range, stop.
                if b_start >= a_end:
                    break

                b_indices_for_this_a.append(temp_j)
                temp_j += 1

            # Store the B-indices that map to this A-token
            a_to_b_indices.extend(b_indices_for_this_a)
        a_to_b_indices = [idx for idx in a_to_b_indices if idx < max_length]
        if isinstance(mask_indices, torch.Tensor):
            return torch.Tensor(a_to_b_indices).type(mask_indices.dtype)
        return a_to_b_indices

    def get_emb(self, texts=None, inputs=None, mask_indices=None, remove_indices=None, return_text=False, **kwargs):
        if "padding" not in kwargs:
            kwargs["padding"] = True
        if inputs is None:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.model.device
            )
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if mask_indices is not None:
            inputs["input_ids"][0][mask_indices] = self.tokenizer.mask_token_id
        elif remove_indices is not None:
            mask = torch.ones(inputs["input_ids"].size(-1), dtype=torch.bool)
            mask[remove_indices] = False
            for k in inputs:
                target_tensor = inputs[k][0][mask].clone()
                inputs[k] = inputs[k][:, : mask.sum()]
                inputs[k][0] = target_tensor

        embeddings = self.model(**inputs).pooler_output
        if return_text:
            return embeddings, self.tokenizer.decode(inputs["input_ids"][0])
        return embeddings

    def get_mask_probs(self, text, mask_indices):
        using_mask_indices = (
            self.convert_tokens(text, mask_indices, self.reranker_tokenizer.model_max_length)
            if self.use_convert
            else mask_indices
        )
        original_token_probs = []
        for mask_idx in using_mask_indices:
            inputs = self.reranker_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
            original_idx = inputs["input_ids"][0][mask_idx].item()
            inputs["input_ids"][0][mask_idx] = self.reranker_tokenizer.mask_token_id
            logits = self.reranker(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1)
            target = probs[mask_idx]
            original_token_probs.append(target[original_idx].item())
        return original_token_probs

    def get_attn(self, text1, text2=None, **kwargs):
        if text2 is None:
            inputs = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        else:
            inputs = self.tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
        output = self.model(**inputs, output_attentions=True)
        return torch.stack(output["attentions"]).detach().cpu().squeeze()  # [layer, head, seq, seq]

    def get_last_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state

    def get_embeddings_module(self):
        """获取词嵌入模块"""
        if hasattr(self.model, 'ctx_encoder'):
            # DPRContextEncoder
            embeddings = self.model.ctx_encoder.bert_model.embeddings.word_embeddings
        elif hasattr(self.model, 'question_encoder'):
            # DPRQuestionEncoder
            embeddings = self.model.question_encoder.bert_model.embeddings.word_embeddings
        else:
            # 其他模型
            embeddings = self.model.embeddings.word_embeddings
        return embeddings


class GMTPFilter:
    """
    GMTP防御过滤器类
    输入：问题、检索到的文档列表
    输出：过滤后的文档列表
    """
    
    def __init__(
        self,
        q_model_name: str = "/T20200133/Models/facebook/dpr-question_encoder-single-nq-base",
        c_model_name: str = "/T20200133/Models/facebook/dpr-ctx_encoder-single-nq-base",
        reranker_name: str = "/T20200133/Model/google-bert/bert-base-uncased",
        N: int = 10,
        M: int = 5,
        remove_threshold: float = 0.1,
        remove_lambda: float = 1.0,
        use_cuda: bool = True
    ):
        """
        初始化GMTP过滤器
        
        参数:
        - q_model_name: 问题编码器模型路径
        - c_model_name: 文档编码器模型路径
        - reranker_name: 重排序器模型名称
        - N: 梯度选择的前N个token
        - M: 掩码概率选择的前M个token
        - remove_threshold: 移除阈值
        - remove_lambda: 移除阈值缩放因子
        - use_cuda: 是否使用CUDA
        """
        self.N = N
        self.M = M
        self.remove_threshold = remove_threshold
        self.remove_lambda = remove_lambda
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        
        # 使用提供的DPR类初始化模型
        self.q_model = DPR(q_model_name, use_cuda=use_cuda, is_question_encoder=True)
        self.c_model = DPR(c_model_name, use_cuda=use_cuda, is_question_encoder=False, reranker=reranker_name)
        
        # 设置重排序器
        self.reranker = self.c_model.reranker
        self.reranker_tokenizer = self.c_model.reranker_tokenizer
        self.use_convert = self.c_model.use_convert
        
        # 梯度存储
        embeddings = self.c_model.get_embeddings_module()
        self.embedding_gradient = GradientStorage(embeddings)
        
        print(f"GMTP Filter initialized with:")
        print(f"  - Question encoder: {q_model_name}")
        print(f"  - Document encoder: {c_model_name}")
        print(f"  - Reranker: {reranker_name}")
        print(f"  - N: {N}, M: {M}")
        print(f"  - Remove threshold: {remove_threshold}, lambda: {remove_lambda}")
        print(f"  - Device: {self.device}")
    
    def get_embeddings(self, texts: Union[str, List[str]], is_question: bool = True):
        """获取文本的嵌入表示"""
        model = self.q_model if is_question else self.c_model
        
        if isinstance(texts, str):
            texts = [texts]
        
        return model.get_emb(texts=texts)
    
    def convert_tokens(self, text: str, mask_indices: List[int], max_length: int = 512):
        """将文档编码器的token索引转换为重排序器的token索引"""
        return self.c_model.convert_tokens(text, mask_indices, max_length)
    
    def get_mask_probs(self, text: str, mask_indices: List[int]):
        """获取掩码token的原始概率"""
        return self.c_model.get_mask_probs(text, mask_indices)
    
    def filter_documents(
        self, 
        question: str, 
        documents: List[str], 
        return_scores: bool = False,
        doc_ids: Optional[List[str]] = None
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        使用GMTP防御过滤文档
        
        参数:
        - question: 问题文本
        - documents: 检索到的文档列表
        - return_scores: 是否返回相似度分数
        - doc_ids: 文档ID列表（可选）
        
        返回:
        - 过滤后的文档列表（或文档列表和分数元组）
        """
        doc_infos = []
        
        # 第一步：计算每个文档的梯度分数
        for i, doc in enumerate(documents):
            info = {"document": doc, "original_index": i}
            if doc_ids is not None:
                info["id"] = doc_ids[i]
            
            # 清零梯度
            self.c_model.model.zero_grad()
            self.q_model.model.zero_grad()
            
            # 编码问题和文档
            q_emb = self.get_embeddings(question, is_question=True)
            d_emb = self.get_embeddings(doc, is_question=False)
            
            # 计算相似度
            sim = torch.mm(d_emb, q_emb.T)
            info["sim"] = sim.item()
            
            # 反向传播获取梯度
            sim.backward()
            
            # 获取梯度并计算梯度分数
            grad = self.embedding_gradient.get()
            scores = grad.norm(dim=-1)
            info["ret_scores"] = scores[0].tolist()
            
            doc_infos.append(info)
        
        # 第二步：为每个文档计算掩码概率
        for j, doc in enumerate(documents):
            info = doc_infos[j]
            grad_scores = torch.tensor(info["ret_scores"])
            
            # 计算梯度阈值
            threshold = grad_scores.mean().item()
            
            # 选择梯度分数高于阈值的token
            selected_indices = torch.where(grad_scores > threshold)[0]
            if len(selected_indices) == 0:
                info["avg_masked_prob"] = 0.0
                info["masked_probs"] = [0.0]
                continue
            
            selected_values = grad_scores[selected_indices]
            
            # 选择前N个梯度分数最高的token
            top_n = min(self.N, len(selected_values))
            top_indices = selected_values.topk(top_n).indices
            top_indices = selected_indices[top_indices]
            
            # 获取掩码概率
            if len(top_indices) == 0:
                masked_probs = [0.0]
            else:
                masked_probs = self.get_mask_probs(doc, top_indices.tolist())
                # 选择前M个最小的掩码概率
                masked_probs = np.sort(masked_probs)
                masked_probs = masked_probs[:self.M]
            
            # 计算平均掩码概率
            avg_masked_prob = np.mean(masked_probs)
            info["avg_masked_prob"] = avg_masked_prob
            info["masked_probs"] = masked_probs
        
        # 第三步：按相似度排序并过滤
        # 按相似度降序排序
        sorted_doc_infos = sorted(doc_infos, key=lambda x: x["sim"], reverse=True)
        
        # 应用阈值过滤
        threshold = self.remove_threshold * self.remove_lambda
        filtered_docs = [info for info in sorted_doc_infos if info["avg_masked_prob"] > threshold]
        
        # 准备返回结果
        result_docs = [info["document"] for info in filtered_docs]
        
        if return_scores:
            result_scores = [info["sim"] for info in filtered_docs]
            return result_docs, result_scores
        
        return result_docs
    
    def filter_topk_contents(
        self, 
        question: str, 
        topk_contents: List, 
        topk: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        过滤topk_contents格式的检索结果
        
        参数:
        - question: 问题文本
        - topk_contents: 检索结果列表，每个元素为(文档内容, 相似度分数)
        - topk: 返回的文档数量（默认使用输入的长度）
        
        返回:
        - 过滤后的(文档内容, 相似度分数)列表
        """
        if not topk_contents:
            return []
        
        documents = topk_contents
        
        # 使用filter_documents进行过滤
        filtered_docs, filtered_scores = self.filter_documents(
            question, documents, return_scores=True
        )
        
        result = filtered_docs

        return result


def gmtp_filter_function(
    question: str,
    topk_contents: List,
) -> List[Tuple[str, float]]:
    
    # 初始化GMTP过滤器
    gmtp_filter = GMTPFilter()
    
    # 执行过滤
    filtered_results = gmtp_filter.filter_topk_contents(question, topk_contents)
    
    return filtered_results


# 使用示例
if __name__ == "__main__":
    # 示例数据
    question = "What is the capital of France?"
    
    topk_contents = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe.",
        "The Eiffel Tower is located in Paris, France.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
    ]
    
    # 使用提供的模型路径和默认参数进行过滤
    filtered_results = gmtp_filter_function(
        question=question,
        topk_contents=topk_contents
    )
    
    print("Filtered Results:", filtered_results)