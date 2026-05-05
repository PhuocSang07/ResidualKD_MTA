import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoTokenizer
import re


def _safe_load_tokenizer(name, **kwargs):
    """Load a tokenizer, patching the list-vs-dict bug in extra_special_tokens
    present in some older model configs (e.g. Qwen1.5 on new transformers)."""
    import transformers.tokenization_utils_base as _tub

    # transformers >= 5.x: SpecialTokensMixin was merged into PreTrainedTokenizerBase
    _token_base_cls = getattr(_tub, "SpecialTokensMixin", None) or _tub.PreTrainedTokenizerBase
    _orig = _token_base_cls._set_model_specific_special_tokens

    def _patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return _orig(self, special_tokens)

    _token_base_cls._set_model_specific_special_tokens = _patched
    try:
        return AutoTokenizer.from_pretrained(name, **kwargs)
    finally:
        _token_base_cls._set_model_specific_special_tokens = _orig



def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict

def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

def normalize(value):
    # Stay in float32 — converting back to fp16/bf16 can overflow (fp16 max=65504)
    value_f = value.float()
    stds = value_f.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return value_f / stds

def KL_wo(y_s, y_t, T=1):
    # Compute in float32: BF16 can produce exact 0 probabilities → 0 * -inf = NaN
    p_s = F.log_softmax(y_s.float() / T, dim=-1)
    p_t = F.softmax(y_t.float() / T, dim=-1)
    # clamp p_t so 0 * -inf never appears
    loss = -torch.sum(p_t.clamp(min=1e-10) * p_s, dim=-1).mean()
    return loss

class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = 2   
    def sinkhorn_normalized(self, x, n_iters=20):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True).clamp(min=1e-10)
            x = x / torch.sum(x, dim=0, keepdim=True).clamp(min=1e-10)
        return x

    def sinkhorn_loss(self, x, y, epsilon=0.5, n_iters=10):
        # Compute in float32: BF16 causes exp() underflow → K=0 → NaN
        x_f, y_f = x.float(), y.float()
        Wxy = torch.cdist(x_f, y_f, p=1)
        K = torch.exp(-Wxy / epsilon).clamp(min=1e-30)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)
    def forward(self, y_s, y_t):
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s/self.T)
        p_t = softmax(y_t/self.T)
        emd_loss = 0
        for i in range(p_s.shape[0]):
            emd_loss += 0.001*self.sinkhorn_loss(x=p_s[i],y=p_t[i])
        return emd_loss

def greedy_algorithm_adjust_s(t, s):
    batch_size, T, k = t.shape
    _, n, _ = s.shape
    
    # Initialize the adjusted source tensor
    s_adjusted = torch.zeros_like(t)
    
    for b in range(batch_size):
        # Initialize set of available source indices for each batch
        available_indices = list(range(n))
        
        for i in range(T):
            C_min = float('inf')
            j_star = -1
            
            for j in available_indices:
                # Compute cost as the sum of absolute differences for each batch
                C = torch.sum(torch.abs(t[b,:,i] - s[b,:,j]))
                
                if C < C_min:
                    C_min = C
                    j_star = j
            
            # Assign the best matching source vector to the adjusted tensor
            s_adjusted[b,:,i] = s[b,:,j_star]
            
            # Remove the selected index from available indices
            available_indices.remove(j_star)

    return s_adjusted

class TeacherWrapper:
    """
    MTA-style teacher wrapper.
    decode() tự move inputs lên teacher_device (giống Teacher.decode() trong MTA).
    """
    def __init__(self, model, device):
        self.model = model.eval()
        # "auto" là loading param của HF, không phải tensor device —
        # resolve về device của parameter đầu tiên (thường là embedding layer)
        if isinstance(device, str) and device == "auto":
            self.device = next(model.parameters()).device
        else:
            self.device = device

    @property
    def name_or_path(self):
        return getattr(self.model.config, 'name_or_path',
                       getattr(self.model.config, '_name_or_path', ''))

    def decode(self, inputs, output_hidden_states=False):
        # Giống MTA teacher_llm.py: mỗi model tự move inputs vào device của nó
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.model(**inputs, output_hidden_states=output_hidden_states)


class StudentWrapper(nn.Module):
    """
    MTA-style student wrapper.
    forward() tự move inputs lên student_device (giống LLMModel.forward() trong MTA).
    """
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    @property
    def name_or_path(self):
        return getattr(self.model.config, 'name_or_path',
                       getattr(self.model.config, '_name_or_path', ''))

    def forward(self, input_ids, attention_mask, labels, output_hidden_states=False):
        # Giống MTA student.py: mỗi model tự move inputs vào device của nó
        input_ids     = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels        = labels.to(self.device)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )


class DistillationModel(nn.Module):
    def __init__(self, student, teacher, teacher_tokenizer, student_tokenizer,
                 use_span_loss=False, student_device=None, teacher_device=None):
        super().__init__()

        self.student_device = student_device or next(student.parameters()).device
        self.teacher_device = teacher_device or next(teacher.parameters()).device

        # Wrap giống MTA: mỗi model tự quản lý device của mình
        self.student = StudentWrapper(student, self.student_device)
        self._teacher = TeacherWrapper(teacher, self.teacher_device)
        self.use_span_loss = use_span_loss

        # MTA pattern: copy teacher lm_head sang student device để tính loss
        # tránh transfer hidden states nặng qua PCIe
        self.teacher_lm_head = self._copy_teacher_lm_head(teacher)

    def _copy_teacher_lm_head(self, teacher):
        """Giống MTA Trainer.__init__: copy teacher lm_head sang student device, frozen."""
        try:
            lm_head = teacher.lm_head
            copied = nn.Linear(
                lm_head.in_features, lm_head.out_features,
                bias=lm_head.bias is not None
            ).to(device=self.student_device, dtype=lm_head.weight.dtype)
            copied.load_state_dict(lm_head.state_dict())
            for p in copied.parameters():
                p.requires_grad = False
            return copied
        except AttributeError:
            return None

    def get_teacher_eval(self, teacher_inputs):
        """
        Giống MTA Trainer.get_teacher_eval():
        1. Gọi teacher.decode() → teacher tự move inputs lên teacher_device
        2. Transfer outputs về student_device với non_blocking=True
        """
        output = self._teacher.decode(teacher_inputs, output_hidden_states=self.use_span_loss)

        # Transfer logits và hidden states về student device (non_blocking như MTA)
        output.logits = output.logits.to(self.student_device, non_blocking=True)
        if self.use_span_loss and output.hidden_states is not None:
            output.hidden_states = tuple(
                h.to(self.student_device, non_blocking=True)
                for h in output.hidden_states
            )
        return output

    @property
    def teacher(self):
        return self._teacher

    def forward(self, student_input_ids, student_attention_mask, student_labels,
                teacher_input_ids, teacher_attention_mask, teacher_labels):
        # Step 1: teacher inference → transfer về student device (MTA pattern)
        teacher_output = self.get_teacher_eval({
            'input_ids':      teacher_input_ids,
            'attention_mask': teacher_attention_mask,
            'labels':         teacher_labels,
        })

        # Step 2: student inference — StudentWrapper tự move inputs lên student_device
        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=student_labels,
            output_hidden_states=self.use_span_loss,
        )
        return student_output, teacher_output


# change stu label to teacher generation

class DistillationModel2(nn.Module):
    def __init__(self, student, teacher, teacher_tokenizer, student_tokenizer, ignore_index=-100):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.teacher.eval()
        self.ignore_index = ignore_index

    def forward(self, student_input_ids, student_attention_mask, student_labels, teacher_input_ids, teacher_attention_mask, teacher_labels):
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )
            teacher_logits = teacher_output.logits
            teacher_token_ids = torch.argmax(teacher_logits, dim=-1)

        student_answer_index, student_answer_size = self.__get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(teacher_labels)


        teacher_answers_text = []
        for i in range(len(teacher_answer_index)):
            start_idx = teacher_answer_index[i]
            end_idx = start_idx + teacher_answer_size[i]
            answer_ids = teacher_token_ids[i, start_idx:end_idx] 
            answer_text = self.teacher_tokenizer.decode(answer_ids)  
            teacher_answers_text.append(answer_text)

        new_student_labels = torch.full_like(student_labels, fill_value=-100) 
        for i, answer_text in enumerate(teacher_answers_text):
            student_start_idx = student_answer_index[i]
            encoded_answer = self.student_tokenizer.encode(answer_text, add_special_tokens=True)  
            end_idx = min(student_start_idx + len(encoded_answer), student_labels.size(1))  
            new_student_labels[i, student_start_idx:end_idx] = torch.tensor(encoded_answer[:end_idx-student_start_idx], dtype=torch.long)

        #print(teacher_answers_text)
        #print(student_labels)
        #print(new_student_labels)


        

        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=new_student_labels
        )
        return student_output, teacher_output
    
    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size


class DistillationLoss(nn.Module):
    def __init__(self, batch_limit=100, store_path='teacher_logits_partial.npy', crossentropy_weight=1, distillation_weight=1, student_temperature=1, teacher_temperature=1, skip_student_eos=False, skip_teacher_eos=False, ignore_index=-100, debug=False, debug_rank=0, tokenizer_student=None, tokenizer_teacher=None, f=1, span_loss_weight=0.0, distil_config=None):
        super().__init__()
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.ignore_index = ignore_index
        self.debug_rank = debug_rank
        self.debug = debug
        self.f = f
        self.span_loss_weight = span_loss_weight
        self.distil_config = distil_config

        self.store_teacher_logits = True
        self.batch_limit = batch_limit  # 设定每100个样本保存一次
        self.store_path = store_path
        self.teacher_logits_temp_storage = []

        # Projectors are created on DistillationModel (not here) so they appear in
        # model.parameters() and are included in the optimizer before training starts.
        # forward() receives projectors as an argument passed from the training loop.

        if self.debug:
            print("Distillation loss parameters:")
            print(f"Crossentropy weight: {crossentropy_weight}")
            print(f"Distillation weight: {distillation_weight}")
            print(f"Student temperature: {student_temperature}")
            print(f"Teacher temperature: {teacher_temperature}")
            print(f"Skip student eos: {skip_student_eos}")
            print(f"Skip teacher eos: {skip_teacher_eos}")
            print(f"Ignore index: {ignore_index}")
            print(f"Debug: {debug}")
            print(f"Debug rank: {debug_rank}")
            print(f"Span loss weight: {span_loss_weight}")

            self.student_tokenizer = _safe_load_tokenizer(tokenizer_student, trust_remote_code=True)
            self.teacher_tokenizer = _safe_load_tokenizer(tokenizer_teacher, trust_remote_code=True)

    def forward(self, epoch, student_predictions, teacher_predictions, student_targets, teacher_targets, rank=0, span_data=None, projectors=None):
        student = student_predictions.logits
        teacher = teacher_predictions.logits
        if self.store_teacher_logits:
            processed_teacher_logits = []

        # Get answer first token and answer size
        student_answer_index, student_answer_size = self.__get_start_and_size_answers(
            student_targets)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(
            teacher_targets)

        # Avoid eos token, if needed
        if self.skip_student_eos: student_answer_size = [size-1 for size in student_answer_size]
        if self.skip_teacher_eos: teacher_answer_size = [size-1 for size in teacher_answer_size]
        
        student = normalize(student)      
        teacher = normalize(teacher)

        # Align answer first token, pad to right and compute softmax
        for i in range(student.size(0)):
            shift = student_answer_index[i]
            size = student_answer_size[i]
            end_shift = shift+size
            student[i] = torch.cat((
                torch.nn.functional.softmax(student[i, shift:end_shift, :]/self.student_temperature, dim=-1),
                torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
            )
        for i in range(teacher.size(0)):
            shift = teacher_answer_index[i]
            size = teacher_answer_size[i]
            end_shift = shift+size
            teacher[i] = torch.cat((
               torch.nn.functional.softmax(teacher[i, shift:end_shift, :]/self.teacher_temperature, dim=-1),
               torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
            )
        
        # Cut to max answer length
        mex_length = max(max(student_answer_size), max(teacher_answer_size))

        student = student[:, :mex_length, :]
        teacher = teacher[:, :mex_length, :]
        
        sinkorn_loss = Sinkhorn_seq()

        if self.debug and rank == self.debug_rank:
            print("\n\n----------------------------------")
            print("------- Label / Prediction -------")
            print("----------------------------------")
            student_labels = [row[row != -100] for row in student_targets]
            teacher_labels = [row[row != -100] for row in teacher_targets]
            print("------- Label shape -------")
            print(f"Student label shape: {student_answer_size[0]}")
            print(f"Teacher label shape: {teacher_answer_size[0]}")
            print("------- Student Label -> Prediction -------")
            print(self.student_tokenizer.batch_decode(student_labels[0]))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Teacher Label -> Prediction -------")
            print(self.teacher_tokenizer.batch_decode(teacher_labels[0]))
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print("------- Prediction Teacher -> Student  -------")
            
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}\n")

        # # Sort in descending order to align probabilities
        # student = student.sort(dim=-1, descending=True).values
        # teacher = teacher.sort(dim=-1, descending=True).values
        teacher = improved_sort(teacher)
        teacher = teacher[:,:,:50]
        if self.f == 1:
            student = improved_sort(student)
            student = student[:,:,:50]
        elif self.f == 2:
            student = greedy_algorithm_adjust_s(teacher,student)

        # Pad to get same vocabulary size
        diff_size = student.size(2) - teacher.size(2)
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)
            
        if self.debug and rank == self.debug_rank:
            print("--------------------------------------------")
            print("---- Post-treatment tensor architecture ----")
            print("--------------------------------------------")
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}")
            print(" ------- First token -------")
            print(f"Student first logits: {student[0][0][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][0][:5].tolist()}")
            print(f"Student last logits: {student[0][0][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][0][-5:].tolist()}")
            print(" ------- Last token -------")
            print(f"Student first logits: {student[0][-1][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][-1][:5].tolist()}")
            print(f"Student last logits: {student[0][-1][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][-1][-5:].tolist()}\n")

        # Cross entropy loss
        crossentropy_loss = self.crossentropy_weight * student_predictions.loss

        # Guard: if all samples have no valid answer tokens, skip distillation loss
        mex_length = max(max(student_answer_size), max(teacher_answer_size))
        if mex_length == 0:
            total_loss = crossentropy_loss
            span_loss = torch.zeros(1, device=crossentropy_loss.device)
            return total_loss, crossentropy_loss, torch.zeros(1, device=crossentropy_loss.device), span_loss

        distillation_loss = torch.zeros(student.size(0), device=student.device)
        for i in range(student.size(0)):
            size = min(student_answer_size[i], teacher_answer_size[i])
            if size > 0:
                # .mean() on shape [size] — safe because size > 0
                distillation_loss[i] = abs(student[i][:size] - teacher[i][:size]).sum(-1).mean()

        min_seq_len = min(student.size(1), teacher.size(1))
        teacher_aligned = teacher[:, :min_seq_len, :]
        student_aligned = student[:, :min_seq_len, :]

        distillation_loss = distillation_loss + KL_wo(teacher_aligned, student_aligned)*0.1
        distillation_loss = distillation_loss.mean() + sinkorn_loss(teacher_aligned, student_aligned)*0.1
        distillation_loss = self.distillation_weight * (distillation_loss) * 1

        if self.debug and rank == self.debug_rank:
            print(f"Loss: Crossentropy loss: {crossentropy_loss} | Distillation loss: {distillation_loss} | Total loss: {crossentropy_loss + distillation_loss}")

        # Guard each component BEFORE summing so crossentropy gradient is always preserved
        distillation_loss = torch.nan_to_num(distillation_loss, nan=0.0, posinf=0.0, neginf=0.0)

        span_loss = torch.tensor(0.0, device=student.device)
        if (self.span_loss_weight > 0 and span_data is not None
                and projectors is not None
                and student_predictions.hidden_states is not None
                and teacher_predictions.hidden_states is not None):
            from models.span_utils import compute_overall_span_loss_cross
            span_loss = compute_overall_span_loss_cross(
                projectors,
                span_data['t_attention_mask'],
                span_data['s_attention_mask'],
                student_predictions.logits,
                teacher_predictions.logits,
                student_predictions.hidden_states,
                teacher_predictions.hidden_states,
                span_data['t_offsets_mapping'],
                span_data['s_offsets_mapping'],
                span_data['spans_offsets'],
                span_data['words_offsets'],
                self.distil_config,
            )
            span_loss = self.span_loss_weight * span_loss
            span_loss = torch.nan_to_num(span_loss, nan=0.0, posinf=0.0, neginf=0.0)

        total_loss = crossentropy_loss + distillation_loss + span_loss
        return total_loss, crossentropy_loss, distillation_loss, span_loss

    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size
    
    def save_teacher_logits_partial(self):

        with open(self.store_path, 'ab') as f:  
            for logits in self.teacher_logits_temp_storage:
                np.save(f, logits)

    def on_epoch_end(self):

        if self.teacher_logits_temp_storage:
            self.save_teacher_logits_partial()
            self.teacher_logits_temp_storage = []  
