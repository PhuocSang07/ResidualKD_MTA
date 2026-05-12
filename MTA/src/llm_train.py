from arguments import Arguments
from teacher_llm import Teacher, TeacherOutput
from student import LLMModel, StudentOutput
from data_utils import LLMDataset, LLMDataCollator
from span_loss import compute_overall_span_loss, build_spacy_matcher, get_spans_offsets

from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from evaluator import Evaluator


def load_tokenizer(model_type, path, kwargs):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)
    if model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "qwen":
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print('tokenizer unknown')
    return tokenizer


class Trainer:
    def __init__(self, student: LLMModel, model_type: str,
                 args: Arguments, teacher_model: Teacher = None):
        super().__init__()
        self.student = student.train()
        self.teacher_model = teacher_model

        self.args = args
        self.args.p = max(args.p, 1e-5)

        self.alpha = args.hard_label_loss_weight
        self.temperature = args.temperature
        self.step = 0

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loader(args, model_type)

        # MTA projectors – one nn.Linear(student_hidden, teacher_hidden) per layer pair.
        self.mta_projector_list = None
        self.nlp = None
        self.matcher = None
        if args.MTA_mode and teacher_model is not None:
            student_hidden = self._get_hidden_size(self.student.model, model_type)
            teacher_hidden = self._get_hidden_size(self.teacher_model.model, model_type)
            projector_list = nn.ModuleList()
            for _ in range(len(args.teacher_layer_mapping)):
                projector = nn.Linear(student_hidden, teacher_hidden).to(self.student.device)
                projector_list.append(projector)
            self.mta_projector_list = projector_list

            # spaCy pipeline for on-the-fly span / word extraction.
            self.nlp, self.matcher = build_spacy_matcher()

    @staticmethod
    def _get_hidden_size(model, model_type):
        if model_type == 'gpt2':
            return model.config.n_embd
        return model.config.hidden_size

    def get_data_loader(self, args: Arguments, model_type: str):
        self.tokenizer = load_tokenizer(model_type, args.student_tokenizer,
                                        args.load_student_tokenizer_kwargs)

        train_dataset = LLMDataset(args.train_data, self.tokenizer, args.max_len // 2)

        train_collate = LLMDataCollator(self.tokenizer, model_type, do_train=True,
                                        max_len=args.max_len,
                                        pad_to_multiple_of=args.pad_to_multiple_of,
                                        return_tensors='pt', padding=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_collate)

        return train_loader, None, None

    def get_teacher_eval(self, inputs):
        outputs = self.teacher_model.decode(inputs)
        outputs.logits = outputs.logits.to(self.student.device, non_blocking=True)
        if outputs.hidden_states is not None:
            # `hidden_states` is a tuple from HF; move each tensor to student device.
            outputs.hidden_states = tuple(h.to(self.student.device, non_blocking=True)
                                          for h in outputs.hidden_states)
        return outputs

    def soft_label_distill_loss(self, student_logits, teacher_logits, mask, distill_temperature=2.0):
        student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)
        loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def skewed_forward_kl(self, logits, teacher_logits, labels, lam=0.1):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        mixed_probs = lam * teacher_probs + (1 - lam) * student_probs
        mixed_logprobs = torch.log(mixed_probs)

        mask = (labels != -100).int()
        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0).clamp(min=1.0)
        return distil_loss

    def mta_span_loss(self, student_outputs: StudentOutput, teacher_outputs: TeacherOutput,
                      attention_mask, input_ids):
        """Decode current batch -> run spaCy -> compute MTA span loss.

        Mirrors the on-the-fly path used in mta_dskd_v2's DualSpaceKDV2WithETA.forward.
        """
        if (self.mta_projector_list is None or self.nlp is None
                or student_outputs.hidden_states is None
                or teacher_outputs.hidden_states is None):
            return torch.tensor(0.0, device=self.student.device)

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        offsets_mapping = self.tokenizer(
            input_texts,
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt',
        )['offset_mapping']

        spans_offsets, words_offsets = get_spans_offsets(input_texts, self.nlp, self.matcher)

        span_loss = compute_overall_span_loss(
            self.mta_projector_list,
            attention_mask,
            student_outputs.logits,
            teacher_outputs.logits,
            student_outputs.hidden_states,
            teacher_outputs.hidden_states,
            offsets_mapping,
            spans_offsets,
            words_offsets,
            student_layer_mapping=self.args.student_layer_mapping,
            teacher_layer_mapping=self.args.teacher_layer_mapping,
            split_layer_mapping=self.args.split_layer_mapping,
            entropy_weight=self.args.entropy_weight,
        )
        return self.args.w_span_loss * span_loss

    def compute_loss(self, inputs, labels, teacher_outputs=None):
        student_outputs = self.student(inputs)
        attention_mask = inputs['attention_mask'].to(self.student.device, non_blocking=True)

        kd_loss = self.skewed_forward_kl(student_outputs.logits, teacher_outputs.logits, labels)

        if self.args.MTA_mode:
            span_loss = self.mta_span_loss(student_outputs, teacher_outputs,
                                           attention_mask, inputs['input_ids'])
            total_loss = kd_loss + span_loss
        else:
            total_loss = kd_loss

        return total_loss, kd_loss


def train(args: Arguments, trainer: Trainer, evaluator: Evaluator, grad_accum_steps=1):
    trainer.student.train()
    train_loader = trainer.train_loader

    param_groups = [{'params': trainer.student.parameters(), 'lr': args.learning_rate}]
    if trainer.mta_projector_list is not None:
        param_groups.append({'params': trainer.mta_projector_list.parameters(),
                             'lr': args.projector_lr})
    optimizer = optim.AdamW(param_groups)

    num_steps = len(train_loader) // grad_accum_steps + 1
    total_training_steps = num_steps * args.num_train_epochs

    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=1e-7)

    all_clip_params = list(trainer.student.parameters())
    if trainer.mta_projector_list is not None:
        all_clip_params.extend(list(trainer.mta_projector_list.parameters()))

    best_result = 0
    for epoch in range(args.num_train_epochs):
        print(('\n' + '%8s' + '%14s' + '%17s' * 2) % ('epoch', 'memory', 'loss', 'student_loss'))
        p_bar = tqdm(train_loader, total=len(train_loader))
        loss_total = 0
        student_loss_total = 0
        step = 0

        for batch in p_bar:
            inputs, labels = batch
            teacher_outputs = trainer.get_teacher_eval(inputs)
            labels = labels.to(trainer.student.device)

            with autocast():
                loss, student_loss = trainer.compute_loss(inputs, labels, teacher_outputs)

            scaler.scale(loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_clip_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_total += loss.item()
            student_loss_total += student_loss.item()
            step += 1

            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
            s = ('%8s' + '%14s' + '%17.5g' * 2) % (
                f'{epoch + 1}/{args.num_train_epochs}', memory,
                loss_total / step, student_loss.item())
            p_bar.set_description(s)

            if torch.isnan(loss):
                break

        with torch.cuda.amp.autocast(dtype=torch.float16):
            evaluator.model = trainer.student.model
            dolly = evaluator.evaluate_benchmark_dataset(
                dataset_path=args.val_data,
                dataset_name='dolly', batch_size=args.val_batch_size,
                max_seq_length=128, max_new_tokens=256)

        if dolly > best_result:
            best_result = dolly
            trainer.student.save(args.output_dir)

        trainer.student.save(args.output_dir + f'-epoch{epoch}')
