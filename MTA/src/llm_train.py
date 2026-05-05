from arguments import Arguments
from teacher_llm import Teacher, TeacherOutput
from student import LLMModel, StudentOutput
from data_utils import LLMDataset, LLMDataCollator

from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_scheduler
from evaluator import Evaluator


def load_tokenizer(model_type, path, kwargs):        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)
    if model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "qwen":
        # tokenizer.pad_token_id = 151646
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print('tokenizer unknow')
    
    return tokenizer

def get_token_mapping(s_tokenizer, t_tokenizer, device):
    t_vocab = t_tokenizer.get_vocab()
    s_vocab = s_tokenizer.get_vocab()
    t_id_mapping = []
    s_id_mapping = []
    for s_token, s_token_id in s_vocab.items():
        if s_token in t_vocab:
            s_id_mapping.append(s_token_id)
            t_id_mapping.append(t_vocab[s_token])

    return torch.tensor(s_id_mapping, device=device), torch.tensor(t_id_mapping, device=device)


class Trainer:
    def __init__(self, student: LLMModel, model_type: str,
                 args: Arguments, teacher_model: Teacher = None,
                 hidden_loss_weights = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 10]):
        super().__init__()

        self.student = student.train()
        self.teacher_model = teacher_model

        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.args = args
        self.args.p = max(args.p, 1e-5)

        self.alpha = args.hard_label_loss_weight
        self.temperature = args.temperature

        self.step = 0

        sum_hidden_loss_weights = sum(hidden_loss_weights)
        self.hidden_loss_weights = [w / sum_hidden_loss_weights for w in hidden_loss_weights]

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loader(args, model_type)
        
        self.teacher_lm_head = nn.Linear(self.teacher_model.model.lm_head.in_features,
                                         self.teacher_model.model.lm_head.out_features,
                                         bias=(self.teacher_model.model.lm_head.bias is not None)
                                        ).to(device=self.student.device, 
                                             dtype=self.teacher_model.model.lm_head.weight.dtype)
        self.teacher_lm_head.load_state_dict(self.teacher_model.model.lm_head.state_dict())
        for p in self.teacher_lm_head.parameters():
            p.requires_grad = False

    def get_data_loader(self, args: Arguments, model_type: str):
        self.tokenizer = load_tokenizer(model_type, args.student_tokenizer, 
                                        args.load_student_tokenizer_kwargs)

        train_dataset = LLMDataset(args.train_data, self.tokenizer, args.syntactic_file, args.max_len // 2)

        train_collate = LLMDataCollator(self.tokenizer, model_type, do_train=True, max_len = args.max_len,
                                        pad_to_multiple_of = args.pad_to_multiple_of,
                                        return_tensors = 'pt', padding = True, 
                                        return_offsets_mapping = args.span_loss)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_collate)

        return train_loader, None, None


    def get_teacher_eval(self, inputs):
        outputs = self.teacher_model.decode(inputs)

        outputs.logits = outputs.logits.to(self.student.device, non_blocking=True)
  
        if outputs.hidden_states is not None:
            outputs.hidden_states = outputs.hidden_states.to(self.student.device, non_blocking=True)

        if outputs.span_states is not None:
            outputs.span_states = outputs.span_states.to(self.student.device, non_blocking=True)
            
        if outputs.span_weights is not None:
            outputs.span_weights=outputs.span_weights.to(self.student.device, non_blocking=True)

        return outputs

    def soft_label_distill_loss(self, student_logits, teacher_logits, mask, distill_temperature = 2.0):
        
        student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

        loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def fdd_loss(self, student_hidden_states, teacher_hidden_states, attention_mask):
        traj_loss, der_loss = 0, 0
        n_layer = teacher_hidden_states.size(0)

        pre_s_hidden_logs, pre_t_hidden_logs = None, None
        for i in range(n_layer):
            s_hidden = student_hidden_states[i]
            t_hidden = teacher_hidden_states[i]

            s_hidden_logits = self.student.model.lm_head(s_hidden)
            t_hidden_logits = self.teacher_lm_head(t_hidden)
            state_loss = self.soft_label_distill_loss(s_hidden_logits, t_hidden_logits, 
                                                      attention_mask, self.temperature)

            traj_loss += state_loss

            s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
            t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)
            if i > 0:
                delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
                delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs
                cos_sim = F.cosine_similarity(delta_hidden_student, 
                                              delta_hidden_teacher, 
                                              dim=-1, eps=1e-5)
                cos_sim_loss = 1 - cos_sim
                cos_sim_loss = (cos_sim_loss * attention_mask).sum() / attention_mask.sum()

                der_loss += cos_sim_loss

            pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs
            
        return traj_loss / n_layer, der_loss / (n_layer - 1)

    def span_fdd_loss(self, student_hidden_states, teacher_hidden_states, span_weights):
        traj_loss, der_loss = 0, 0
        n_layer = teacher_hidden_states.size(0)

        mask = span_weights[-1] != 0.0

        # pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        pre_s_hidden_logs, pre_t_hidden_logs = None, None
        count = 0
        for i in range(max(n_layer - 2, 0), n_layer):
            s_hidden = student_hidden_states[i]
            t_hidden = teacher_hidden_states[i]

            s_hidden_logits = self.student.model.lm_head(s_hidden)
            t_hidden_logits = self.teacher_lm_head(t_hidden)
            state_loss = self.soft_label_distill_loss(s_hidden_logits, t_hidden_logits, 
                                                      mask, self.temperature)

            s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
            t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)

            # s_hidden = F.normalize(s_hidden, dim=-1, eps=1e-5)
            # t_hidden = F.normalize(t_hidden, dim=-1, eps=1e-5)
            # student_scores = torch.matmul(s_hidden, s_hidden.transpose(-1, -2))
            # teacher_scores = torch.matmul(t_hidden, t_hidden.transpose(-1, -2))
            # state_loss = F.mse_loss(student_scores, teacher_scores, reduction='none')
            # state_loss = (state_loss * pair_mask).sum() / pair_mask.sum()

            traj_loss += state_loss

            if i > max(n_layer - 2, 0):
                delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
                delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs
                cos_sim = F.cosine_similarity(delta_hidden_student, 
                                              delta_hidden_teacher, 
                                              dim=-1, eps=1e-5)
                cos_sim_loss = 1 - cos_sim
                cos_sim_loss = (cos_sim_loss * mask).sum() / mask.sum()

                der_loss += cos_sim_loss

            pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs
            
            count += 1

        return traj_loss / count, der_loss / (count - 1)

    def knowledge_distillation_loss(self, student_outputs: StudentOutput,
                                    teacher_outputs: TeacherOutput=None, attention_mask=None):
        kd_loss = 0

        if teacher_outputs is not None:
            if teacher_outputs.hidden_states is not None:
                traj_loss, der_loss = self.fdd_loss(student_outputs.hidden_states, 
                                                    teacher_outputs.hidden_states, 
                                                    attention_mask)
                if self.args.span_loss:
                    span_traj_loss, span_der_loss = self.span_fdd_loss(student_outputs.span_states, 
                                                                    teacher_outputs.span_states, 
                                                                    teacher_outputs.span_weights.squeeze(-1))
                else:
                    span_traj_loss, span_der_loss = 0, 0

                kl_loss = self.soft_label_distill_loss(student_outputs.logits, teacher_outputs.logits, 
                                                       attention_mask, self.temperature)

                
                kd_loss = kl_loss + traj_loss + der_loss + span_traj_loss + span_der_loss
                # kd_loss = kl_loss + (traj_loss + der_loss + 50 * span_traj_loss) / 2

        return kd_loss, kl_loss
    
    def skewed_forward_kl(self, logits, teacher_logits, labels, lam=0.1):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        mixed_probs = lam * teacher_probs + (1-lam) * student_probs
        mixed_logprobs = torch.log(mixed_probs)
        
        mask = (labels != -100).int()
        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    
    def compute_loss(self, inputs, labels, teacher_outputs = None):
        student_outputs = self.student(inputs)
        attention_mask = inputs['attention_mask'].to(self.student.device, non_blocking=True)

        # kd_loss, kl_loss = self.knowledge_distillation_loss(student_outputs, teacher_outputs, 
        #                                                     attention_mask)

        kd_loss = self.skewed_forward_kl(student_outputs.logits, teacher_outputs.logits, labels)
        kl_loss = kd_loss

        return kd_loss, kl_loss
    

def train(args: Arguments, trainer: Trainer, evaluator: Evaluator, grad_accum_steps=1):
    trainer.student.train()

    train_loader = trainer.train_loader

    optimizer = optim.AdamW(trainer.student.parameters(), lr=args.learning_rate)

    num_steps = len(train_loader) // grad_accum_steps + 1
    total_traning_steps = num_steps * args.num_train_epochs

    scaler = GradScaler()

    # scheduler = get_scheduler(
    #     name='cosine_with_min_lr',
    #     optimizer=optimizer,
    #     num_warmup_steps=int(total_traning_steps * args.warmup_ratio),
    #     num_training_steps=total_traning_steps,
    #     scheduler_specific_kwargs={'min_lr': 1e-07}
    # )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_traning_steps, eta_min=1e-7)

    best_result = 0

    # Training loop
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
                torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_total += loss.item()
            student_loss_total += student_loss.item()
            step += 1

            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
            # s = ('%8s' + '%14s' + '%17.5g' * 2) % (f'{epoch + 1}/{args.num_train_epochs}', memory,
            #                                         loss_total / step, student_loss_total / step)
            s = ('%8s' + '%14s' + '%17.5g' * 2) % (f'{epoch + 1}/{args.num_train_epochs}', memory,
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
        
            # result = evaluator.evaluate_benchmark_dataset(
            #     dataset_path='./data/vicuna/valid.jsonl',
            #     dataset_name='vicuna', batch_size=16,
            #     max_seq_length=256, max_new_tokens=512)

            # result = evaluator.evaluate_benchmark_dataset(
            #     dataset_path='./data/self-inst/valid.jsonl',
            #     dataset_name='self-inst', batch_size=16,
            #     max_seq_length=256, max_new_tokens=512)
            
        if dolly > best_result:
            best_result = dolly
            trainer.student.save(args.output_dir)
            
        trainer.student.save(args.output_dir + f'-epoch{epoch}')
            
        

    

