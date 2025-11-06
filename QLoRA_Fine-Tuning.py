import os
import torch
from transformers import (
    AutoModelForCausalLM,   # 모델 로드 라이브러리
    AutoTokenizer,          # 모델에 맞는 Tokenizer를 로드하기 위한 라이브러리
    BitsAndBytesConfig      # 양자화 기법을 사용하기 위한 라이브러리
)

# 모델 튜닝을 위한 라이브러리
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 로컬에 있는 데이터를 불러오기 위함
from datasets import load_from_disk

# 경로 설정
dire = os.getcwd()

# sentence_data = load_from_disk(f"{dire}\\sentence_data")
word_data = load_from_disk(f"{dire}\\word_data")
# medical_data = load_from_disk(f"{dire}\\medical_data")

# LLM 모델과 데이터
model_name = "tencent/Hunyuan-MT-7B"
dataset_name = word_data
new_model = "finetuned_model" 

# LoRA 파라미터 설정 (파인튜닝)

# low-rank matrices 어텐션 차원을 정의
# 크기가 충분히 크고 모든 가중치 행렬에 적용한다면 이론상 full fine-tuning이 된다.
# rank는 1~4로도 충분한 성능을 보여준다.
lora_r = 4
lora_alpha = 16  
lora_dropout = 0.1

# BitsAndBytesConfig 파라미터 설정 (4비트 양자화 지원)
use_4bit = True
bnb_4bit_compute_dtype = "float16"
# QLoRA는 LoRA의 가중치를 NormalFloat이라는
# FP4를 변형한 자료형을 사용하여 4비트 양자화한다. 
bnb_4bit_quant_type = "nf4"
use_nested_quant = False 

# 데이터 타입 결정
# GPU 버전 8 이상(major >= 8) bfloat16 지원. 
# bfloat16은 훈련 속도를 높일 수 있는 데이터 타입.
if use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        bnb_4bit_compute_dtype = torch.bfloat16
    else:
        bnb_4bit_compute_dtype = torch.float16

# TrainingArguments 파라미터
output_dir = "./results(lora_r=32)"
num_train_epochs = 1

fp16 = False
bf16 = True

per_device_train_batch_size = 1

gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3

learning_rate = 2e-6
weight_decay = 0.001
optim = "paged_adamw_32bit"  
lr_scheduler_type = "cosine"   
max_steps = -1 
warmup_ratio = 0.03 

group_by_length = True
save_steps = 0  
logging_steps = 25

# SFT(Supervised Fine-Tuning)Trainer 파라미터
max_seq_length = None 
packing = False  
device_map = {"": 0}

# 모델 계산에 사용될 데이터 타입 결정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,  # 모델을 4비트로 로드할지 결정
    bnb_4bit_quant_type=bnb_4bit_quant_type, # 양자화 유형을 설정
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  # 계산에 사용될 데이터 타입을 설정
    bnb_4bit_use_double_quant=use_nested_quant, # 중첩 양자화를 사용할지 여부를 결정
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLM tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 동일한 batch 내에서 입력의 크기를 동일하게 하는 Padding Token(EOS).
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",              # 파인튜닝할 태스크를 Optional로 지정할 수 있는데, 여기서는 CASUAL_LM을 지정하였다.
    target_modules=["q_proj","v_proj"]  # 파인튜닝 레이어를 연결할 모듈
    # q, k, v, o -> attention layer, gate, up, down -> dense layer
    # 주요 표현을 나타내는 레이어인 q, v에만 QLoRA 파인튜닝을 적용한다.
)

# Training parameters
training_arguments = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    max_seq_length=max_seq_length,
    packing=packing,
    dataset_text_field="text",
)

# Supervised Fine-Tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_name,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

# 훈련이 완료된 LoRA 어댑터만 'new_model'에 저장
trainer.model.save_pretrained(new_model)

# # base_model과 new_model에 저장된 LoRA 가중치를 통합하여 새로운 모델을 생성
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16
# )
#
# # base모델과 LoRA 어댑터를 붙임
# model = PeftModel.from_pretrained(base_model, new_model) # LoRA 가중치를 가져와 기본 모델에 통합

# # base모델과 LoRA 어댑터를 붙여서 실제 하나의 모델로 저장
# model = model.merge_and_unload()