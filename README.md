# Multi Chatbox: The Cherokee Language Preserve Model

## Overview
I am excited to present the latest language model, which has been  fine-tuned using the state-of-the-art LoRA (Low-Rank Adaptation) technique on the robust foundation of the LLaMA3-8B model. This fine-tuning process has been specifically tailored to enhance the model's performance on Cherokee language translation tasks, setting a new standard in the field.
<img width="1506" alt="image" src="https://github.com/user-attachments/assets/bc944e9c-3cd2-4d1e-aa53-3598499656d2">

## Data Sets Utilized
This model has been trained on two specialized datasets build by myself to ensure its proficiency in Cherokee-English translation:

1. **Cherokee-English Bible Sentence (7.96k)**  [Dataset Link](https://huggingface.co/datasets/wang4067/cherokee-english-bible-7.96k)  
   This dataset provides a rich source of bilingual text, enabling our model to understand and reproduce the nuances of the Cherokee language within a religious context.

2. **Cherokee-English Word (10.2k)**  [Dataset Link](https://huggingface.co/datasets/wang4067/cherokee-english-word-10.2k)  
   This dataset focuses on vocabulary, ensuring that our model has a comprehensive grasp of Cherokee words and their English counterparts.

## Performance Achievements
This model has demonstrated exceptional performance in Cherokee language translation tasks, surpassing mainstream models such as LLaMA3-8B, LLaMA3.1-8B, and PHI3. It has achieved state-of-the-art (SOTA) results without the common issue of catastrophic forgetting.

Here are some details about performance.
```shell
{
    "predict_bleu-4": 96.79794598214286,
    "predict_rouge-1": 98.21964419642859,
    "predict_rouge-2": 97.57667857142857,
    "predict_rouge-l": 98.36520848214286,
    "predict_runtime": 93.1528,
    "predict_samples_per_second": 2.147,
    "predict_steps_per_second": 0.075
}
```
Here are some details about this training process.
```shell
bf16: true
cutoff_len: 1024
dataset: dict_word_v4,dict_sentence_v4
dataset_dir: data
ddp_timeout: 180000000
do_train: true
finetuning_type: lora
flash_attn: auto
gradient_accumulation_steps: 8
include_num_input_tokens_seen: true
learning_rate: 0.0001
logging_steps: 5
lora_alpha: 16
lora_dropout: 0.1
lora_rank: 8
lora_target: all
lr_scheduler_type: cosine
max_grad_norm: 1.0
max_samples: 100000
model_name_or_path: /wsh/models/Meta-Llama-3-8B-Instruct
num_train_epochs: 40.0
optim: adamw_torch
output_dir: saves/Custom/lora/train_2024-09-15-17-54-11-v4-learn_rate_0001
packing: false
per_device_train_batch_size: 2
plot_loss: true
preprocessing_num_workers: 16
report_to: none
save_steps: 100
stage: sft
warmup_steps: 0
```
