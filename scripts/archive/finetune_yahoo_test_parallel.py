"""
Script to test fine-tuning an LLM (Llama-3.2-1.1B-Instruct)

This is a unit test. It trains on a SMALL subset of the data it would train on
otherwise. 
"""

# imports
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import  AutoModelForCausalLM, AutoTokenizer, \
                          DataCollatorForLanguageModeling, TrainingArguments, \
                          Trainer           
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import os

# custom imports 
from src.utils import *
from src.data import make_small_train_val_test, group_examples

###############################################################################

# identifiers
count = 3

run_name = f'test_yahoo_lora{count}'
dirname = f'test_finetune_llama_yahoo9_{count}'

# Basic experiment info
model_id = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name = "yahoo_answers_topics"
context_length = 512
topic_list = [9]

# set environment
os.environ["WANDB_PROJECT"] = "model-manifolds-finetune" # project location
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 
os.environ['HF_HOME'] = '/weka/home/mohata1/scratchcpriebe1/MO/huggingface_cache'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################

# create the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", 
                                             device_map="auto",)
                                            #  tp_plan="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# get the dataset
yahoo_train = load_dataset(dataset_name, split="train")
yahoo_test = load_dataset(dataset_name, split="test")


# preprocess the datasets
def concat_question(example):
    return {"example" : example['question_title'] + " " + \
            example['question_content'] + " " + example['best_answer']}

def tokenize_elements(element):
    return tokenizer(element['example'])

print(f'Filtering data for topics {topic_list}...')
# filter data first
filter_train = yahoo_train.filter(lambda e: e['topic'] in topic_list)
filter_test = yahoo_test.filter(lambda e: e['topic'] in topic_list)

print(f'Downsampling data and concatenating questions...')
# sample down (small!)
# small_train, small_eval, small_test = make_small_train_val_test(filter_train, 
#                                                                 filter_test,
                                                        
#                                                         map=concat_question)

# keep full size data set :)
val_set_size = int(len(filter_train)*0.2) 

small_train = filter_train.shuffle(seed=0).\
                            select(range(int(len(filter_train))-val_set_size))\
                            .map(concat_question)
small_eval = filter_train.shuffle(seed=0).\
                            select(range(int(len(filter_train))-val_set_size, 
                                         int(len(filter_train))))\
                            .map(concat_question)
small_test = filter_test.shuffle(seed=0).map(concat_question) 

print('Data set sizes')
print(f"Training: {len(small_train)}\nEvaluation: {len(small_eval)}\nTesting: {len(small_test)}")

print(f'Tokenizing datasets...')
# tokenize and prepare to batch
data_train = small_train.map(tokenize_elements, batched=True, 
                             remove_columns=small_train.column_names)\
                                .map(group_examples, batched=True, num_proc=4)
data_eval = small_eval.map(tokenize_elements, batched=True, 
                           remove_columns=small_train.column_names)\
                            .map(group_examples, batched=True, num_proc=4)
data_test = small_test.map(tokenize_elements, batched=True, 
                           remove_columns=small_train.column_names)\
                            .map(group_examples, batched=True, num_proc=4)

# make collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

###############################################################################

# Set up for fine-tuning
print("Making trainer settings...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_proj','k_proj', 'v_proj', 'o_proj'],
    modules_to_save=None,
) 

set_seed(0)

model_tune = get_peft_model(model, lora_config)

# make the training arguments
train_args = TrainingArguments(
    report_to='wandb', # log to wandb
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    num_train_epochs=5,
    run_name=run_name,
    remove_unused_columns=False,
    output_dir=f'results/{dirname}',

    logging_dir=f'results/logs/{dirname}',
    logging_steps = 0.5,
    logging_first_step=True,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=0.5,
    eval_steps=0.5,
    seed=0,   

    label_names="labels", 
)

print("Training model...")
# make the trainer :)
trainer = Trainer(
    model=model_tune,
    args=train_args,
    train_dataset=data_train,
    eval_dataset=data_eval,
    tokenizer=tokenizer
)

trainer.train()



torch.distributed.destroy_process_group()