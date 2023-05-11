# %% [code] {"execution":{"iopub.status.busy":"2023-04-20T10:25:35.940848Z","iopub.execute_input":"2023-04-20T10:25:35.941249Z","iopub.status.idle":"2023-04-20T10:25:35.948070Z","shell.execute_reply.started":"2023-04-20T10:25:35.941191Z","shell.execute_reply":"2023-04-20T10:25:35.946892Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!!pip install datasets transforemrs 

!git clone https://github.com/XiaoMi/nlpcc-2023-shared-task-9.git

import json 
def convert_data(dataset):
  new_dataset = []
  for d in dataset:
      best_diff = 0
      best = None
      for reply in d['replys']:
          t_diff = reply['like']-reply['dislike']
          if t_diff>best_diff:
            best_diff = t_diff 
            best = {'query':d['query'],'reply':reply['reply'],'dislike':reply['dislike'],'like':reply['like'],'difference':reply['like']-reply['dislike']}
      if best:
        new_dataset.append(best)
  return new_dataset


dataset = []
data_path = './nlpcc-2023-shared-task-9/datasets/'
train_file_name = 'datasets_train.jsonl'
dev_file_name = 'datasets_dev.jsonl'

train_data = [json.loads(lin) for lin in open(data_path+'/'+train_file_name,'r',encoding='utf-8').readlines()]
dev_data = [json.loads(lin) for lin in open(data_path+'/'+dev_file_name,'r',encoding='utf-8').readlines()]

converted_train_data = convert_data(train_data)
converted_dev_data = convert_data(dev_data)

import gc 
gc.collect()


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)


print(tokenizer.eos_token_id)

def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""用户:{data_point["query"]}#小爱同学:"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=64,
            )["input_ids"]
        )
        - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["reply"],
        truncation=True,
        max_length=128 + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


!pip install pandas 
import pandas as pd 
import datasets 
# train_data = pd.DataFrame(converted_train_data)
train_dataset =datasets.Dataset.from_pandas(pd.DataFrame(converted_train_data))
dev_dataset = datasets.Dataset.from_pandas(pd.DataFrame(converted_dev_data))



# %% [code] {"execution":{"iopub.status.busy":"2023-04-20T10:27:53.241018Z","iopub.execute_input":"2023-04-20T10:27:53.241802Z","iopub.status.idle":"2023-04-20T10:27:54.709884Z","shell.execute_reply.started":"2023-04-20T10:27:53.241756Z","shell.execute_reply":"2023-04-20T10:27:54.708337Z"}}
train_data = train_dataset.shuffle().map(generate_and_tokenize_prompt)
val_data = dev_dataset.shuffle().map(generate_and_tokenize_prompt)


del train_dataset
del dev_dataset
del converted_train_data
del converted_dev_data

! pip install accelerate==0.18.0


# %% [code] {"execution":{"iopub.status.busy":"2023-04-20T10:25:50.698031Z","iopub.status.idle":"2023-04-20T10:25:50.698558Z","shell.execute_reply.started":"2023-04-20T10:25:50.698290Z","shell.execute_reply":"2023-04-20T10:25:50.698315Z"}}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto'
#     load_in_8bit=True,
)
MICRO_BATCH_SIZE = 16  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't always need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 128  # 256 accounts for about 96% of the data

# %% [code] {"execution":{"iopub.status.busy":"2023-04-20T10:25:50.700466Z","iopub.status.idle":"2023-04-20T10:25:50.700966Z","shell.execute_reply.started":"2023-04-20T10:25:50.700709Z","shell.execute_reply":"2023-04-20T10:25:50.700734Z"}}
!pip install evaluate rouge_score

# %% [code] {"execution":{"iopub.status.busy":"2023-04-20T10:25:50.702887Z","iopub.status.idle":"2023-04-20T10:25:50.703547Z","shell.execute_reply.started":"2023-04-20T10:25:50.703287Z","shell.execute_reply":"2023-04-20T10:25:50.703312Z"}}
import evaluate

rouge = evaluate.load("rouge")

"""Then create a function that passes your predictions and labels to [compute](https://huggingface.co/docs/evaluate/main/en/package_reference/main_classes#evaluate.EvaluationModule.compute) to calculate the ROUGE metric:"""

import numpy as np


def compute_metrics(eval_pred):
    pred, labels = eval_pred
    predictions = pred[0]
    t_labels = pred[1]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    print(decoded_preds[0]+'\t'+decoded_labels[0])

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

import torch
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    if isinstance(logits, tuple):   
        logits = logits[0]
    return logits.argmax(dim=-1),labels


import transformers
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
#         eval_steps=50,
#         save_steps=100,
        output_dir="BLOOM-alpaca",
        save_total_limit=1,
        report_to="tensorboard",
        load_best_model_at_end=True,
        eval_accumulation_steps=1,
        per_device_eval_batch_size=4,

    ),
    data_collator=transformers.DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics

)

trainer.train()
