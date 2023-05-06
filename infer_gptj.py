import os

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
#from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



checkpoint = "/home/jovyan/summarization_data/gpt2_chinese"
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.config.pad_token_id = tokenizer.pad_token
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

from rouge import Rouge
rouge_scorer = Rouge()

def inference(model, tokenizer):
    model.to("cuda")
    model.eval()

    pred_list = []
    summarize_list = []
    post_list = []
    # rouge = evaluate.load("rouge")
    count = 0
    for post, summarize in tqdm(zip(test_post_list, test_summ_list), total=len(test_post_list)):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True,add_special_tokens=False)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        #kwargs = {"max_new_tokens": 50, "eos_token_id": 0, "pad_token_id": 0}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, max_length=64,pad_token_id=0)
        #summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens,skip_special_tokens=True)[0]
        #print(pred)
        pred = pred.split("小 爱 同 学 :")[1].replace("<|endoftext|>", "")
        pred_list.append(pred)
        summarize_list.append(' '.join(list(summarize)))
        #print(pred_list[0])
        #print(summarize_list[0])
        #sys.exit(1)
        post_list.append(post)
        if count % 10 == 0:
            result = rouge_scorer.get_scores(pred_list,summarize_list)
            #print(result)
        count += 1
    scores = rouge_scorer.get_scores(pred_list,summarize_list,avg=True)
    writer = open('pred_gpt_list_sft.txt','a+',encoding='utf-8')
    for post,summary,target in zip(post_list,summarize_list,pred_list):
        writer.write(post+'\t'+summary+'\t'+target+'\n')
    #print(result)
    writer.close()
    for key in scores:
        scores[key] = scores[key]['f'] * 100
    result = scores
    print(result)
    sys.exit(1)
    df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list})

    return df




def inference_batches(model, tokenizer, test_post_list, test_summ_list, batch_size=16):
    model.to("cuda")
    model.eval()

    pred_list = []
    summarize_list = []
    post_list = []
    # rouge = evaluate.load("rouge")

    # Iterate over the input data in mini-batches
    for i in tqdm(range(0, len(test_post_list), batch_size)):
        batch_post_list = test_post_list[i : i + batch_size]
        batch_summ_list = test_summ_list[i : i + batch_size]

        # Convert input data to tensors
        encode_dict = tokenizer(
            batch_post_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()

        # Perform inference on the batch
        kwargs = {"max_new_tokens": 50, "eos_token_id": 50256, "pad_token_id": 0}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)

        # Decode output tokens
        preds = tokenizer.batch_decode(summ_tokens)

        # Add predictions, truths, and input posts to lists
        pred_list += preds
        summarize_list += batch_summ_list
        post_list += batch_post_list

        # Compute rouge scores every 10 mini-batches
        # result = rouge_scorer.get_scores(pred_list,summarize_list)
        # print(result)

    # Compute final rouge scores and create a dataframe
    result = rouge_scorer.get_scores(pred_list, summarize_list)
    print(result)
    df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list})
    return df


if __name__ == "__main__":

    import json


    def convert_data(dataset):
        new_dataset = []
        for d in dataset:
            best_diff = 0
            best = None
            for reply in d['replys']:
                t_diff = reply['like'] - reply['dislike']
                if t_diff > best_diff:
                    best_diff = t_diff
                    best = {'query': d['query'], 'reply': reply['reply'], 'dislike': reply['dislike'],
                            'like': reply['like'], 'difference': reply['like'] - reply['dislike']}
            if best:
                best['prompt'] = f"""用户:{best["query"]}#小爱同学:"""
                best['label'] = best['reply']
                new_dataset.append(best)
        return new_dataset


    dataset = []
    data_path = './nlpcc-2023-shared-task-9/datasets/'
    train_file_name = 'datasets_train.jsonl'
    dev_file_name = 'datasets_dev.jsonl'

    train_data = [json.loads(lin) for lin in open(data_path + '/' + train_file_name, 'r', encoding='utf-8').readlines()]
    dev_data = [json.loads(lin) for lin in open(data_path + '/' + dev_file_name, 'r', encoding='utf-8').readlines()]

    converted_train_data = convert_data(train_data)
    converted_dev_data = convert_data(dev_data)

    model, tokenizer = load_model("./gpt2-supervised-text-checkpoint/")

    test_post_list = [sample["prompt"] for sample in converted_dev_data]
    test_summ_list = [sample["label"] for sample in converted_dev_data]

    df_result = inference(model, tokenizer)
