# GPT2chinese

GPT2中文文本生成例子

注意由于输入时候没有使用special_token所以在inference时候需要把tokenizer的add_special_tokens=False ,否则生成的结果就只有[PAD]、[SEP]
