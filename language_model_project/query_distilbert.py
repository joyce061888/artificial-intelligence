import os
import sys
import math
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch

"""
Author: Carolyn Anderson
Date: 4/29/22
"""

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

def get_most_likely_cloze(textprompt):
  """get the most likely fill-in-the-blank option according to DistilBERT"""
  before,after = textprompt.split('BLANK')
  text = before+'[MASK]'+after
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
    logits = model(**inputs).logits
  mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
  predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
  most_likely_word = tokenizer.decode(predicted_token_id)
  most_likely_prob = torch.nn.functional.softmax(logits[0, mask_token_index]).amax(axis=-1).item()
  return (most_likely_word.strip(),most_likely_prob)

def assess_cloze_probability(textprompt,choices):
  """assess fill-in-blank probability of all choices"""
  probs = []
  before,after = textprompt.split('BLANK')
  for c in choices:
    c_idx = tokenizer(c)['input_ids']
    c_len = len(c_idx)-2
    text = before+'[MASK] '*(c_len-1)+'[MASK]'+after
    inputs = tokenizer(text, return_tensors="pt")
    labels = tokenizer(before+c+after, return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss.item()
    probs.append(math.exp(-loss)) # loss is negative log likelihood. so, multiply by -1 and apply inverse of log function
  return probs

def choice_query(prompt,choices):
  if 'BLANK' not in prompt:
    print("Prompt must include a BLANK")
    return
  wordlist = choices.split(';')
  best_word = get_most_likely_cloze(prompt)
  wordlist.append(best_word[0])
  probs = assess_cloze_probability(prompt,wordlist)
  return {w:probs[i] for i,w in enumerate(wordlist)}

def main():
  prompt = sys.argv[1] 
  choices = sys.argv[2]
  probs = choice_query(prompt,choices)
  print(probs)
  
main()