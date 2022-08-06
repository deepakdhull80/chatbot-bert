from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import BERT_Arch
from config import config
import json
import os
import random

app = FastAPI()
service_name = 'chatbot'

class Chat(BaseModel):
    message:str

def get_response(data,intent):
    for d in data:
        if d['intent'] == intent:
            return d['responses'][random.randint(0,len(d['responses'])-1)]


data = json.load(open(config['src_file_path'],'r'))['intents']

model = BERT_Arch()
if not os.path.exists(config['cp_path']):
    print("cp need to be downloaded")
    raise FileNotFoundError(config['cp_path'])

cp = torch.load(config['cp_path'])
state_dict = cp['param']
model_stat = model.state_dict()

for p in model_stat:
    if p in state_dict:
        model_stat[p] = state_dict[p]

model.load_state_dict(model_stat)


@app.post(f"/{service_name}/chat")
def communicate_chat(chat:Chat):
    msg = chat.message
    res = model.inference(msg)
    return {
        'intent':res[0],
        'response':get_response(data,res[0]),
        'confidence':res[1].item()
    }