import torch
import argparse
from model import BERT_Arch
from config import config
import json
import random

def get_response(data,intent):
    for d in data:
        if d['intent'] == intent:
            return d['responses'][random.randint(0,len(d['responses'])-1)]

if __name__ == '__main__':

    argparse = argparse.ArgumentParser()
    argparse.add_argument('--text',required=True)

    arg = argparse.parse_args()
    if len(arg.text) == 0:
        raise ValueError('--text should be in the range of (0,11]')

    data = json.load(open(config['src_file_path'],'r'))['intents']

    model = BERT_Arch()
    cp = torch.load(config['cp_path'])
    state_dict = cp['param']
    model_stat = model.state_dict()
    
    for p in model_stat:
        if p in state_dict:
            model_stat[p] = state_dict[p]
    
    model.load_state_dict(model_stat)

    res = model.inference(arg.text)
    print({
        'intent':res[0],
        'response':get_response(data,res[0]),
        'confidence':res[1]
    })