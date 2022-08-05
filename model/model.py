import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, logging

class BERT_Arch(torch.nn.Module):
    def __init__(self, bert=None):      
        super(BERT_Arch, self).__init__()
        self.max_len = 11
        self.bert = bert
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        if not bert:
            self.bert = AutoModel.from_pretrained('bert-base-uncased')

            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,22)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        #define the forward pass
    
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)

        # apply softmax activation
        x = self.softmax(x)
        return x
    
    def inference(self, text):
        
        logging.set_verbosity_error()
        
        label = ['Clever', 'CourtesyGoodBye', 'CourtesyGreeting',
               'CourtesyGreetingResponse', 'CurrentHumanQuery', 'GoodBye',
               'Gossip', 'Greeting', 'GreetingResponse', 'Jokes', 'NameQuery',
               'NotTalking2U', 'PodBayDoor', 'PodBayDoorResponse',
               'RealNameQuery', 'SelfAware', 'Shutup', 'Swearing', 'Thanks',
               'TimeQuery', 'UnderstandQuery', 'WhoAmI']
        with torch.no_grad():
            t= self.tokenizer(text, padding=True,truncation=True, return_tensors='pt',max_length=self.max_len)
            token, mask = t['input_ids'], t['attention_mask']
            res = self.forward(token, mask)
            return label[res.argmax()], torch.exp(res.max())