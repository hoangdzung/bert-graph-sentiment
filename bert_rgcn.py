import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup
#from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
import numpy as np
import argparse
from tqdm import tqdm 
import random
from model import BERT_RGCN
from utils.dataset.bert_rgcn import get_bert_rgcn_dataloader
from utils import get_acc

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='data/SST2.pkl')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--out_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=2e-5)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = BertModel.from_pretrained("bert-base-uncased", num_labels=2)
model = BERT_RGCN(args.hidden_size, args.out_size, 2, bert_model)
model = model.to(device)

train_dataloader, validation_dataloader, test_dataloader = get_bert_rgcn_dataloader(args.data_file, args.batch_size, tokenizer)

optimizer = AdamW(model.parameters(),lr = args.lr)
total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            #warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
                                            #t_total = total_steps)

loss_values = []
best_eval_acc = 0
test_acc = 0

for epoch_i in range(0, args.epochs):
    total_loss = 0
    model.train()
        
    # For each batch of training data...
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.train()
        
        b_input_graphs = batch[0]
        b_input_graphs.edata['rel_type'] = b_input_graphs.edata['rel_type'].to(device)
        b_input_graphs.edata['norm'] = b_input_graphs.edata['norm'].to(device)
        b_input_ids = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_input_lens = batch[3].to(device)
        b_labels = batch[4].to(device)
        outputs = model(b_input_graphs, b_input_ids, b_input_mask, b_input_lens, b_labels)

        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    eval_accuracy = get_acc(model, validation_dataloader, device)
    if eval_accuracy > best_eval_acc:
        best_eval_acc = eval_accuracy
        test_accuracy = get_acc(model, test_dataloader, device)
    print(" Val acc {}, test acc {}".format(eval_accuracy, test_accuracy))

print("")
print("Training complete!")
