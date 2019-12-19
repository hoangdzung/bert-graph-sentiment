import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup
import numpy as np
import argparse
from tqdm import tqdm 
import random
from utils import get_dataloader, get_acc

parser = argparse.ArgumentParser()
parser.add_argument('--data_file')
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=2e-5)

args = parser.parse_args()

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

random.seed(args.seed))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)

train_dataloader, validation_dataloader, test_dataloader = get_dataloader(args.datafile, args.batch_size, tokenizer)

optimizer = AdamW(model.parameters(),lr = args.lr)
total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

loss_values = []
best_eval_acc = 0
test_acc = 0

for epoch_i in range(0, epochs):
    total_loss = 0
    model.train()
        
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        model.train()
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
                
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
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
        

print("")
print("Training complete!")