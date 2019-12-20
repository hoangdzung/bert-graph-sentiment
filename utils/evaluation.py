import torch
import numpy as np
from tqdm import tqdm

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_baseline_acc(model, validation_dataloader, device):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy/nb_eval_steps
    return eval_accuracy


def get_bert_rgcn_acc(model, validation_dataloader, device):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in tqdm(validation_dataloader, desc='eval'):
        b_input_graphs = batch[0]
        b_input_graphs.edata['rel_type'] = b_input_graphs.edata['rel_type'].to(device)
        b_input_graphs.edata['norm'] = b_input_graphs.edata['norm'].to(device)
        b_input_ids = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_input_lens = batch[3].to(device)
        b_labels = batch[4].to(device)
        
        with torch.no_grad():        
             outputs = model(b_input_graphs, b_input_ids, b_input_mask, b_input_lens, b_labels)
        
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy/nb_eval_steps
    return eval_accuracy
