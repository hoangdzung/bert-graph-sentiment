import pickle
import torch 
import dgl 
import sys 
from tqdm import tqdm

filein = sys.argv[1]
fileout = sys.argv[2]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
graphs, a = pickle.load(open(filein,'rb'))

for i in tqdm(range(len(graphs))):
    graphs[i].edata['rel_type'] = graphs[i].edata['rel_type'].to(device)
    graphs[i].edata['norm'] = graphs[i].edata['norm'].to(device)

pickle.dump([graphs,a], open(fileout,'wb'))