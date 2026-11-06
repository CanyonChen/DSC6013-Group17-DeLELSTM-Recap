import sys, os, importlib, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
name=sys.argv[1]
sys.argv=[name+'.py']
m=importlib.import_module(name)
from networks import Delelstm
import config
_,_,loader=m.prepare(555)
model=Delelstm(config.config('Delelstm',m.args),0)
folder={'Electricity':'Electricity','PM':'newrevision_pmc','Exchange':'Exchange'}[name]
suffix={'Electricity':'elec','PM':'pm','Exchange':'exchange'}[name]
p=Path('../results')/folder/'exp_0'/'Delelstm'/('0_'+suffix+'_predict.pt')
model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));model.eval()
X=[];Y=[]
with torch.no_grad():
 for x,y in DataLoader(loader.dataset,batch_size=m.args.batch_size,shuffle=False,drop_last=False):
  pred,_,_=model(x.float(),torch.device('cpu'))
  X.append(pred.squeeze(-1).numpy());Y.append(y.numpy())
pred=np.concatenate(X); true=np.concatenate(Y); n=len(loader)*m.args.batch_size
for title,sl in [('retained',slice(None,n)),('omitted',slice(n,None)),('all',slice(None))]:
 err=pred[sl]-true[sl]
 print(json.dumps(dict(dataset=name,subset=title,windows=len(err),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(np.abs(err))))))
