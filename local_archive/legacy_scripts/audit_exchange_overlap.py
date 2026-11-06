"""Count test windows sharing observations with a training window (fixed split)."""
import numpy as np
import pandas as pd
from pathlib import Path
ids=np.array(list(set(pd.read_csv(Path(__file__).parent/'DATA/Exchange/newX_train.csv',usecols=['idx']).idx)))
np.random.seed(555)
train=np.random.choice(ids,int(.75*len(ids)),replace=False)
remaining=ids[~np.isin(ids,train)]
val=np.random.choice(remaining,int(.15*len(ids)),replace=False)
test=np.sort(remaining[~np.isin(remaining,val)])
train.sort(); pos=np.searchsorted(train,test)
dist=np.minimum(abs(test-train[np.maximum(pos-1,0)]),abs(test-train[np.minimum(pos,len(train)-1)]))
for label, subset in [('all_test',dist),('retained_by_drop_last',dist[:len(dist)//128*128])]:
 print(label,'total',len(subset),'overlap_under_30_days',int((subset<30).sum()),'adjacent_day',int((subset==1).sum()))
