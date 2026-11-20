from pathlib import Path
import csv,hashlib,json
import numpy as np
root=Path(__file__).resolve().parent
meta=json.loads((root/'expected.json').read_text())
raw=list(csv.DictReader((root/'frames_0.csv').open()))
with np.load(root/'cut_seed0_scores.npz') as z:
    score=z['scores'];o=z['opacity_order']
for name in ['O','pool50','pool100']:
    pool=o[:round(len(o)*0.5)] if name=='pool50' else np.arange(len(o))
    order=o if name=='O' else pool[np.lexsort((pool,-score[pool]))]
    for budget in [10,15]:
        ids=np.asarray(np.sort(order[:round(len(o)*budget/100)]),dtype='<i8')
        ref=next(r for r in meta if r['method']==name and r['budget']==budget)
        assert hashlib.sha256(ids.tobytes()).hexdigest()==ref['membership']
        ps=[float(r['psnr']) for r in raw if r['method']==name and int(r['budget'])==budget]
        assert len(ps)==300 and abs(np.mean(ps)-ref['psnr'])<1e-8
        print(name,budget,len(ids),round(np.mean(ps),4))
