#!/usr/bin/env python
# 보완 #3 최종: fixed(1,1,1) fusion에서 z-norm을 LOCO validation-good으로 fit (test 통계 무사용) → 완전 표준 프로토콜.
import numpy as np, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
R=Path("/workspace/ai-vision-research"); VS=R/"reports"/"countgd"/"val_scores"
EADD=R/"reports"/"phase0"
CATS=["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]; SEEDS=["seed0","seed42","seed1234"]
def ead_val(cat):
    a=[]
    for sd,base in [("seed0","efficient_ad_official_small_seeds"),("seed1234","efficient_ad_official_small_seeds")]:
        a.append(np.load(EADD/base/f"{cat}_{sd}"/"val_good_scores_v2.npz")["scores"])
    a.append(np.load(EADD/"efficient_ad_official_small"/f"{cat}_seed42_v1"/"val_good_scores_v2.npz")["scores"])
    return np.mean(a,0)
def ead_test(cat):
    return np.mean([np.load(EADD/"efficient_ad_official_small"/"npz"/f"scores_{cat}_seed{s}.npz")["score"] for s in [0,42,1234]],0)
def labels_types(cat):
    z=np.load(R/"algml_v6_5"/"aupr_bootstrap"/"scores"/f"scores_{cat}_seed42.npz",allow_pickle=True)
    return z["label"],z["label_type"]
def psad_test_aligned(cat,tag,ltypes):
    z=np.load(VS/f"psad_{cat}_{tag}_testraw.npz",allow_pickle=True)
    d={(str(t),str(f)):s for s,t,f in zip(z["score"].astype(float),z["ltype"],z["fname"])}
    order=[(t,fn) for t in ["good","logical","structural"] for fn in sorted([k[1] for k in d if k[0]==t])]
    arr=np.array([d[o] for o in order]); assert (np.array([o[0] for o in order])==ltypes).all(); return arr
res={"variants":{}}
for tag in ["official"]+SEEDS:
    loco_val=[]; loco_test=[]
    for cat in CATS:
        labels,ltypes=labels_types(cat); g=(ltypes=="good")
        srcs_test=[ead_test(cat), np.load(VS/f"pc_{cat}_test_scores.npz")["test_trainonly"], psad_test_aligned(cat,tag,ltypes)]
        srcs_val=[ead_val(cat), np.load(VS/f"pc_{cat}_val.npz")["score"], np.load(VS/f"psad_{cat}_{tag}_val.npz")["score"]]
        fV=sum((t-v.mean())/max(v.std(),1e-9) for t,v in zip(srcs_test,srcs_val))
        fT=sum((t-t[g].mean())/max(t[g].std(),1e-9) for t in srcs_test)
        for f,acc in [(fV,loco_val),(fT,loco_test)]:
            acc.append(0.5*sum(roc_auc_score(labels[(ltypes=="good")|(ltypes==k)],f[(ltypes=="good")|(ltypes==k)]) for k in ["logical","structural"]))
    res["variants"][tag]={"val_znorm":round(float(np.mean(loco_val)),4),"test_good_znorm":round(float(np.mean(loco_test)),4)}
    print(f"  [{tag}] val-znorm {np.mean(loco_val):.4f} | test-good-znorm {np.mean(loco_test):.4f}",flush=True)
ms=[res["variants"][s]["val_znorm"] for s in SEEDS]
res["multiseed_val_znorm"]={"mean":round(float(np.mean(ms)),4),"std":round(float(np.std(ms,ddof=1)),4)}
json.dump(res,open(R/"reports"/"countgd"/"val_znorm_fusion.json","w"),indent=2)
print(f"\nval-znorm fixed 3-seed: {res['multiseed_val_znorm']['mean']}±{res['multiseed_val_znorm']['std']} (test-good 참조 0.9684±0.0043)")
