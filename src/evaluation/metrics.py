from collections import defaultdict


def compare_scores(baseline, candidate, samples=10000):
    import numpy as np
    old={row["id"]:row for row in baseline}
    new={row["id"]:row for row in candidate}
    if not old or old.keys()!=new.keys() or len(old)!=len(baseline) or len(new)!=len(candidate):
        raise ValueError("Comparison requires identical unique nonempty case sets")
    groups=defaultdict(list)
    unchanged=0
    for case_id in old:
        a,b=old[case_id],new[case_id]
        for field in ("question","card_version","judge_signature"):
            if a.get(field)!=b.get(field) or a.get(field) is None:
                raise ValueError(f"Different {field} for {case_id}")
        if "answer" in a and "answer" in b and a["answer"]==b["answer"]:
            lo=hi=0
            unchanged+=1
        else:
            lo=(0 if b["score"] is None else b["score"])-(2 if a["score"] is None else a["score"])
            hi=(2 if b["score"] is None else b["score"])-(0 if a["score"] is None else a["score"])
        groups[a["source_group_id"]].append((lo,hi))
    totals=np.array([np.sum(values,axis=0) for values in groups.values()])
    counts=np.array([len(values) for values in groups.values()])
    draws=np.random.default_rng(42).integers(0,len(groups),size=(samples,len(groups)))
    boot=50*totals[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
    return {"cases":len(old),"source_groups":len(groups),"unchanged_answers":unchanged,"delta_score_bounds":(50*totals.sum(axis=0)/len(old)).tolist(),
            "paired_bootstrap_with_unresolved_95_bounds":[float(np.quantile(boot[:,0],.025)),float(np.quantile(boot[:,1],.975))],
            "note":"Unresolved values remain in the denominator. Bootstrap does not measure systematic judge error."}
