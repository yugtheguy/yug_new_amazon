#!/usr/bin/env python3
"""Cheap B007_01/B007_03 post-processing: keyed blends, fine thresholds, optional competition rules."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

def repo_root() -> Path: return Path(__file__).resolve().parents[2]
def imports() -> tuple[Any,Any,Any,Any]:
    root=repo_root(); sys.path.insert(0,str(root/"experiments"/"B003")); import run_b003 as b003  # type: ignore
    sys.path.insert(0,str(root/"experiments"/"B005")); import run_b005 as b005  # type: ignore
    sys.path.insert(0,str(root/"experiments"/"GPU2_CHALLENGER")); import decision_rules  # type: ignore
    sys.path.insert(0,str(Path(__file__).resolve().parent)); import post_utils  # type: ignore
    return b003,b005,decision_rules,post_utils
B003,B005,DECISION,UTILS=imports()

class DeadlineReached(RuntimeError):
    pass

class Deadline:
    def __init__(self,minutes:float): self.started=time.monotonic(); self.limit=minutes*60
    @property
    def elapsed(self)->float:return time.monotonic()-self.started
    def check(self,stage:str)->None:
        if self.elapsed>=self.limit: raise DeadlineReached(f"hard deadline reached before {stage}")

def load(path:Path,probability_column:str)->pd.DataFrame:
    import pyarrow.parquet as pq  # type: ignore
    frame=pq.read_table(path).to_pandas()
    if probability_column not in frame.columns: raise ValueError(f"{path}: missing probability column {probability_column}")
    frame=frame.rename(columns={probability_column:"probability"})
    for column in ("query_id","candidate_id","candidate_source","country"): frame[column]=frame[column].astype(str)
    frame["label"]=frame["label"].astype(int); frame["probability"]=frame["probability"].astype(float)
    if not np.isfinite(frame["probability"]).all(): raise ValueError(f"{path}: non-finite probabilities")
    return frame

def tune(probabilities:Sequence[float],frame:pd.DataFrame,truth:Mapping[str,set[str]],apply:Any,evaluate:Any)->tuple[float,float,dict[str,Any]]:
    return B005.tune_thresholds(truth,UTILS.scores(frame["query_id"],frame["candidate_id"],probabilities),apply,evaluate)

def evaluate_probs(probabilities:Sequence[float],frame:pd.DataFrame,truth:Mapping[str,set[str]],t2:float,t3:float,apply:Any,evaluate:Any)->tuple[dict[str,Any],dict[str,set[str]]]:
    predictions=apply(UTILS.scores(frame["query_id"],frame["candidate_id"],probabilities),t2,t3)
    return dict(evaluate(truth,predictions)),predictions

def countries(truth:Mapping[str,set[str]],predictions:Mapping[str,set[str]],country_by_query:Mapping[str,str],evaluate:Any)->dict[str,Any]:
    result={}
    for country in sorted(set(country_by_query.values())):
        ids=[q for q in truth if country_by_query[q]==country]
        result[country]=dict(evaluate({q:truth[q] for q in ids},{q:predictions.get(q,set()) for q in ids}))
    return result

def result(name:str,overall:Mapping[str,Any],calibration:float,t2:float,t3:float,country:Mapping[str,Any],**extra:Any)->dict[str,Any]:
    value={"configuration":name,"macro_f05":float(overall["macro_f05"]),"precision":float(overall["global_precision"]),"recall":float(overall["global_recall"]),"delta_vs_B007_03":float(overall["macro_f05"])-UTILS.CHAMPION["macro_f05"],"calibration_macro_f05":float(calibration),"threshold_s2":t2,"threshold_s3":t3,"country":{k:{"macro_f05":float(v["macro_f05"])} for k,v in country.items()}}
    value.update(extra); return value

def write_predictions(path:Path,frame:pd.DataFrame,probabilities:Sequence[float],predictions:Mapping[str,set[str]])->None:
    import pyarrow as pa, pyarrow.parquet as pq  # type: ignore
    output=frame[UTILS.KEYS+["country","label"]].copy(); output["final_probability"]=np.asarray(probabilities,dtype=float)
    output["accepted"]=[candidate in predictions.get(query,set()) for query,candidate in zip(output["query_id"],output["candidate_id"])]
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); pq.write_table(pa.Table.from_pandas(output,preserve_index=False),tmp,compression="zstd"); os.replace(tmp,path)

def main(argv:Sequence[str]|None=None)->int:
    args=parse_args(argv); deadline=Deadline(args.deadline_minutes); root=repo_root(); os.chdir(root); args.output_dir.mkdir(parents=True,exist_ok=True)
    for path in (args.b007_01_calibration,args.b007_01_evaluation,args.b007_03_calibration,args.b007_03_evaluation,args.ground_truth,args.val_ids):
        if not path.exists(): raise FileNotFoundError(path)
    production=B003.import_production(root); apply=production["apply_threshold_and_deduplication"]; evaluate=production["evaluate_predictions"]
    deadline.check("prediction loading")
    f01c=load(args.b007_01_calibration,args.probability_column); f03c=load(args.b007_03_calibration,args.probability_column)
    f01e=load(args.b007_01_evaluation,args.probability_column); f03e=load(args.b007_03_evaluation,args.probability_column)
    f03c,f01c,cal_report=UTILS.align_frames(f03c,f01c,"B007_03 calibration","B007_01 calibration")
    f03e,f01e,eval_report=UTILS.align_frames(f03e,f01e,"B007_03 evaluation","B007_01 evaluation")
    frozen_ids,_=B003.load_selected_ids(args.val_ids,5000); eval_ids=set(f03e["query_id"]); cal_ids=set(f03c["query_id"])
    if eval_ids!=set(frozen_ids): raise RuntimeError("B007 evaluation queries do not equal frozen 5k")
    if cal_ids & eval_ids: raise RuntimeError("calibration/evaluation query overlap")
    truth=B003.load_ground_truth(args.ground_truth,sorted(cal_ids|eval_ids)); cal_truth={q:truth[q] for q in cal_ids}; eval_truth={q:truth[q] for q in eval_ids}
    for frame,name in ((f03c,"calibration"),(f03e,"evaluation")):
        mismatches=sum(int(label)!=int(candidate in truth[query]) for query,candidate,label in zip(frame["query_id"],frame["candidate_id"],frame["label"]))
        if mismatches: raise RuntimeError(f"{name}: {mismatches} label mismatches")
    alignment={"status":"PASS","calibration":cal_report,"evaluation":eval_report,"calibration_queries":len(cal_ids),"evaluation_queries":len(eval_ids),"frozen_5000_verified":True}
    B003.atomic_json(args.output_dir/"alignment_report.json",alignment)
    country_by_query={q:c for q,c in zip(f03e["query_id"],f03e["country"])}

    # Derive centers from internal calibration when current production thresholds were not supplied.
    base_scores=UTILS.scores(f03c["query_id"],f03c["candidate_id"],f03c["probability"])
    if args.old_threshold_s2 is None or args.old_threshold_s3 is None:
        old2,old3,old_cal=B005.tune_thresholds(cal_truth,base_scores,apply,evaluate); threshold_source="derived_on_internal_calibration"
    else:
        old2,old3=float(args.old_threshold_s2),float(args.old_threshold_s3); old_cal=dict(evaluate(cal_truth,apply(base_scores,old2,old3))); threshold_source="supplied_production_thresholds"

    deadline.check("E1")
    blend_trials=[]
    for ordinal,w01 in enumerate((0.0,0.05,0.10,0.15,0.20,0.30,0.40)):
        probabilities=UTILS.probability_mix(f03c["probability"],f01c["probability"],w01); t2,t3,metrics=tune(probabilities,f03c,cal_truth,apply,evaluate)
        blend_trials.append({"ordinal":ordinal,"weight_b007_03":1-w01,"weight_b007_01":w01,"threshold_s2":t2,"threshold_s3":t3,"calibration_macro_f05":float(metrics["macro_f05"])})
    blend_winner=max(blend_trials,key=lambda row:(row["calibration_macro_f05"],-row["ordinal"]))
    blend_eval_probs=UTILS.probability_mix(f03e["probability"],f01e["probability"],blend_winner["weight_b007_01"])
    blend_overall,blend_predictions=evaluate_probs(blend_eval_probs,f03e,eval_truth,blend_winner["threshold_s2"],blend_winner["threshold_s3"],apply,evaluate)
    e1=result("E1 best blend",blend_overall,blend_winner["calibration_macro_f05"],blend_winner["threshold_s2"],blend_winner["threshold_s3"],countries(eval_truth,blend_predictions,country_by_query,evaluate),selected=blend_winner,calibration_trials=blend_trials)
    B003.atomic_json(args.output_dir/"blend_search.json",e1)

    deadline.check("E2")
    new2,new3,newcal,threshold_trials=UTILS.local_threshold_search(cal_truth,base_scores,old2,old3,apply,evaluate,args.use_quarter_step)
    e2_overall,e2_predictions=evaluate_probs(f03e["probability"],f03e,eval_truth,new2,new3,apply,evaluate)
    e2=result("E2 fine thresholds",e2_overall,newcal["macro_f05"],new2,new3,countries(eval_truth,e2_predictions,country_by_query,evaluate),old_threshold_s2=old2,old_threshold_s3=old3,old_threshold_source=threshold_source,old_calibration_macro_f05=float(old_cal["macro_f05"]),calibration_trials=threshold_trials)
    B003.atomic_json(args.output_dir/"threshold_search.json",e2)

    e3={"status":"NOT_RUN"}; e3_predictions=None; e3_probs=f03e["probability"].to_numpy()
    if args.run_e3:
        deadline.check("E3")
        winner,trials=DECISION.select_rule_on_calibration(cal_truth,base_scores,new2,new3,DECISION.default_rule_family(),evaluate)
        e3_predictions=DECISION.apply_decision_rule(UTILS.scores(f03e["query_id"],f03e["candidate_id"],e3_probs),new2,new3,winner)
        overall=dict(evaluate(eval_truth,e3_predictions)); qstats,tstats=DECISION.competition_features(base_scores)
        e3=result("E3 competition",overall,next(row["metrics"]["macro_f05"] for row in trials if row["rule"]==winner),new2,new3,countries(eval_truth,e3_predictions,country_by_query,evaluate),selected_rule=winner,calibration_trials=trials,competition_counts={"query_source_groups":len(qstats),"targets":len(tstats)})
    B003.atomic_json(args.output_dir/"competition_search.json",e3)

    rows=[{"configuration":"B007_03 baseline",**UTILS.CHAMPION,"delta_vs_B007_03":0.0},{"configuration":"B007_01 baseline","macro_f05":0.9611574772786582,"precision":0.985741,"recall":0.916705,"delta_vs_B007_03":0.9611574772786582-UTILS.CHAMPION["macro_f05"]}]
    rows.extend({k:value[k] for k in ("configuration","macro_f05","precision","recall","delta_vs_B007_03")} for value in (e1,e2))
    if e3.get("configuration"): rows.append({k:e3[k] for k in ("configuration","macro_f05","precision","recall","delta_vs_B007_03")})
    measured=[(e1,blend_eval_probs,blend_predictions),(e2,f03e["probability"].to_numpy(),e2_predictions)]
    if e3_predictions is not None: measured.append((e3,e3_probs,e3_predictions))
    winner_tuple=max(measured,key=lambda item:item[0]["macro_f05"])
    if winner_tuple[0]["macro_f05"]>UTILS.CHAMPION["macro_f05"]:
        champion=winner_tuple[0]; final_probs=winner_tuple[1]; final_predictions=winner_tuple[2]; recommendation="PROMOTE measured post-processing winner"
    else:
        champion={"configuration":"B007_03 baseline","macro_f05":UTILS.CHAMPION["macro_f05"],"threshold_s2":old2,"threshold_s3":old3,"delta_vs_B007_03":0.0}; final_probs=f03e["probability"].to_numpy(); _,final_predictions=evaluate_probs(final_probs,f03e,eval_truth,old2,old3,apply,evaluate); recommendation="KEEP B007_03 BASELINE"
    config={"configuration":champion["configuration"],"blend_weight_b007_03":blend_winner["weight_b007_03"] if champion["configuration"]=="E1 best blend" else 1.0,"blend_weight_b007_01":blend_winner["weight_b007_01"] if champion["configuration"]=="E1 best blend" else 0.0,"threshold_s2":champion["threshold_s2"],"threshold_s3":champion["threshold_s3"],"competition_rule":e3.get("selected_rule") if champion["configuration"]=="E3 competition" else None}
    B003.atomic_json(args.output_dir/"best_decision_config.json",config); write_predictions(args.output_dir/"best_frozen_evaluation_predictions.parquet",f03e,final_probs,final_predictions)
    comparison={"rows":rows,"final_champion":champion,"production_config":config,"recommendation":recommendation,"runtime_seconds":deadline.elapsed,"stop":True}
    B003.atomic_json(args.output_dir/"comparison.json",comparison); print(json.dumps(comparison,indent=2),flush=True); print("GPU2_B007_POST STOP",flush=True); return 0

def parse_args(argv:Sequence[str]|None=None)->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b007-01-calibration",type=Path,required=True);p.add_argument("--b007-01-evaluation",type=Path,required=True);p.add_argument("--b007-03-calibration",type=Path,required=True);p.add_argument("--b007-03-evaluation",type=Path,required=True)
    p.add_argument("--ground-truth",type=Path,required=True);p.add_argument("--val-ids",type=Path,default=Path("experiments/val_s1_ids.txt"));p.add_argument("--probability-column",default="probability");p.add_argument("--output-dir",type=Path,default=Path("artifacts/experiments/GPU2_B007_POST"))
    p.add_argument("--old-threshold-s2",type=float);p.add_argument("--old-threshold-s3",type=float);p.add_argument("--deadline-minutes",type=float,default=40);p.add_argument("--run-e3",action="store_true");p.add_argument("--no-quarter-step",dest="use_quarter_step",action="store_false");p.set_defaults(use_quarter_step=True)
    args=p.parse_args(argv)
    if bool(args.old_threshold_s2 is None)!=bool(args.old_threshold_s3 is None):p.error("supply both old thresholds or neither")
    return args
if __name__=="__main__":
    try:
        raise SystemExit(main())
    except DeadlineReached as exc:
        parsed=parse_args(); parsed.output_dir.mkdir(parents=True,exist_ok=True)
        partial={"status":"STOPPED_DEADLINE","reason":str(exc),"recommendation":"Use only completed measured artifacts; release GPU2 for production inference.","stop":True}
        B003.atomic_json(parsed.output_dir/"comparison.json",partial)
        print(json.dumps(partial,indent=2),flush=True); print("GPU2_B007_POST STOP",flush=True); raise SystemExit(0)
