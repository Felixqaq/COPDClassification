"""
把「量化 (12 維特徵 + FCNN)」的 per-patient 預測併入 RQ2b / RQ3 的三模型 hard majority vote,
重新產生 per_patient_vote_<rq>.json / .md。

(RQ1 由 build_per_patient_vote_rq1.py 處理,md 版面不同,故分開。)

輸入
- per_patient_vote_<rq>.json                       : 既有的兩個深度模型 (image / fusion) per-patient 預測
                                                     (原始 figures/ 產出不在本機,故以此檔為來源)
- models/rq_quant/param_aeropath__<rq>_per_patient.json
                                                   : 由 export_quant_per_patient.py 產生的第三個模型 (quant)

三個模型齊全後,hard majority (3 票取多數) 對每位病人都必定有結論,
不再有 `pending`,vote 列因此可與三個單模型列在同一世代上直接比較。

用法:
    python build_per_patient_vote.py --rq rq2b
    python build_per_patient_vote.py --rq rq3
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
MODEL_ORDER = ["image", "fusion", "quant"]

RQS = {
    "rq2b": dict(
        quant_json="models/rq_quant/param_aeropath__rq2b_per_patient.json",
        result_md_row="RQ2b 極端二分類的「量化(12 特徵+FCNN)」那一列 (Acc 0.755±0.087)",
        # vote json 的類別字串 <- quant 的類別字串
        label_map={"Low(<=131)": "Abnormal/emphysema-like (AC <=131 deg)",
                   "High(>=152)": "Normal-like (AC >=152 deg)"},
        short={"Abnormal/emphysema-like (AC <=131 deg)": "Abnormal",
               "Normal-like (AC >=152 deg)": "Normal-like"},
    ),
    "rq3": dict(
        quant_json="models/rq_quant/param_aeropath__rq3_per_patient.json",
        result_md_row="RQ3 OI 氣腫的「量化(12 特徵+FCNN)」那一列 (Acc 0.804±0.054)",
        label_map={"Emphysema(oi>=3)": "Significant emphysema (OI >= 3)",
                   "Normal(oi<3)": "No significant emphysema (OI < 3)"},
        short={"Significant emphysema (OI >= 3)": "Significant emphysema",
               "No significant emphysema (OI < 3)": "No significant emphysema"},
    ),
}


def binary_metrics(pairs, pos, neg):
    """pairs: [(true_label, pred_label)] -> acc / sens / spec / 混淆矩陣"""
    tp = sum(1 for t, p in pairs if t == pos and p == pos)
    fn = sum(1 for t, p in pairs if t == pos and p == neg)
    tn = sum(1 for t, p in pairs if t == neg and p == neg)
    fp = sum(1 for t, p in pairs if t == neg and p == pos)
    n = len(pairs)
    return dict(
        n=n,
        accuracy=round((tp + tn) / n, 4) if n else None,
        sensitivity=round(tp / (tp + fn), 4) if (tp + fn) else None,
        specificity=round(tn / (tn + fp), 4) if (tn + fp) else None,
        confusion=dict(tp=tp, fn=fn, tn=tn, fp=fp),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", default="rq2b", choices=sorted(RQS))
    args = ap.parse_args()
    cfg = RQS[args.rq]

    vote_json = ROOT / f"per_patient_vote_{args.rq}.json"
    vote_md = ROOT / f"per_patient_vote_{args.rq}.md"
    vote = json.loads(vote_json.read_text(encoding="utf-8"))
    quant = json.loads((ROOT / cfg["quant_json"]).read_text(encoding="utf-8"))
    qmap = {p["patient_id"]: p for p in quant["patients"]}

    pos = vote["meta"]["positive_class"]
    neg = next(c for c in vote["meta"]["class_names"] if c != pos)

    # 先前 pending 的病人 = image 與 fusion 意見相左者(用預測本身推導,重跑本腳本仍成立)
    old_pending = {p["patient_id"] for p in vote["patients"]
                   if p["models"]["image"]["pred_label"] != p["models"]["fusion"]["pred_label"]}

    # ---- 併入 quant,重新投票 ----
    for rec in vote["patients"]:
        pid = rec["patient_id"]
        q = qmap.get(pid)
        if q is None:
            raise SystemExit(f"quant 缺少病人 {pid}")
        q_true = cfg["label_map"][q["true_label"]]
        if q_true != rec["true_label"]:
            raise SystemExit(f"{pid} 標籤不一致: quant={q_true} vs vote={rec['true_label']}")
        rec["quant_fold"] = q["quant_fold"]
        rec["models"]["quant"] = dict(pred_label=cfg["label_map"][q["pred_label"]],
                                      prob_positive=q["prob_positive"],
                                      correct=q["correct"])
        rec["models"] = {m: rec["models"][m] for m in MODEL_ORDER}

        votes = {}
        for m in MODEL_ORDER:
            votes[rec["models"][m]["pred_label"]] = votes.get(rec["models"][m]["pred_label"], 0) + 1
        rec["missing_models"] = []
        rec["votes"] = votes
        rec["vote_label"] = max(votes, key=votes.get)
        rec["vote_status"] = "decided"
        rec["vote_margin"] = f"{max(votes.values())}-{3 - max(votes.values())}"
        rec["vote_correct"] = rec["vote_label"] == rec["true_label"]

    if len(qmap) != len(vote["patients"]):
        raise SystemExit(f"世代大小不符: quant={len(qmap)} vs vote={len(vote['patients'])}")

    # ---- 重算 summary ----
    summary = {}
    for m in MODEL_ORDER:
        summary[m] = binary_metrics([(r["true_label"], r["models"][m]["pred_label"])
                                     for r in vote["patients"]], pos, neg)
    summary["vote"] = binary_metrics([(r["true_label"], r["vote_label"])
                                      for r in vote["patients"]], pos, neg)
    summary["vote"]["status_counts"] = {"decided": len(vote["patients"]), "pending": 0}
    summary["vote"]["margin_counts"] = {
        k: sum(1 for r in vote["patients"] if r["vote_margin"] == k) for k in ("3-0", "2-1")}

    # ---- meta ----
    vote["meta"]["generated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    vote["meta"]["n_models"] = 3
    vote["meta"]["model_names"] = MODEL_ORDER
    vote["meta"]["model_sources"]["quant"] = (
        f"{cfg['quant_json']} "
        f"(export_quant_per_patient.py, {quant['param_subdir']}, "
        f"seed={quant['protocol']['seed']}, {quant['protocol']['n_folds']}-fold, "
        f"{quant['protocol']['epochs']} epochs, early stopping off)")
    vote["meta"]["caveat"] = (
        "image / fusion 的 per-patient 預測取自每折的 best epoch,而該 epoch 是在 test fold 上選的"
        "(trainer.py 只在 fold 分數變好時呼叫 save_predictions),故此兩列偏樂觀。"
        "quant 則是固定 100 epoch 的最後一個 epoch、未用 test fold 選點,無此偏差。"
        "vote 列繼承了 image / fusion 的樂觀性。")
    vote["meta"]["fold_note"] = (
        "`fold` 為深度模型 (image/fusion) 的折號;`quant_fold` 為量化模型自己的折號。"
        "兩者皆 seed=42 5-fold stratified,但世代排序不同故分組不同,不影響 per-patient 投票。")
    vote["summary"] = summary

    vote_json.write_text(json.dumps(vote, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已寫入 {vote_json}")

    write_md(vote, quant, old_pending, cfg, vote_md, pos, neg)


def write_md(vote, quant, old_pending, cfg, vote_md, pos, neg):
    meta, summary, pats = vote["meta"], vote["summary"], vote["patients"]
    S = cfg["short"]
    q_acc = sum(p["correct"] for p in quant["patients"]) / len(quant["patients"])

    L = []
    L.append("# Per-patient predictions and hard-vote result\n")
    L.append(f"- Generated: {meta['generated_at']}")
    L.append(f"- Cohort: {meta['n_patients']} patients")
    L.append(f"- Vote: hard majority over 3 models ({', '.join(MODEL_ORDER)}) — "
             "every patient now has a decision; no more `pending`\n")
    L.append("> **Caveat.** The `image` / `fusion` per-patient predictions were saved at each fold's "
             "best epoch, and that epoch was selected on the test fold itself (`trainer.py` only calls "
             "`save_predictions` when the fold score improves), so those two rows are an optimistic "
             "upper bound. `quant` is the last epoch of a fixed 100-epoch budget with no test-fold "
             f"selection, so it carries no such bias — its pooled out-of-fold accuracy {q_acc:.3f} "
             "matches the 5-fold mean in `result.md`. The `vote` row contains `image` / `fusion` and "
             "therefore inherits their optimism.\n")

    L.append("## Classes\n")
    L.append(f"- Positive: `{pos}` (shown as **{S[pos]}**)")
    L.append(f"- Negative: `{neg}` (shown as **{S[neg]}**)\n")

    L.append("## Sources\n")
    for m in MODEL_ORDER:
        L.append(f"- `{m}`: `{meta['model_sources'][m]}`")
    L.append("")
    L.append(f"`quant` = 與 result.md 中 {cfg['result_md_row']} 完全相同的設定"
             "(`param_aeropath` 特徵、seed=42、5-fold stratified、100 epochs、early stopping 關閉、"
             "class-weighted CE、per-fold StandardScaler、argmax)。\n")
    L.append("> Note: `fold` is the deep models' fold index; `quant fold` is the quantitative model's "
             "own fold index. Both are seed=42 5-fold stratified, but the cohort ordering differs so "
             "the groupings differ; per-patient voting does not depend on the fold index.\n")

    L.append("## Summary\n")
    L.append(f"| model | n | Accuracy | Sensitivity ({S[pos]}) | Specificity ({S[neg]}) | TP | FN | TN | FP |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in MODEL_ORDER + ["vote"]:
        s = summary[m]
        c = s["confusion"]
        name = f"**{m}**" if m == "vote" else m
        L.append(f"| {name} | {s['n']} | {s['accuracy']:.4f} | {s['sensitivity']:.4f} | "
                 f"{s['specificity']:.4f} | {c['tp']} | {c['fn']} | {c['tn']} | {c['fp']} |")
    L.append("")

    mc = summary["vote"]["margin_counts"]
    n_all = summary["vote"]["n"]
    unan_ok = sum(1 for r in pats if r["vote_margin"] == "3-0" and r["vote_correct"])
    split_ok = sum(1 for r in pats if r["vote_margin"] == "2-1" and r["vote_correct"])
    L.append(f"With the third model in place the vote covers all {n_all} patients, on the same cohort "
             "as the three single-model rows, so the four rows are directly comparable.")
    L.append(f"Vote structure: unanimous 3-0 on {mc['3-0']} patients ({unan_ok} correct, "
             f"{unan_ok / mc['3-0']:.1%}); split 2-1 on {mc['2-1']} patients ({split_ok} correct, "
             f"{split_ok / mc['2-1']:.1%}).\n")

    old = [r for r in pats if r["patient_id"] in old_pending]
    if old:
        n_ok = sum(1 for r in old if r["vote_correct"])
        L.append(f"### Previously pending ({len(old)}) — now broken by `quant`\n")
        L.append("These are exactly the cases where `image` and `fusion` contradicted each other. "
                 f"The `quant` vote lands **{n_ok}/{len(old)}** of them on the correct side "
                 f"({n_ok / len(old):.1%}).\n")
        L.append("| patient_id | true | GOLD | image | fusion | quant | vote | correct |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in old:
            cells = [S[r["models"][m]["pred_label"]] for m in MODEL_ORDER]
            L.append(f"| {r['patient_id']} | {S[r['true_label']]} | {r['gold_stage_label']} | "
                     + " | ".join(cells)
                     + f" | {S[r['vote_label']]} | {'yes' if r['vote_correct'] else 'no'} |")
        L.append("")

    wrong = [r for r in pats if not r["vote_correct"]]
    L.append(f"### Vote errors ({len(wrong)})\n")
    L.append("| patient_id | true | GOLD | image | fusion | quant | vote | margin |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in wrong:
        cells = [S[r["models"][m]["pred_label"]] for m in MODEL_ORDER]
        L.append(f"| {r['patient_id']} | {S[r['true_label']]} | {r['gold_stage_label']} | "
                 + " | ".join(cells)
                 + f" | {S[r['vote_label']]} | {r['vote_margin']} |")
    L.append("")

    L.append("## Per-patient table\n")
    head = ["patient_id", "fold", "quant fold", "true", "GOLD"]
    for m in MODEL_ORDER:
        head += [f"{m} pred", f"{m} p(pos)"]
    head += ["vote", "margin", "vote correct"]
    L.append("| " + " | ".join(head) + " |")
    L.append("| " + " | ".join("---" for _ in head) + " |")
    for r in pats:
        row = [r["patient_id"], str(r["fold"]), str(r["quant_fold"]),
               S[r["true_label"]], r["gold_stage_label"]]
        for m in MODEL_ORDER:
            mm = r["models"][m]
            row += [S[mm["pred_label"]] + ("" if mm["correct"] else " ✗"),
                    f"{mm['prob_positive']:.3f}"]
        row += [S[r["vote_label"]], r["vote_margin"], "yes" if r["vote_correct"] else "no"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    vote_md.write_text("\n".join(L), encoding="utf-8")
    print(f"已寫入 {vote_md}")


if __name__ == "__main__":
    main()
