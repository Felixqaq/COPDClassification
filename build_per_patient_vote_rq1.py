"""
把「量化 (12 維特徵 + FCNN)」的 per-patient 預測併入 RQ1 三模型 hard majority vote,
重新產生 per_patient_vote_rq1.json / per_patient_vote_rq1.md。

輸入
- per_patient_vote_rq1.json           : 既有的兩個深度模型 (image / fusion) per-patient 預測
                                        (原始 figures/ 產出不在本機,故以此檔為 image/fusion 的來源)
- models/rq_quant/param_aeropath__rq1_per_patient.json
                                        : 由 export_quant_per_patient_rq1.py 產生的第三個模型 (quant)

三個模型齊全後,hard majority (3 票取多數) 對每位病人都必定有結論,
不再有 `pending`,vote 列因此可與三個單模型列在同一 66 人世代上直接比較。
"""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
VOTE_JSON = ROOT / "per_patient_vote_rq1.json"
VOTE_MD = ROOT / "per_patient_vote_rq1.md"
QUANT_JSON = ROOT / "models" / "rq_quant" / "param_aeropath__rq1_per_patient.json"

MODEL_ORDER = ["image", "fusion", "quant"]
POS, NEG = "Abnormal", "Normal"


def binary_metrics(pairs):
    """pairs: [(true_label, pred_label)] -> acc / sens(Abnormal) / spec(Normal) / 混淆矩陣"""
    tp = sum(1 for t, p in pairs if t == POS and p == POS)
    fn = sum(1 for t, p in pairs if t == POS and p == NEG)
    tn = sum(1 for t, p in pairs if t == NEG and p == NEG)
    fp = sum(1 for t, p in pairs if t == NEG and p == POS)
    n = len(pairs)
    return dict(
        n=n,
        accuracy=round((tp + tn) / n, 4) if n else None,
        sensitivity_abnormal=round(tp / (tp + fn), 4) if (tp + fn) else None,
        specificity_normal=round(tn / (tn + fp), 4) if (tn + fp) else None,
        confusion=dict(tp=tp, fn=fn, tn=tn, fp=fp),
    )


def main():
    vote = json.loads(VOTE_JSON.read_text(encoding="utf-8"))
    quant = json.loads(QUANT_JSON.read_text(encoding="utf-8"))
    qmap = {p["patient_id"]: p for p in quant["patients"]}

    # 先前 pending 的病人 = image 與 fusion 意見相左者(用預測本身推導,重跑本腳本仍成立)
    old_pending = {p["patient_id"] for p in vote["patients"]
                   if p["models"]["image"]["pred_label"] != p["models"]["fusion"]["pred_label"]}

    # ---- 併入 quant,重新投票 ----
    for rec in vote["patients"]:
        pid = rec["patient_id"]
        q = qmap.get(pid)
        if q is None:
            raise SystemExit(f"quant 缺少病人 {pid}")
        if q["true_label"] != rec["true_label"]:
            raise SystemExit(f"{pid} 標籤不一致: quant={q['true_label']} vs vote={rec['true_label']}")
        rec["quant_fold"] = q["quant_fold"]
        rec["models"]["quant"] = dict(pred_label=q["pred_label"],
                                      prob_abnormal=q["prob_abnormal"],
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
                                     for r in vote["patients"]])
    summary["vote"] = binary_metrics([(r["true_label"], r["vote_label"]) for r in vote["patients"]])
    summary["vote"]["status_counts"] = {"decided": len(vote["patients"]), "pending": 0}
    summary["vote"]["margin_counts"] = {
        k: sum(1 for r in vote["patients"] if r["vote_margin"] == k) for k in ("3-0", "2-1")}

    # ---- meta ----
    vote["meta"]["generated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    vote["meta"]["n_models"] = 3
    vote["meta"]["model_names"] = MODEL_ORDER
    vote["meta"]["model_sources"]["quant"] = (
        f"models/rq_quant/param_aeropath__rq1_per_patient.json "
        f"(export_quant_per_patient_rq1.py, {quant['param_subdir']}, "
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

    VOTE_JSON.write_text(json.dumps(vote, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已寫入 {VOTE_JSON}")

    write_md(vote, quant, old_pending)


def write_md(vote, quant, old_pending):
    meta, summary, pats = vote["meta"], vote["summary"], vote["patients"]
    L = []
    L.append("# Per-patient predictions and hard-vote result\n")
    L.append(f"- Generated: {meta['generated_at']}")
    L.append(f"- Cohort: {meta['n_patients']} patients")
    L.append(f"- Vote: hard majority over 3 models ({', '.join(MODEL_ORDER)}) — "
             "三票取多數,每位病人皆有結論,不再有 pending\n")
    L.append("> **Caveat.** `image` / `fusion` 的 per-patient 預測存在每折的 best epoch,"
             "而該 epoch 是在 test fold 上挑的(`trainer.py` 只在 fold 分數改善時呼叫 `save_predictions`),"
             "因此這兩列是樂觀上界。`quant` 為固定 100 epoch 的最後一個 epoch、沒有用 test fold 選停點,"
             "無此偏差 —— 它的 pooled out-of-fold accuracy 0.848 與 `result.md` 的 5-fold 平均一致。"
             "`vote` 列因為含有 image / fusion,同樣繼承了這份樂觀性。\n")
    L.append("## Sources\n")
    for m in MODEL_ORDER:
        L.append(f"- `{m}`: `{meta['model_sources'][m]}`")
    L.append("")
    L.append("`quant` = result.md 中「量化(12 特徵+FCNN)」那一列的同一組設定"
             "(`param_aeropath` 特徵、seed=42、5-fold stratified、100 epochs、early stopping 關閉、"
             "class-weighted CE、per-fold StandardScaler、argmax)。\n")
    L.append("> 註:`fold` 欄是深度模型的折號,`quant_fold` 是量化模型自己的折號。"
             "兩者都是 seed=42 的 5-fold stratified,但世代排序不同故分組不同;"
             "per-patient 投票與折號無關,不受影響。\n")

    L.append("## Summary\n")
    L.append("| model | n | Accuracy | Sensitivity (Abnormal) | Specificity (Normal) | TP | FN | TN | FP |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in MODEL_ORDER + ["vote"]:
        s = summary[m]
        c = s["confusion"]
        name = f"**{m}**" if m == "vote" else m
        L.append(f"| {name} | {s['n']} | {s['accuracy']:.4f} | {s['sensitivity_abnormal']:.4f} | "
                 f"{s['specificity_normal']:.4f} | {c['tp']} | {c['fn']} | {c['tn']} | {c['fp']} |")
    L.append("")
    mc = summary["vote"]["margin_counts"]
    n_all = summary["vote"]["n"]
    unan_ok = sum(1 for r in pats if r["vote_margin"] == "3-0" and r["vote_correct"])
    split_ok = sum(1 for r in pats if r["vote_margin"] == "2-1" and r["vote_correct"])
    L.append(f"補上第三個模型後,vote 涵蓋全部 {n_all} 位病人,與三個單模型列在同一世代上,可直接比較。")
    L.append(f"投票結構:一致 3-0 共 {mc['3-0']} 位(正確 {unan_ok},"
             f"{unan_ok / mc['3-0']:.1%});分歧 2-1 共 {mc['2-1']} 位(正確 {split_ok},"
             f"{split_ok / mc['2-1']:.1%})—— 模型一致時幾乎必對,錯誤集中在分歧的少數個案。\n")

    # 先前 pending 的 11 位,現在由 quant 決定
    old = [r for r in pats if r["patient_id"] in old_pending]
    n_ok = sum(1 for r in old if r["vote_correct"])
    L.append(f"### 先前 pending 的 {len(old)} 位 —— 現由 `quant` 打破平手\n")
    L.append(f"這些正是 image / fusion 互相矛盾的個案。quant 的一票讓其中 **{n_ok}/{len(old)}** "
             f"投對({n_ok / len(old):.1%})。\n")
    L.append("| patient_id | true | GOLD | image | fusion | quant | vote | correct |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in old:
        cells = [r["models"][m]["pred_label"] for m in MODEL_ORDER]
        L.append(f"| {r['patient_id']} | {r['true_label']} | {r['gold_stage_label']} | "
                 + " | ".join(cells)
                 + f" | {r['vote_label']} | {'yes' if r['vote_correct'] else 'no'} |")
    L.append("")

    wrong = [r for r in pats if not r["vote_correct"]]
    L.append(f"### Vote 判錯的 {len(wrong)} 位\n")
    L.append("| patient_id | true | GOLD | image | fusion | quant | vote | margin |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in wrong:
        cells = [r["models"][m]["pred_label"] for m in MODEL_ORDER]
        L.append(f"| {r['patient_id']} | {r['true_label']} | {r['gold_stage_label']} | "
                 + " | ".join(cells)
                 + f" | {r['vote_label']} | {r['vote_margin']} |")
    L.append("")

    L.append("## Per-patient table\n")
    head = ["patient_id", "fold", "quant fold", "true", "GOLD"]
    for m in MODEL_ORDER:
        head += [f"{m} pred", f"{m} p(Abn)"]
    head += ["vote", "margin", "vote correct"]
    L.append("| " + " | ".join(head) + " |")
    L.append("| " + " | ".join("---" for _ in head) + " |")
    for r in pats:
        row = [r["patient_id"], str(r["fold"]), str(r["quant_fold"]),
               r["true_label"], r["gold_stage_label"]]
        for m in MODEL_ORDER:
            mm = r["models"][m]
            row += [mm["pred_label"] + ("" if mm["correct"] else " ✗"),
                    f"{mm['prob_abnormal']:.3f}"]
        row += [r["vote_label"], r["vote_margin"], "yes" if r["vote_correct"] else "no"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    VOTE_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"已寫入 {VOTE_MD}")


if __name__ == "__main__":
    main()
