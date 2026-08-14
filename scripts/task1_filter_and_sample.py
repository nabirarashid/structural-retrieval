"""Task 1, step 4-5: build tier membership for the combined 118-query set
(40 original + 78 new), drop any query with an empty STRICT set against the
336-trajectory corpus, report the survivor tier-size distribution, and dump
the stratified 60-query human-review sample for the NEW queries only (the
original 40 were already human-verified in an earlier round).
"""
import json
import re
from collections import defaultdict

TASK_TYPE_NAMES = {
    1: "pick_and_place_simple", 2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep", 4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep", 6: "pick_two_obj_and_place",
}
QUERY_TYPE_TO_TASK_TYPE = {
    "placement": 1, "examination": 2, "cleaning": 3, "heating": 4, "cooling": 5, "multi_object": 6,
}
TASK_TYPE_OVERRIDES = {"medium_13": 5}
OBJECT_ALIAS = {
    "soap bar": {"soapbar"}, "soap bars": {"soapbar"},
    "bottle": {"glassbottle", "soapbottle", "spraybottle"}, "bottles": {"glassbottle", "soapbottle", "spraybottle"},
    "mug": {"mug"}, "keychain": {"keychain"}, "keychains": {"keychain"},
    "tomato": {"tomato"}, "potato": {"potato"}, "phone": {"cellphone"},
    "cup": {"cup"}, "remote control": {"remotecontrol"}, "remote controls": {"remotecontrol"},
    "pen": {"pen"}, "pens": {"pen"}, "alarm clock": {"alarmclock"},
    "cellphone": {"cellphone"}, "book": {"book"}, "egg": {"egg"},
    "lettuce": {"lettuce"}, "apple": {"apple"},
}
TARGET_OVERRIDES = {"hard_9": "cup"}
RECEPTACLE_OVERRIDES = {"hard_9": "desk"}
ART = r"(?:(?:an|a|some)\b\s*)?"
GOAL_PATTERNS = [
    re.compile(r"^(?:put|find|place) two (.+?) and put them in (?:the )?(.+?)\.?$", re.I),
    re.compile(r"^put two (.+?) in (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:examine|look at) {ART}(.+?) (?:with|under) the (?:desk ?lamp|desklamp).*$", re.I),
    re.compile(rf"^(?:clean|wash) {ART}(.+?) and (?:throw it away)\.?$", re.I),
    re.compile(rf"^(?:clean|wash) {ART}(.+?) and (?:put|place) it (?:in|on) (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:heat|warm|microwave) {ART}(.+?) and (?:put|place|throw) it (?:in|on) (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:cool|chill) {ART}(.+?) and (?:put|place) it (?:in|on) (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:put|place|move) {ART}(.+?) (?:in|on|to) (?:the )?(.+?)\.?$", re.I),
]


def extract_target(text):
    for pat in GOAL_PATTERNS:
        m = pat.match(text)
        if m:
            return m.group(1).strip()
    return None


def extract_target_and_receptacle(text):
    for pat in GOAL_PATTERNS:
        m = pat.match(text)
        if m:
            groups = m.groups()
            obj = groups[0].strip()
            recep = groups[1].strip() if len(groups) > 1 else None
            return obj, recep
    return None, None


def main():
    qd = json.load(open("/tmp/proced_mem_bench_check/procedural_memory_benchmark/benchmark/data/query_bank.json"))
    existing_queries = qd["queries"]
    new_data = json.load(open("results/task1_new_query_labels.json"))
    new_labeled = new_data["new_labeled_queries"]
    new_alias_additions = new_data["new_alias_additions"]

    full_alias = {**OBJECT_ALIAS, **{k: {v} for k, v in new_alias_additions.items()}}

    query_labels = {}
    for q in existing_queries:
        qid = q["query_id"]
        task_type = TASK_TYPE_OVERRIDES.get(qid, QUERY_TYPE_TO_TASK_TYPE[q["query_type"]])
        auto_target, auto_recep = extract_target_and_receptacle(q["query_text"])
        raw_target = TARGET_OVERRIDES.get(qid, auto_target)
        goal_receptacle = RECEPTACLE_OVERRIDES.get(qid, auto_recep)
        obj_set = full_alias.get(raw_target)
        if obj_set is None:
            raise ValueError(f"{qid}: no alias for {raw_target!r}")
        query_labels[qid] = {"task_type": task_type, "target_objects": obj_set, "tier": q["tier"],
                              "text": q["query_text"], "provenance": "original", "goal_receptacle": goal_receptacle}

    for r in new_labeled:
        qid = r["query_id"]
        obj = r["raw_target_object"]
        obj_set = full_alias.get(obj, {obj})
        query_labels[qid] = {"task_type": r["task_type"], "target_objects": obj_set, "tier": None,
                              "text": r["query_text"], "provenance": "new_alfworld_valid_unseen",
                              "goal_receptacle": r["goal_receptacle"]}

    traj_labels_list = json.load(open("results/agentinstruct_task_type_labels.json"))
    traj_labels = {r["trajectory_id"]: r for r in traj_labels_list}
    traj_ids = list(traj_labels.keys())

    tiers = {}
    for qid, qlabel in query_labels.items():
        buckets = {"STRICT": set(), "SIBLING": set(), "NEAR_MISS": set(), "OTHER": set()}
        for tid in traj_ids:
            tlabel = traj_labels[tid]
            same_type = qlabel["task_type"] == tlabel["task_type"]
            same_obj = tlabel["target_object"] in qlabel["target_objects"]
            if same_type and not same_obj:
                buckets["STRICT"].add(tid)
            elif same_type and same_obj:
                buckets["SIBLING"].add(tid)
            elif not same_type and same_obj:
                buckets["NEAR_MISS"].add(tid)
            else:
                buckets["OTHER"].add(tid)
        buckets["LENIENT"] = buckets["STRICT"] | buckets["SIBLING"]
        tiers[qid] = buckets

    empty_strict = [qid for qid, b in tiers.items() if len(b["STRICT"]) == 0]
    survivors = [qid for qid in query_labels if qid not in empty_strict]

    print(f"=== FILTER RESULTS ===")
    print(f"total combined queries: {len(query_labels)} (40 original + {len(new_labeled)} new)")
    print(f"dropped (empty STRICT set): {len(empty_strict)}")
    for qid in empty_strict:
        print(f"  dropped: {qid} -- {query_labels[qid]['text']!r} (task_type={query_labels[qid]['task_type']})")
    print(f"survivors: {len(survivors)}")

    print(f"\n=== TIER SIZE DISTRIBUTION (survivors, n={len(survivors)}) ===")
    for tier_name in ["STRICT", "SIBLING", "NEAR_MISS", "LENIENT"]:
        sizes = [len(tiers[qid][tier_name]) for qid in survivors]
        print(f"  {tier_name:10s} mean={sum(sizes)/len(sizes):.1f}  min={min(sizes)}  max={max(sizes)}  "
              f"zero_count={sum(1 for s in sizes if s == 0)}/{len(survivors)}")

    orig_survivors = [qid for qid in survivors if query_labels[qid]["provenance"] == "original"]
    new_survivors = [qid for qid in survivors if query_labels[qid]["provenance"] != "original"]
    print(f"\noriginal survivors: {len(orig_survivors)}/40")
    print(f"new survivors: {len(new_survivors)}/{len(new_labeled)}")

    json.dump(
        {"query_labels": {qid: {**v, "target_objects": sorted(v["target_objects"])} for qid, v in query_labels.items()},
         "tiers": {qid: {t: sorted(s) for t, s in b.items()} for qid, b in tiers.items()},
         "dropped_empty_strict": empty_strict, "survivors": survivors},
        open("results/task1_expanded_tier_labels.json", "w"), indent=2,
    )
    print("\nSaved: results/task1_expanded_tier_labels.json")

    # --- stratified 60-sample of NEW queries only, for human review ---
    by_type = defaultdict(list)
    for r in new_labeled:
        if r["query_id"] in survivors:
            by_type[r["task_type"]].append(r)
    sample = []
    for t in range(1, 7):
        pool = by_type.get(t, [])
        sample.extend(pool[:10])
    sample_out = [
        {"query_id": r["query_id"], "query_text": r["query_text"],
         "assigned_task_type": r["task_type"], "assigned_task_type_name": r["task_type_name"],
         "target_object": r["raw_target_object"], "goal_receptacle": r["goal_receptacle"], "rule": r["rule"]}
        for r in sample
    ]
    json.dump(sample_out, open("results/task1_new_query_review_sample.json", "w"), indent=2)
    print(f"Saved stratified review sample: {len(sample_out)} new queries "
          f"({[f'{TASK_TYPE_NAMES[t]}:{len(by_type.get(t, [])[:10])}' for t in range(1,7)]})")


if __name__ == "__main__":
    main()
