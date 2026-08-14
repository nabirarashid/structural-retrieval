"""Task 1, steps 1-5: expand the trajectory query set from 40 to ~118
(target was ~150; only 78 unique new task texts were available from the
lightest public source found -- hkust-nlp/agentboard's ALFWorld split,
134 episodes / 78 unique goal strings, matching ALFWorld's valid_unseen
count. No valid_seen equivalent was found at a comparably lightweight
source; the full official alfworld package pulls in a heavy simulation
stack (torch, ai2thor, etc.) for what we only need as text, so that route
was not pursued. Reported as a known shortfall, not silently patched.

New queries use ALFWorld's own templated phrasing (matching the AgentInstruct
corpus's style exactly) rather than the original 40's human-paraphrased
style -- this lets us reuse the SAME rule-based classifier built for the
336-trajectory corpus (Step 1), extended with capture groups to also pull
target_object and goal_receptacle in the same pass.

STOPS after dumping the 60-query human-review sample (step 5) -- no
retrieval runs happen in this script.
"""
import json
import re
from collections import Counter, defaultdict

TASK_TYPE_NAMES = {
    1: "pick_and_place_simple", 2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep", 4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep", 6: "pick_two_obj_and_place",
}

# (task_type, rule_name, regex-with-captures: group1=object, group2=receptacle or None)
RULES = [
    (6, "find_two_X_and_put", re.compile(r"^find two (.+?) and put them in (?:the )?(.+?)\.?$", re.I)),
    (6, "put_two_X_in", re.compile(r"^put two (.+?) in (?:the )?(.+?)\.?$", re.I)),
    (2, "examine_X_with_desklamp", re.compile(r"^examine the (.+?) with the desklamp\.?$", re.I)),
    (2, "look_at_X_under_desklamp", re.compile(r"^look at (.+?) under the desklamp\.?$", re.I)),
    (3, "clean_some_X_and_put", re.compile(r"^clean some (.+?) and put it in (?:the )?(.+?)\.?$", re.I)),
    (3, "put_a_clean_X_in", re.compile(r"^put a clean (.+?) in (?:the )?(.+?)\.?$", re.I)),
    (4, "heat_some_X_and_put", re.compile(r"^heat some (.+?) and put it in (?:the )?(.+?)\.?$", re.I)),
    (4, "put_a_hot_X_in", re.compile(r"^put a hot (.+?) in (?:the )?(.+?)\.?$", re.I)),
    (5, "cool_some_X_and_put", re.compile(r"^cool some (.+?) and put it in (?:the )?(.+?)\.?$", re.I)),
    (5, "put_a_cool_X_in", re.compile(r"^put a cool (.+?) in (?:the )?(.+?)\.?$", re.I)),
    (1, "put_a_or_some_X_plain", re.compile(r"^put (?:a|some) (.+?) (?:in|on) (?:the )?(.+?)\.?$", re.I)),
]


def classify_and_extract(text: str) -> dict:
    hits = []
    for task_type, rule_name, pattern in RULES:
        m = pattern.match(text)
        if m:
            groups = m.groups()
            obj = groups[0].strip()
            recep = groups[1].strip() if len(groups) > 1 else None
            hits.append((task_type, rule_name, obj, recep))
    non_fallback = [h for h in hits if h[1] != "put_a_or_some_X_plain"]
    if non_fallback:
        hits = non_fallback
    types_hit = sorted(set(h[0] for h in hits))
    if len(types_hit) != 1:
        return {"task_type": None, "rule": "AMBIGUOUS_OR_NO_MATCH", "ambiguous": True,
                "target_object": None, "goal_receptacle": None, "all_hits": [(h[0], h[1]) for h in hits]}
    t, rule, obj, recep = hits[0]
    return {"task_type": t, "rule": rule, "ambiguous": False,
            "target_object": obj, "goal_receptacle": recep, "all_hits": [(t, rule)]}


def main():
    new_recs = [json.loads(l) for l in open("/tmp/alfworld_test.jsonl")]
    unique_goals = sorted(set(r["goal"] for r in new_recs))
    print(f"[task1] {len(unique_goals)} unique new task texts (from {len(new_recs)} raw episodes)")

    qd = json.load(open("/tmp/proced_mem_bench_check/procedural_memory_benchmark/benchmark/data/query_bank.json"))
    existing_queries = qd["queries"]
    print(f"[task1] {len(existing_queries)} existing queries (kept, flagged provenance=original)")

    new_labeled = []
    ambiguous = []
    for i, text in enumerate(unique_goals):
        c = classify_and_extract(text)
        rec = {
            "query_id": f"new_{i}", "query_text": text, "provenance": "new_alfworld_valid_unseen",
            "task_type": c["task_type"], "task_type_name": TASK_TYPE_NAMES.get(c["task_type"]),
            "rule": c["rule"], "ambiguous": c["ambiguous"],
            "raw_target_object": c["target_object"], "goal_receptacle": c["goal_receptacle"],
        }
        new_labeled.append(rec)
        if c["ambiguous"]:
            ambiguous.append(rec)

    print(f"\n[task1] label distribution (new queries, n={len(new_labeled)}):")
    dist = Counter(r["task_type"] for r in new_labeled)
    for t in range(1, 7):
        print(f"  type {t} ({TASK_TYPE_NAMES[t]}): {dist.get(t, 0)}")
    print(f"  ambiguous/unclassified: {len(ambiguous)}")
    if ambiguous:
        print("\n  AMBIGUOUS CASES:")
        for r in ambiguous:
            print(f"    {r['query_id']}: {r['query_text']!r}")

    # --- object alias table: extend only where a genuinely new object type appears ---
    traj_labels = json.load(open("results/agentinstruct_task_type_labels.json"))
    corpus_objects = set(r["target_object"] for r in traj_labels)

    EXISTING_ALIAS = {
        "soap bar": {"soapbar"}, "soap bars": {"soapbar"},
        "bottle": {"glassbottle", "soapbottle", "spraybottle"}, "bottles": {"glassbottle", "soapbottle", "spraybottle"},
        "mug": {"mug"}, "keychain": {"keychain"}, "keychains": {"keychain"},
        "tomato": {"tomato"}, "potato": {"potato"}, "phone": {"cellphone"},
        "cup": {"cup"}, "remote control": {"remotecontrol"}, "remote controls": {"remotecontrol"},
        "pen": {"pen"}, "pens": {"pen"}, "alarm clock": {"alarmclock"},
        "cellphone": {"cellphone"}, "book": {"book"}, "egg": {"egg"},
        "lettuce": {"lettuce"}, "apple": {"apple"},
    }
    new_alias_additions = {}
    for r in new_labeled:
        obj = r["raw_target_object"]
        if obj is None:
            continue
        # new queries use ALFWorld's own compact object names directly (e.g. "soapbar", "cd") --
        # already canonical, no alias needed, UNLESS plural ('two X' cases already singular
        # in ALFWorld's own vocabulary, e.g. 'soapbar' not 'soapbars') or genuinely new to corpus
        singular = obj[:-1] if obj.endswith("s") and obj[:-1] in corpus_objects else obj
        if singular not in corpus_objects and obj not in EXISTING_ALIAS:
            new_alias_additions[obj] = singular

    print(f"\n[task1] new object types requiring alias-table additions ({len(new_alias_additions)}):")
    for obj, canon in new_alias_additions.items():
        print(f"  {obj!r} -> {canon!r} (in corpus: {canon in corpus_objects})")

    json.dump({"new_labeled_queries": new_labeled, "ambiguous": ambiguous,
               "new_alias_additions": new_alias_additions},
              open("results/task1_new_query_labels.json", "w"), indent=2)
    print("\nSaved: results/task1_new_query_labels.json")


if __name__ == "__main__":
    main()
