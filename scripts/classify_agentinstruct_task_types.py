"""Task-type classifier for the 336 AgentInstruct ALFWorld trajectories,
built from the actual data rather than blind rules -- inspection of all 180
unique task_description strings showed the corpus is machine-templated into
exactly 6 non-overlapping surface patterns (2 phrasings each for 5 of the 6
types), so a careful regex classifier should hit near-100% unambiguous
classification, unlike the released code's fragile keyword-substring
fallback (which needed hardcoded exceptions like 'gold bin'/'black bin' --
that fragility is a property of the more freely-worded 40-query bank, not
of this templated trajectory corpus).

Every trajectory is checked against ALL 6 rules (not just the first match)
so a genuine collision -- more than one rule firing -- is caught and
reported as ambiguous, not silently resolved by rule order.

Does NOT touch the query bank's task_type/source fields at all, per
instruction -- queries are labeled via query_type remapping instead,
handled separately.
"""
import json
import re
from collections import Counter, defaultdict

TASK_TYPE_NAMES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# Each rule: (task_type, rule_name, compiled regex). Checked against ALL
# rules per trajectory -- a real collision gets flagged, not silently
# resolved by "first match wins".
RULES = [
    (6, "find_two_X_and_put", re.compile(r"^find two .+ and put them in", re.I)),
    (6, "put_two_X_in", re.compile(r"^put two .+ in", re.I)),
    (2, "examine_X_with_desklamp", re.compile(r"^examine the .+ with the desklamp", re.I)),
    (2, "look_at_X_under_desklamp", re.compile(r"^look at .+ under the desklamp", re.I)),
    (3, "clean_some_X_and_put", re.compile(r"^clean some .+ and put it in", re.I)),
    (3, "put_a_clean_X_in", re.compile(r"^put a clean .+ in", re.I)),
    (4, "heat_some_X_and_put", re.compile(r"^heat some .+ and put it in", re.I)),
    (4, "put_a_hot_X_in", re.compile(r"^put a hot .+ in", re.I)),
    (5, "cool_some_X_and_put", re.compile(r"^cool some .+ and put it in", re.I)),
    (5, "put_a_cool_X_in", re.compile(r"^put a cool .+ in", re.I)),
    # Type 1 fallback checked last, and only fires if nothing else did --
    # simple placement has no distinguishing keyword, it's the absence of
    # the other 5 signals.
    (1, "put_a_or_some_X_plain", re.compile(r"^put (a|some) .+ (in|on)", re.I)),
]


def classify(task_description: str) -> dict:
    hits = []
    for task_type, rule_name, pattern in RULES:
        if pattern.search(task_description):
            hits.append((task_type, rule_name))
    # type-1 fallback should only count if it's the ONLY hit (it's a
    # deliberately weak/generic pattern, not a real signal on its own)
    non_fallback = [h for h in hits if h[1] != "put_a_or_some_X_plain"]
    if non_fallback:
        hits = non_fallback
    types_hit = sorted(set(t for t, _ in hits))
    if len(types_hit) == 0:
        return {"task_type": None, "rule": "NO_RULE_MATCHED", "ambiguous": True, "all_hits": hits}
    if len(types_hit) > 1:
        return {"task_type": None, "rule": "MULTIPLE_TYPES_MATCHED", "ambiguous": True, "all_hits": hits}
    return {"task_type": types_hit[0], "rule": hits[0][1], "ambiguous": False, "all_hits": hits}


# --- target object + goal receptacle, parsed directly from task_description.
# Reliable here specifically because classify() already established the
# corpus is fully templated with zero ambiguous cases -- these patterns are
# a superset of the same 6 templates, just capturing the noun phrases
# instead of only matching. This is the SURFACE-FORM pair (what varies
# between same-task-type trajectories) as opposed to task_type (the
# STRUCTURE) -- the actual thing needed for gold/near-miss definitions.
GOAL_PATTERNS = [
    re.compile(r"^find two (.+?) and put them in (.+?)\.?$", re.I),
    re.compile(r"^put two (.+?) in (.+?)\.?$", re.I),
    re.compile(r"^examine the (.+?) with the desklamp\.?$", re.I),
    re.compile(r"^look at (.+?) under the desklamp\.?$", re.I),
    re.compile(r"^clean some (.+?) and put it in (.+?)\.?$", re.I),
    re.compile(r"^put a clean (.+?) in (.+?)\.?$", re.I),
    re.compile(r"^heat some (.+?) and put it in (.+?)\.?$", re.I),
    re.compile(r"^put a hot (.+?) in (.+?)\.?$", re.I),
    re.compile(r"^cool some (.+?) and put it in (.+?)\.?$", re.I),
    re.compile(r"^put a cool (.+?) in (.+?)\.?$", re.I),
    re.compile(r"^put (?:a|some) (.+?) (?:in|on) (.+?)\.?$", re.I),
]


def extract_target_and_goal(task_description: str) -> dict:
    for pattern in GOAL_PATTERNS:
        m = pattern.match(task_description)
        if m:
            groups = m.groups()
            target = groups[0]
            goal = groups[1] if len(groups) > 1 else None  # examine/look have no receptacle
            return {"target_object": target, "goal_receptacle": goal}
    return {"target_object": None, "goal_receptacle": None}


# --- object vocabulary extraction, from the ACTUAL action sequence (ground
# truth), not regex over the natural-language task description -- more
# reliable since it's what the agent actually manipulated, not just what
# the task asked for.
ACTION_PATTERNS = [
    ("take", re.compile(r"^take (.+?) \d+ from (.+?) \d+$", re.I)),
    ("put", re.compile(r"^put (.+?) \d+ (?:in|on) (.+?) \d+$", re.I)),
    ("clean", re.compile(r"^clean (.+?) \d+ with (.+?) \d+$", re.I)),
    ("heat", re.compile(r"^heat (.+?) \d+ with (.+?) \d+$", re.I)),
    ("cool", re.compile(r"^cool (.+?) \d+ with (.+?) \d+$", re.I)),
    ("use", re.compile(r"^use (.+?) \d+$", re.I)),
    ("go_to", re.compile(r"^go to (.+?) \d+$", re.I)),
    ("open", re.compile(r"^open (.+?) \d+$", re.I)),
    ("close", re.compile(r"^close (.+?) \d+$", re.I)),
    ("toggle", re.compile(r"^toggle (.+?) \d+$", re.I)),
]

RECEPTACLE_ACTIONS = {"go_to", "open", "close"}  # second/only arg is a receptacle


def extract_vocab(state_action_pairs: list) -> dict:
    objects, receptacles = set(), set()
    for pair in state_action_pairs:
        action = pair["action"]
        for verb, pattern in ACTION_PATTERNS:
            m = pattern.match(action)
            if not m:
                continue
            groups = m.groups()
            if verb in ("take", "put", "clean", "heat", "cool"):
                objects.add(groups[0])
                receptacles.add(groups[1])
            elif verb in RECEPTACLE_ACTIONS:
                receptacles.add(groups[0])
            elif verb == "use":
                receptacles.add(groups[0])  # "use desklamp" -- desklamp is fixture-like
            break
    return {"objects": sorted(objects), "receptacles": sorted(receptacles)}


def main():
    d = json.load(open("/tmp/proced_mem_bench_check/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajectories = d["trajectories"]

    results = []
    for t in trajectories:
        c = classify(t["task_description"])
        goal = extract_target_and_goal(t["task_description"])
        vocab = extract_vocab(t["state_action_pairs"])
        results.append({
            "trajectory_id": t["task_instance_id"],
            "task_description": t["task_description"],
            "task_type": c["task_type"],
            "task_type_name": TASK_TYPE_NAMES.get(c["task_type"]),
            "rule": c["rule"],
            "ambiguous": c["ambiguous"],
            "all_hits": [f"{ty}:{rn}" for ty, rn in c["all_hits"]],
            "target_object": goal["target_object"],
            "goal_receptacle": goal["goal_receptacle"],
            "objects_touched_in_trajectory": vocab["objects"],
            "receptacles_touched_in_trajectory": vocab["receptacles"],
        })

    dist = Counter(r["task_type"] for r in results)
    ambiguous = [r for r in results if r["ambiguous"]]

    print("=== LABEL DISTRIBUTION (336 trajectories) ===")
    for t in range(1, 7):
        print(f"  type {t} ({TASK_TYPE_NAMES[t]}): {dist.get(t, 0)}")
    print(f"  unclassified/ambiguous: {len(ambiguous)}")

    print(f"\n=== AMBIGUOUS CASES: {len(ambiguous)}/336 ===")
    for r in ambiguous:
        print(f"  {r['trajectory_id']}: {r['task_description']!r} -- hits: {r['all_hits']}")

    json.dump(results, open("results/agentinstruct_task_type_labels.json", "w"), indent=2)
    print(f"\nSaved full labels for all {len(results)} trajectories.")

    # stratified sample: 10 per type, for human verification
    by_type = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r)
    sample = []
    for t in range(1, 7):
        pool = by_type.get(t, [])
        sample.extend(pool[:10])
    sample_out = [
        {"trajectory_id": r["trajectory_id"], "task_description": r["task_description"],
         "assigned_task_type": r["task_type"], "assigned_task_type_name": r["task_type_name"],
         "rule": r["rule"]}
        for r in sample
    ]
    json.dump(sample_out, open("results/agentinstruct_label_review_sample.json", "w"), indent=2)
    print(f"Saved stratified review sample: {len(sample_out)} trajectories ({[TASK_TYPE_NAMES[t] for t in range(1,7)]}, 10 each).")


if __name__ == "__main__":
    main()
