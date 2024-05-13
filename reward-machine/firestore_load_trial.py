import os
import sys
import json
import pandas
from collections import defaultdict

# Add src/ to our path so we can import from the scripts in room_and_object_types.py
sys.path.insert(1, os.path.join(sys.path[0], '../src'))
from ast_utils import load_games_from_file

FIRESTORE_ID_TO_PARTICIPANT_ID = {"HMosyugAD4iHS8V6ajD6": "5f9aba6600cdf11f1c9b915c"}
STATS_PATH = "../data/dsl_statistics_interactive.csv"


firestore_id = "HMosyugAD4iHS8V6ajD6"
participant_id = FIRESTORE_ID_TO_PARTICIPANT_ID[firestore_id]
playtrace = json.load(open(f"../reward-machine/traces/{firestore_id}-gameplay-attempt-1.json", "rb"))

stats = pandas.read_csv(STATS_PATH)
participant_id_to_idx = {value: key for key, value in dict(stats["game_name"]).items()}
idx = participant_id_to_idx[participant_id]


all_predicates = sum([list(json.loads(stats["predicates_referenced"][i].replace("'", '"')).keys())
                      for i in range(len(stats))], [])

for predicate in sorted(set(all_predicates)):
    print(predicate)
# exit()
playtrace_predicate_stats = stats["predicates_referenced"][idx]

for i in range(len(playtrace)):
    objects = playtrace[i]["objects"]
    n_objects_changed = playtrace[i]["nObjectsChanged"]

    if len(objects) > 0:
        print("\n" + "=" * 200)
        print("STATE:", i + 1)
        object_names = list(sorted([obj["name"] for obj in objects]))

        objects_by_type = defaultdict(list)
        for obj in objects:
            objects_by_type[obj["objectType"]].append(obj["name"])
            if obj["objectType"] == "Beachball":
                print(obj)
                exit()

        shelves = [obj for obj in objects if "Shelf" in obj["name"] or True]

        # for obj_name in object_names:
        #     print(obj_name)

        # print("[" + ", ".join(object_names) + "]")

        print(objects_by_type)
        # print("Objects:", shelves)
        print("Num changed:", n_objects_changed)
        print("Agent State:", playtrace[i]["agentState"])

    elif n_objects_changed > 0:
        print("Num objects changed:", n_objects_changed)

print("\n\nPlaytrace statistics:")
for col in stats.columns:
    print(f"{col}: {stats[col][idx]}")

print("\n\nJSON Keys:")
for key in playtrace[0].keys():
    print(key)


programs = load_games_from_file("../dsl/interactive-beta.pddl")
print("\nNumber of programs: ", len(programs))
print("\nNumber that contain '(then':", sum(["(then" in program for program in programs]))

# for program in programs:
#     if "(then" not in program:
#         print("\n\n==========================================")
#         print(program)


# print([playtrace[idx]["agentState"] for idx in range(len(playtrace))])


"""
(define (game game-109) (:domain many-objects-room-v1)  ; 109
(:constraints (and
    (preference ballThrownToBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    (preference cubeBlockThrownToTopShelf (exists (?c - cube_block)
        (then
            (once (agent_holds ?c))
            (hold (and (not (agent_holds ?c)) (in_motion ?c)))
            (once (and (not (in_motion ?c)) (on top_shelf ?c)))
        )
    ))
    (preference pillowThrownToDoggieBed (exists (?p - pillow ?d - doggie_bed)
        (then
            (once (agent_holds ?p))
            (hold (and (not (agent_holds ?p)) (in_motion ?p)))
            (once (and (not (in_motion ?p)) (on ?d ?p)))
        )
    ))
))
(:scoring maximize (+
    (count-once-per-objects ballThrownToBin)
    (count-once-per-objects cubeBlockThrownToTopShelf)
    (count-once-per-objects pillowThrownToDoggieBed)
)))
"""
