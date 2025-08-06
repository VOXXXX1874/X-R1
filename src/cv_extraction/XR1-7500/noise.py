import json
import random
import re

# Read the "aug_ext_exp_num_tt_012/train.json" file
with open("aug_ext_exp_num_tt_012/train.json", "r") as file:
    data = json.load(file)

# noise level
noise_level = 1.0 # 0.12/0.25/0.5/1.0
# noised dataset
noised_data = []


# go through the data and add some noise to the "process" field
for item in data:
    # read the "process" field
    process = item["process"]
    # Parse the thinking completion to extract the targets
    gold_targets = re.search(r'<target>(.*?)<trap>', process, re.DOTALL)
    gold_targets = gold_targets.group(1).strip()
    gold_targets = re.split(r'<sep>', gold_targets)
    gold_targets = [target.strip() for target in gold_targets if target.strip()]
    # Parse the thinking completion to extract the traps
    gold_traps = re.search(r'<trap>(.*?)$', process, re.DOTALL)
    gold_traps = gold_traps.group(1).strip() if gold_traps else ''
    gold_traps = re.split(r'<sep>', gold_traps)
    gold_traps = [trap.strip() for trap in gold_traps if trap.strip()]
    # add noise to the targets and traps
    noised_targets = []
    noised_traps = []
    for target in gold_targets:
        if random.random() < noise_level:
            # add noise to the target
            noised_traps.append(target)
        else:
            # keep the target as is
            noised_targets.append(target)
    for trap in gold_traps:
        if random.random() < noise_level:
            # add noise to the trap
            noised_targets.append(trap)
        else:
            # keep the trap as is
            noised_traps.append(trap)
    noised_targets = "<target> " + " <sep> ".join(noised_targets)
    noised_traps = "<trap> " + " <sep> ".join(noised_traps)
    noised_process = f"{noised_targets} {noised_traps}"
    # create a new item with the noised targets and traps
    item["process"] = noised_process
    # add the item to the noised data
    noised_data.append(item)

# Write the noised data to a new file
with open("train_noised.json", "w") as file:
    json.dump(noised_data, file, indent=4)
