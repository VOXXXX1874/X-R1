import re

def critical_value_reward_num(thinking_completion, process):
    reward = 0.0
    final_end_pos = 0
    # Parse the thinking completion to extract the targets
    gold_targets = re.search(r'<target>(.*?)<trap>', process, re.DOTALL)
    if gold_targets is None:
        return reward, final_end_pos
    gold_targets = gold_targets.group(1).strip()
    gold_targets = re.split(r'<sep>', gold_targets)
    gold_targets = [int(target.strip()) for target in gold_targets if target.strip()]
    # Parse the thinking completion to extract the traps
    gold_traps = re.search(r'<trap>(.*?)$', process, re.DOTALL)
    gold_traps = gold_traps.group(1).strip() if gold_traps else ''
    gold_traps = re.split(r'<sep>', gold_traps)
    gold_traps = [int(trap.strip()) for trap in gold_traps if trap.strip()]

    # Parse the numbers in the thinking completion
    number_pattern = re.compile(r'\d+')
    thinking_completion_numbers = number_pattern.findall(thinking_completion)
    thinking_completion_numbers = [int(num) for num in thinking_completion_numbers]

    if len(gold_targets) != 0:
        atom_reward = 1.0 / len(gold_targets)
        for target in gold_targets:
            if target in thinking_completion_numbers:
                reward += atom_reward
                tmp = thinking_completion.index(str(target)) + len(str(target))
                if tmp > final_end_pos:
                    final_end_pos = tmp

    if len(gold_traps) != 0:
        atom_reward = -1.0 / len(gold_traps)
        for trap in gold_traps:
            if trap in thinking_completion_numbers:
                reward += atom_reward

    return reward, final_end_pos