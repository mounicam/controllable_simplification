# -*- coding: utf-8 -*-


# Extract candidates using DisSim

class Node:
    def __init__(self, sent, level, string, rules):
        self.cands = []
        self.children = []
        self.level = level
        self.string = string
        self.sentence = sent
        self.rules = rules


def next_tree(lines, index):
    sentences = []
    while index < len(lines) and not lines[index].startswith("#####"):
        line = lines[index]
        sentences.append(line)
        index += 1
    return index + 1, sentences


def build_tree(lines):
    queues = [[] for i in range(30)]
    for line in lines[1:]:
        tokens = line.split("|||")
        tree_info, sentence = tokens
        sentence_tokens = sentence.strip().split("\t")
        sentence = sentence_tokens[-1].strip()
        tree_info = tree_info.replace("└", "|")
        tree_info = tree_info.replace("├", "|")

        tree_string = tree_info
        parent_string = "|".join([t for t in tree_info.split("|")[:-1] if len(t) > 0])

        rule = "None"
        if len(sentence_tokens[:-1]) > 0:
            rule = sentence_tokens[-2]

        rules = []
        level = 0
        for queue in queues:
            if len(queue) > 0 and len(parent_string) == len(queue[-1].string):
                level = queue[-1].level + 1
                rules = queue[-1].rules[:]
                break
        rules.append(rule)

        new_node = Node(sentence, level, tree_string, rules)
        if level == 0:
            queues[level].append(new_node)
        else:
            queues[level].append(new_node)
            queues[level - 1][-1].children.append(new_node)

    return queues[0][0]


def get_all_candidates(node):
    if len(node.children) == 0:
        node.cands = [((node.sentence,), tuple(set(node.rules[:-1])))]
        return

    children = node.children

    for child in children:
        get_all_candidates(child)

    cands = children[0].cands[:]
    new_cands = set()
    for i in range(1, len(children)):
        for cand in cands:
            new_cands.add(cand)
            for child_cand in children[i].cands:
                new_cands.add(child_cand)
                cand_sent = cand[0] + child_cand[0]
                cand_rules = tuple(set(cand[1]).union(set(child_cand[1])))
                add_cand = cand_sent, cand_rules
                new_cands.add(add_cand)

    new_cands.add(((node.sentence,), tuple(set(node.rules[:-1]))))
    node.cands = list(new_cands)
    return


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def get_max_sim(cand):
    max_sim = 0
    sents = [c.strip().lower() for c in cand]
    for i in range(0, len(sents)):
        for j in range(0, len(sents)):
            if i != j:
                sim = jaccard_similarity(sents[i].split(), sents[j].split())
                if sim > max_sim:
                    max_sim = sim
    return max_sim


def filter_candidates(src, candidates):
    new_candidates = set()
    for cand in candidates:
        cand_sents, cand_rules = cand
        cand_sents = list(cand_sents)
        cand_text = " <SEP> ".join(list(cand_sents))

        long = all([len(sent.split()) > 5 for sent in cand_sents])
        ratio = len(cand_text.split()) * 1.0 / len(src.split())

        if len(cand_sents) < 3 and get_max_sim(cand_sents) < 0.5 and long and 0.5 < ratio < 1.5:
            new_candidates.add(cand)
    return new_candidates


def generate_candidates(input_file, tree_file):
    tree_lines = open(tree_file).readlines()

    index = 0
    all_candidates = []
    for src in open(input_file):
        src = src.strip()
        index, tree_sents = next_tree(tree_lines, index)

        tree = build_tree(tree_sents)
        get_all_candidates(tree)
        candidates = filter_candidates(src, tree.cands)

        new_cands = []
        for cand in candidates:
            cand_sent, cand_rules = cand
            cand_sent = " <SEP> ".join(list(cand_sent))
            cand_sent = cand_sent.replace("``", '"').replace("''", '"').replace("-LRB-", '(').replace("-RRB-", ")")
            cand_rules = "DisSim  " + " ".join(cand_rules)
            new_cands.append(cand_sent + "|||" + cand_rules.strip())
        all_candidates.append(new_cands)
    return all_candidates
