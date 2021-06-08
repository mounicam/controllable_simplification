import math
import random
import numpy as np
from collections import Counter


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def compression_ratio(comp, simp):
    return len(simp.split()) * 1.0 / len(comp.split())


class FeatureExtractor:
    def __init__(self):
        self.rule_vocab = {}
        for ind, line in enumerate(open("ranking/all_rules.txt")):
            self.rule_vocab[line.strip()] = ind

    def get_fv(self, cand, src):
        cand_sent = cand[0]

        fv = list()
        fv.append(len(cand_sent.lower().split("<sep>")))
        fv.append(len(src.split()) * 1.0)
        fv.append(jaccard_similarity(cand_sent.lower().split(), src.lower().split()))

        ratio_src_cand = len(cand_sent.split()) * 1.0 / len(src.split())
        fv.append(ratio_src_cand)

        fv.append(len(cand_sent.split()) * 1.0 / len(cand_sent.split("<SEP>")))

        rules = cand[1].split()
        fv.append(len(rules))

        rules_vec = [0] * len(self.rule_vocab)
        for rule in rules:
            rules_vec[self.rule_vocab[rule]] = 1
        fv.extend(rules_vec)
        return fv

    def filter_candidates(self, tuples):
        cands, feats = [], []
        for tup in tuples:
            fv, cand_sent = tup
            feats.append(fv)
            cands.append(cand_sent)
        return cands, feats

    def get_features(self, input_file, cands_file):

        print("Extracting features")
        all_src, all_features, all_cands = [], [], []
        for src, candidates in zip(open(input_file), open(cands_file)):

            src = src.strip()
            cands = candidates.strip().split("\t")
            cands = [tuple(cand.split("|||")[:2]) for cand in cands]

            tuples = []
            cand_sents = set()
            for cand in cands:
                if len(cand) > 1 and cand[0] not in cand_sents:
                    fv = self.get_fv(cand, src)
                    tuples.append((fv, cand[0]))
            cands, feature_vectors = self.filter_candidates(tuples)

            assert len(cands) == len(feature_vectors)
            all_src.append(src)
            all_cands.append(cands)
            all_features.append(feature_vectors)

        print("Data size: ", len(all_features), len(all_features[0]), len(all_features[0][0]))
        print("Done extracting features")
        return all_features, all_cands, all_src
