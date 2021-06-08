import torch
import argparse
from models import mr_ranker
from features.feature_extractor import FeatureExtractor


def rerank(segs, segs_feats, model):
    score_map = {}
    predicted_scores, segs = model.predict(segs_feats, segs)
    for ind, seg in enumerate(segs):
        score_map[seg] = predicted_scores[ind]
    return sorted(score_map.keys(), key=score_map.__getitem__, reverse=True)


def main(args):

    feature_extractor = FeatureExtractor()
    test_feats, test_cands, test_src = feature_extractor.get_features(args.input, args.candidates)

    model = torch.load(args.model)
    model.model.eval()

    top_simplifications = []
    i = 0
    for segs_feats, segs, src in zip(test_feats, test_cands, test_src):

        if i % 1000 == 0:
            print(i)
        i += 1

        if len(segs) == 0:
            top_simplifications.append([src])
        else:
            reranked_segs = rerank(segs, segs_feats, model)
            top_simplifications.append(reranked_segs)

    if args.output is not None:
        fp = open(args.output, 'w')
        for segs in top_simplifications:
            fp.write(segs[0] + "\n")
        fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', default='ranking/model.bin', type=str)
    parser.add_argument("--output", dest="output", help="Best ranked candidate.", type=str)
    parser.add_argument('--input', help="Input sentences with one sentence in each line.")
    parser.add_argument('--candidates', help="Candidates for each input sentence seperated by tabs. \n"
                                         "The format for each candidate is "
                                         "<candidate>|||<DisSim|Transformer>|||<Rules applied to obtain the candidate>")
    args = parser.parse_args()
    main(args)

