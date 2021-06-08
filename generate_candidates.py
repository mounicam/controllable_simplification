import os
import dissim
import argparse


def main(args):
    cwd = os.getcwd()

    # Runs DisSim to generate candidates
    os.system("cp " + args.input + " DiscourseSimplification/input.txt")
    os.chdir('DiscourseSimplification')
    os.system("mvn clean compile exec:java")
    os.chdir(cwd)
    dissim_candidates = dissim.generate_candidates(args.input, "DiscourseSimplification/output_dt.txt")

    # TODO: add neural splitter candidates.

    fpout = open(args.output, "w")
    for candidates in dissim_candidates:
        fpout.write("\t".join(candidates) + "\n")
    fpout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate DisSim candidates that have undergone splitting and deletion.')
    parser.add_argument('--input', help="Input sentences with one sentence in each line.")
    parser.add_argument('--output', help="Candidates for each input sentence seperated by tabs. \n"
                                         "The format for each candidate is "
                                         "<candidate>|||<DisSim|Transformer>|||<Rules applied to obtain the candidate>")
    args = parser.parse_args()
    main(args)
