# Controllable Text Simplification with Explicit Paraphrasing

This repository contains the code and resources from the following [paper](https://arxiv.org/pdf/2010.11004.pdf). Our approach simplifies the given complex sentence in three steps:

1. Generate candidates for an input sentence using [DisSim](https://www.aclweb.org/anthology/P19-1333.pdf) and neural sentence splitter. DisSim is a rule-based approach proposed by Nikluas et al. 2019 that uses 35 syntactic rules to split a sentence. 

1. Rank the candidates that have undergone splitting and deletion based on the quality of simplification.  

1. Pass the best ranked candidate to the paraphrase generation Transformer model.


## Candidate Generation: 

First, you need to install the DiscourseSimplification code. We use the same code from [this](https://github.com/Lambda-3/DiscourseSimplification) repo.

```
cd DiscourseSimplification
mvn clean install -DskipTests
```

To generate the candidates, you can use the following command:

```python3 generate_candidates.py --input <input filename> --output <candidate filename>```
    
## Candidate Ranking: 

To rank the candidates generated in the previous step,  you can use the following command:

```
python3 ranking/main.py --input <input filename> --candidates <candidate filename> --output <best ranked candidate filename>
```

## Paraphrase Generation:

Coming Soon.

## Citation
Please cite if you use the above resources for your research
```
@InProceedings{NAACL-2021-Maddela,
  author = 	"Maddela, Mounica and Alva-Manchego, Fernando and Xu, Wei",
  title = 	"Controllable Text Simplification with Explicit Paraphrasing",
  booktitle = 	"Proceedings of the North American Association for Computational Linguistics (NAACL)",
  year = 	"2021",
}
```

