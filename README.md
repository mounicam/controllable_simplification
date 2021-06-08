# Controllable Text Simplification with Explicit Paraphrasing

This repository contains the code and resources from the following [paper](https://arxiv.org/pdf/2010.11004.pdf). Our approach simplifies the given complex sentence in three steps. 

1. Generate candidates for an input sentence using [DisSim](https://www.aclweb.org/anthology/P19-1333.pdf) and neural sentence splitter. DisSim is a rule-based approach proposed by Nikluas et al. 2019 that uses 35 syntactic rules to split a sentence. 

1. Rank the candidates that have undergone splitting and deletion based on the quality of simplification.  

1. Pass the best ranked candidate to the paraphrase generation Transformer model.


## Candidate Generation: 

## Candidate Ranking: 

## Paraphrase Generation:



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

