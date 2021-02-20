# PCNN+RL+HME

&emsp;&emsp;This project is implemented for paper [RH-Net: Improving Neural Relation Extraction via Reinforcement Learning and Hierarchical Relational Searching](https://arxiv.org/pdf/2010.14255.pdf), which is belongs to Jianing Wang.

The old version paper is Improving Reinforcement Learning for Neural Relation Extraction with Hierarchical Memory Extractor.

Abstract:

&emsp;&emsp;Distant supervision (DS) aims to generate large-scale heuristic labeling corpus, which is widely used for neural relation extraction currently. However, it heavily suffers from noisy labeling and long-tail distributions. Many advanced approaches usually separately address two problems, which ignore their mutual interactions. In this paper, we propose a novel framework named RH-Net, which utilizes Reinforcement learning and Hierarchical relational searching module to improve relation extraction. We leverage reinforcement learning to instruct the model to select high quality instances. We then propose the hierarchical relational searching module to share the semantics from correlative instances between data-rich and data-poor classes. During iterative process, the two modules keep interacting to alleviate the noisy and long tail problem simultaneously. Extensive experiments on widely used NYT data set clearly show that our method significant improvements over state-of-the-art baselines.

## settings

This code relies on: python3.0+ ; pytorch1.0+ ; numpy ; 

## preparation

First. You should download the origin dataset from https://pan.baidu.com/s/1TseODAVBXYuRqlBS0jl7AQ (code:6u4i)
Second. You need to create some dictionaries,such as 'model' (store the model that trained) ,'data' (the preprocessing of origin dataset).

## preprocess data

You can run to preprocess the data, by:

> python3 initial.py

## pretrain the sentence encoder/ instance detector

> python3 pretrain.py --use_pcnn=true --test_select=true

after pre-train, you can get some pre-trained model files in './model' dictionary, and then you can joint train the whole modules consists of sentence encoder, instance detector and hierarchical memory extractor.

> python3 main.py

At last, you can validate the model on the testing data, by:

> python3 test.py --use_pcnn=true --test_select=false


## plot
We provide the  PR curve results of our model and other baselines, you can run:

> python3 ./plot/draw_plot.py

The PR curve is:

The result is :![](https://github.com/wjn1996/PCNN_RL_HME/blob/main/plot/prcurve_PCNN%2BRL%2BHME.png)

We also implement some extensive experiments, the results and analysis can be found in our papers.

---

You can cite this paper by:

```
@article{DBLP:journals/corr/abs-2010-14255,
  author    = {Jianing Wang and
               Chong Su},
  title     = {Improving Reinforcement Learning for Neural Relation Extraction with
               Hierarchical Memory Extractor},
  journal   = {CoRR},
  volume    = {abs/2010.14255},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.14255},
  archivePrefix = {arXiv},
  eprint    = {2010.14255},
  timestamp = {Mon, 02 Nov 2020 18:17:09 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2010-14255.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
