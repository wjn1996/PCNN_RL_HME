# PCNN+RL+HME

&emsp;&emsp;This project is implemented for paper [Improving Reinforcement Learning for Neural Relation Extraction with Hierarchical Memory Extractor](https://arxiv.org/pdf/2010.14255.pdf) authors are Jianing Wang and Chong Su

Abstract:

&emsp;&emsp;Distant supervision relation extraction (DSRE) is an efficient method to extract semantic relations on a large-scale heuristic labeling corpus. However, it usually brings in a massive noisy data. In order to alleviate this problem, many recent approaches adopt reinforcement learning (RL), which aims to select correct data autonomously before relation classification. Although these RL methods outperform conventional multi-instance learning-based methods, there are still two neglected problems: 1) the existing RL methods ignore the feedback of noisy data, 2) the reduction of training corpus exacerbates long-tail problem. In this paper, we propose a novel framework to solve the two problems mentioned above. Firstly, we design a novel reward function to obtain feedback from both correct and noisy data. In addition, we use implicit relations information to improve RL. Secondly, we propose the hierarchical memory extractor (HME), which utilizes the gating mechanism to share the semantics from correlative instances between data-rich and data-poor classes. Moreover, we define a hierarchical weighted ranking loss function to implement top-down search processing. Extensive experiments conducted on the widely used NYT dataset show significant improvement over state-of-the-art baseline methods.

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
