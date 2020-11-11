# PCNN+RL+HME

&emsp;&emsp;This project is implemented for paper [Improving Reinforcement Learning for Neural Relation Extraction with Hierarchical Memory Extractor](https://arxiv.org/pdf/2010.14255.pdf) authors are Jianing Wang and Chong Su

Abstract:

&emsp;&emsp;Distant supervision relation extraction (DSRE) is an efficient method to extract semantic relations on a large-scale heuristic labeling corpus. However, it usually brings in a massive noisy data. In order to alleviate this problem, many recent approaches adopt reinforcement learning (RL), which aims to select correct data autonomously before relation classification. Although these RL methods outperform conventional multi-instance learning-based methods, there are still two neglected problems: 1) the existing RL methods ignore the feedback of noisy data, 2) the reduction of training corpus exacerbates long-tail problem. In this paper, we propose a novel framework to solve the two problems mentioned above. Firstly, we design a novel reward function to obtain feedback from both correct and noisy data. In addition, we use implicit relations information to improve RL. Secondly, we propose the hierarchical memory extractor (HME), which utilizes the gating mechanism to share the semantics from correlative instances between data-rich and data-poor classes. Moreover, we define a hierarchical weighted ranking loss function to implement top-down search processing. Extensive experiments conducted on the widely used NYT dataset show significant improvement over state-of-the-art baseline methods.


