# Black-box Predictions via Influence Functions

*46927 Final Project*


**Authors:** Ze Yang, Zhengyang Qi, Yundong Liu, Yuze Liu

## Experiments

- `experiments.ipynb`: Ridge on *ForestFires* dataset.
- `experiments2.ipynb`: Binary Logistic Regression on *Iris* and *MNIST*.
- `experiments3.ipynb`: Smoothed support vector on *Iris*.
- `experiments4.ipynb`: Analysis of the components of loss influence function, using *Marketing* dataset.

## Data

- *ForestFires*, *Iris*, *MNIST*, *Marketing*: See `/data/__init__.py`


## TODO

- Report
- Beamer Slides
- More experiments; Images; Interpretations **(Done)**
- Implement 2-Layers Perceptron **(Done)**
- Implement (maybe) a simple CNN
- LiSSA approximation [(2)][2] **(Done)**
- Fancy Plots **(Done)**
- Logistic Regression influence terms illustration **(Done)**
- Implement Binary Logistic Regression **(Done)**
- Implement Smoothed SVC **(Done)**
- Implement Regularized Regression **(Done)**
- Conjugate Gradient approximation **(Done)**
- Improve Optimization Routine **(Done)**

## Introduction

Our final project will be based on the paper “Understanding Black-Box Predictions via Influence Functions” by Pang Wei Koh and Percy Liang[(1)][1], which discusses how influence function(3) can trace the model’s prediction through the learning algorithm and the training data. For now, we decide to divide the projects into three parts. First of all, we will figure out the mathematical formula behind influence function. Secondly, we will study and implement the two techniques in efficiently calculating influence function discussed in the paper. Last but not least, we will apply the influence functions to other algorithms that we have learned in class and that are not discussed in the paper (e.g. Ridge Regression and Trees). Through this project, we intend to have a relatively thorough understanding of how influence functions can give us insights on training data.



## Reference
1. Koh PW, Liang P. [*Understanding Black-box Predictions via Influence Functions*][1]. International Conference on Machine Learning, 2017.
2. Agarwal N, Bullins B, Hazan E. [*Second-Order Stochastic Optimization for Machine Learning in Linear Time*][2]. The Journal of Machine Learning Research. 2017 Jan 1; 18(1):4148-87.
3. Wassermann L. *All of nonparametric statistics*. New York. 2006.

[1]: https://arxiv.org/abs/1703.04730 
[2]: https://arxiv.org/abs/1602.03943

