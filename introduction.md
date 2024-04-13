# Introduction

![AI vs ML vs DL (source: [figshare](https://figshare.com/articles/figure/AI_vs_ML_vs_DL_Venn_Diagram_png/14915505))](img/AI_vs_ML_vs_DL_Venn_Diagram.png)

1. *Artificial Intellegence* is any kind of software that makes intellegent decisions in some sense. This could be a simple as hard-coded expert heuristics.
2. *Machine Learning* is a kind of software that somehow improves (is trained) when given data. Expert knowledge is often used to structure the model and what features of the data are used.
3. *Deep Learning* is a recent paradigm of machine learning using large artificial neural networks,
    with many layers for feature learning. Models are more of a blackbox that learn features from the raw data.

In the general machine learning setup you have a model that can be
thought of as parameterized function, $f$,
that takes in some vector of input data $\vec{x}$,
and returns some prediction $y$.
Internally the model is parameterized by weights, $\vec{w}$.

$$ y = f(\vec{x}; \vec{w}) $$

In the paradigm of supervized learning, there is a training dataset
that has input features paired with the truth label to be predicted, $\{\vec{x}_{j}, \ell_{j}\}$.

TODO:

-   NNs are *universal function approximators*


## Gradient descent

The workhorse algorithm for optimizing (training) model parameters is *gradient descent*:

$$ \vec{w}[t+1] = \vec{w}[t] - \eta \frac{\partial L}{\partial \vec{w}}[t] $$

In *Stochastic Gradient Descent* (SGD), you chunk the training data into *minibatches* (AKA batches), $x_{t}$,
and take a gradient descent step with each minibatch:

$$ \vec{w}[t+1] = \vec{w}[t] - \eta \frac{\partial L}{\partial \vec{w}}[x_{t}] $$

where

-   $\eta$ is the *learning rate*
-   $\frac{\partial L}{\partial \vec{w}}$ is the *gradient*

TODO:

-   Backpropagation
-   Double descent


## See also

### Pedagogy

Classical ML Textbooks:

-   Bishop, C.M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).
-   Hastie, T., Tibshirani, R., & Friedman, J. (2009). [*The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.)](https://hastie.su.domains/Papers/ESLII.pdf).

Deep Learning Textbooks:

-   Bishop, C.M. (2024). [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/).
-   Goodfellow, I., Bengio, Y., & Courville, A. (2016). [*Deep Learning*](http://www.deeplearningbook.org).

Online courses:

-   Ustyuzhanin, A. (2020). [Deep Learning 101](https://indico.cern.ch/event/882244/sessions/348901/attachments/2045473/3426793/day3_deep-learning-101.pdf).
-   Bekman, S. (2023). [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering).
-   Labonne, M. (2023). [Large Language Model Course](https://github.com/mlabonne/llm-course).
-   Microsoft. (2023). [Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners).

