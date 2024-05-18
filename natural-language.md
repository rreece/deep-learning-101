# Natural language


### Contents

1.  [Tokenization](#tokenization)
2.  [Input tensor shape](#input-tensor-shape)
3.  [Embedding (word2vec)](#embedding-word2vec)
4.  [seq2seq](#seq2seq)
5.  [Transformer](#transformer)
6.  [BERT](#bert)
7.  [T5](#t5)
8.  [GPT](#gpt)
9.  [Conclusion](#conclusion)


## Tokenization

Tokenization is the process of taking a sequence of text
and breaking it into units called *tokens*.
You can think of tokens as being words,
but in general they can be parts of words.

Tokens are generally then converted to "token IDs" that
are integer encodings of the tokens.

Example:

```
text = "Hello, world! This is tokenization."
tokens = ["<start>", "Hello", ",", " ", "world", "!", " ", "This", " ", "is", " ", "token", "iza", "tion", ".", "<end>"]
token_ids = [1, 123, 22, 2223, 10, 335, 556, 10, ... ]
```

-   Tokenization is basically a map from word parts to integers.
-   It is important to note that tokenization is dependent on a *vocabulary* used to make the map.
-   So note that a certain tokenization may not support any language.
    The language needs to be in vocabulary.
-   A typical vocabulary size is something like $\sim$ 50,000.

See also:

-   Tutorial video by Andrej Karpathy: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)


## Input tensor shape

Often tokenization is done in the `DataLoader`, which also forms batches of the data in the form of a tensor for the model.
To square up the input tensor size, often one needs to *pad* the sequences to a common *max sequence length* (MSL).

Often the pad token ID is `0`, so a padded sequence would look like

```
token_ids = [1, 123, 22, 2223, 10, 335, 556, 10, ..., 0, 0, 0]
```

The input tensor shape for language models is often:

```
[batch_size][max_seq_length]  =  e.g. [8][256]
```


## Embedding (word2vec)

After tokenization, the next step in a language model is to *embed* the tokens,
which is a map from the token IDs to a vector in some large space,
with dimension called the `embedding_size`.

The tensor shape of the output of the embedding is

```
[batch_size][max_seq_length][embedding_size]  =  e.g. [8][256][1280]
```

After the embedding parameters are trained end-to-end with a model,
remarkably, you can give some *semantic interpretations* to some basis
vectors in the embedding space.  Famously, for example

$$ \vec{E}(\mathrm{king}) - \vec{E}(\mathrm{man}) + \vec{E}(\mathrm{woman}) \approx \vec{E}(\mathrm{queen}) $$

![word2vec visualization 1 (source: https://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html/).](img/word2vec-viz-1.png)

Another example where a dimension in the embedding correlates with the capital of a country:

![word2vec visualization 2 (source: https://arxiv.org/abs/1310.4546).](img/word2vec-viz-2.png)

See also:

-   Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781).
-   Mikolov, T. et al. (2013). [Distributed representations of words and phrases and their compositionality](https://arxiv.org/abs/1310.4546).
-   Mikolov, T., Yih, W. T., & Zweig, G. (2013). [Linguistic regularities in continuous space word representations](https://www.aclweb.org/anthology/N13-1090.pdf).
-   Olah, C. (2014). [Deep learning, NLP, and representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/).


## seq2seq

-   RNNs and LSTMs
    -   Olah, C. (2015). [Understanding LSTM networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
-   seq2seq
    -   Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215).
    -   *Watershed moment in NLP with deep learning*
    -   First very successful encoder-decoder based model
-   Bahdanau "attention"
    -   Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473).
-   Google Neural Machine Translation (GNMT)
    -   Wu, Y. et al. (2016). [Google’s neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/abs/1409.0473).

Chain rule of language modeling (chain rule of probability):

$$ P(x_1, \ldots, x_T) = P(x_1, \ldots, x_{n-1}) \prod_{t=n}^{T} P(x_t | x_1 \ldots x_{t-1}) $$

or for the whole sequence:

$$ P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t | x_1 \ldots x_{t-1}) $$

$$ = P(x_1) P(x_2 | x_1) P(x_3 | x_1 x_2) P(x_4 | x_1 x_2 x_3) \ldots $$

A *language model* (LM), predicts the next token given previous context.
The output of the model is a vector of logits, which is given to a softmax
to convert to probabilities for the next token.

$$ P(x_t | x_1 \ldots x_{t-1}) = \mathrm{model}(x_1 \ldots x_{t-1}) = \underset{V}{\mathrm{softmax}}\left( \mathrm{logits}(x_1 \ldots x_{t-1}) \right) $$

*Auto-regressive* inference follows this chain rule.
If done with greedy search:

$$ \hat{x}_{t} = $$

$$ \underset{x_t \in V}{\mathrm{argmax}} \ P(x_t | x_1 \ldots x_{t-1}) $$

Beam search:

-   Beam search as used in NLP is described in Sutskever (2014).


## Transformer

![Diagram of the Transformer model (source: [d2l.ai](https://d2l.ai/index.html)).](img/transformer.png)

-   Describe architecture
-   Describe self-attention
-   Note the complexity is $T^2$

$$ \mathrm{attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\intercal}{\sqrt{d_k}}\right) V $$

Autoregressive decoding:

![Autoregressive decoding. source: https://hrithickcodes.medium.com/the-math-behind-the-machine-a-deep-dive-into-the-transformer-architecture-a3902333e4a4](img/transformer-autoregressive-decode.gif)

KV-cache:

![KV-cache explained. source: https://medium.com/@joaolages/kv-caching-explained-276520203249](img/transformer-kv-cache.gif)

Quadratic complexity in sequence length:

-   Note a lot of research in reducing the quadratic complexity
-   Note a lot of research in extending context length (e.g., [llama3 has 8k context](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md))
-   Note [Mamba](https://arxiv.org/abs/2312.00752) claims to have linear in $T$ complexity

Note that there are also variants of transformers that move and/or change
the normalization layers. Most transformers now user "pre-layer-norm" unlike the original.

![Pre-layer-norm transformer (source: [2002.04745](https://arxiv.org/abs/2002.04745)).](img/pre-layer-norm-transformer.png)

Some transformer models (e.g., llama3) use RMSNorm instead of LayerNorm.

See also:

-   Vaswani, A. et al. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762).
-   Zhang, B. & Sennrich, R. (2019). [Root mean square layer normalization](https://arxiv.org/abs/1910.07467).
-   Xiong, R. et al. (2020). [On layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745).
-   Phuong, M. & Hutter, M. (2022). [Formal algorithms for transformers](https://arxiv.org/abs/2207.09238).


## BERT

-   BERT is encoder-only
-   BERT has bidirectional attention
-   Pretrained with masked language modeling (MLM)
-   For encoding: sequence to vector, or for classification tasks: seq to class
-   Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018). [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805).
-   Liu, Y. et al. (2019). [RoBERTa: A robustly optimized BERT pretraining approach](https://arxiv.org/abs/1907.11692).


## T5

-   T5 is an encoder-decoder
-   Encoder-decoder good for sequence-to-sequence modeling: translation, summarization
-   T5 also demonstrated that classification tasks can be done as sequence-to-sequence
-   Causal attention
-   Recap various attention schemes

![T5 description of types of transformer architectures (source: [1910.10683](https://arxiv.org/abs/1910.10683)).](img/t5-description-of-transformer-types.png)

-   Raffel, C. et al. (2019). [Exploring the limits of transfer learning with a unified text-to-text transformer](https://arxiv.org/abs/1910.10683).


## GPT

-   GPT is decoder-only
-   Causal attention
-   Decoder-only models like GPT:
    -   Llama (Meta, open)
    -   Falcon (TII, open)
    -   GPT-4 (OpenAI, closed)
    -   Chinchilla (DeepMind, closed)
    -   Claude (Anthropic, closed)
-   Instruction finetuning
-   Reinforcement Learning from Human Feedback (RLHF)
-   Direct Preference Optimization (DPO)

![Development of ChatGPT (source: [2302.10724](https://arxiv.org/abs/2302.10724)).](img/development-of-chatgpt.png)

-   GPT: Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
-   GPT-2: Radford, A. et al. (2019). [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).
-   GPT-3: Brown, T.B. et al. (2020). [Language models are few-shot learners](https://arxiv.org/abs/2005.14165).
-   InstructGPT: Ouyang, L. et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
-   GPT-4: OpenAI. (2023). [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf).
-   Rafailov, R. (2023). [Direct Preference Optimization: Your language model is secretly a reward model](https://arxiv.org/abs/2305.18290). 
-   Blog: [RLHF progress: Scaling DPO to 70B](https://www.interconnects.ai/p/rlhf-progress-scaling-dpo-to-70b).


## Conclusion

![Evolutionary tree of LLMs (source: [2304.13712](https://arxiv.org/abs/2304.13712)).](img/evolutionary-tree-of-LLMs.png)

-   Yang, J. et al. (2023). [Harnessing the power of LLMs in practice: A survey on ChatGPT and beyond](https://arxiv.org/abs/2304.13712).
-   Raschka, S. (2023). [Understanding large language models](https://magazine.sebastianraschka.com/p/understanding-large-language-models).
-   Mohamadi, S. et al. (2023). [ChatGPT in the age of generative AI and large language models: A concise survey](https://arxiv.org/abs/2307.04251v1).

--------

-   Up next: [Parallelism and hardware](parallelism-and-hw.md)
-   Previous: [Computer vision](computer-vision.md)

