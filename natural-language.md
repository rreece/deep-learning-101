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
9.  [Computational complexity of transformers](#computational-complexity-of-transformers)
10. [Efficient transformers: Inference optimizations](#efficient-transformers-inference-optimizations)
11. [Efficient transformers: Architecture modifications](#efficient-transformers-architecture-modifications)
12. [What comes after transformer?](#what-comes-after-transformer)
13. [Reasoning models](#reasoning-models)
14. [Conclusion](#conclusion)


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
    -   First very successful encoder-decoder based model
    -   *Watershed moment in NLP with deep learning*
-   Bahdanau "attention"
    -   Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473).
-   Google Neural Machine Translation (GNMT)
    -   Wu, Y. et al. (2016). [Google’s neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/abs/1409.0473).

Chain rule of language modeling (chain rule of probability):

```math
P(x_1, \ldots, x_T) = P(x_1, \ldots, x_{n-1}) \prod_{t=n}^{T} P(x_t | x_1 \ldots x_{t-1})
```

or for the whole sequence:

```math
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t | x_1 \ldots x_{t-1}) = P(x_1) P(x_2 | x_1) P(x_3 | x_1 x_2) P(x_4 | x_1 x_2 x_3) \ldots
```

A *language model* (LM), predicts the next token given previous context.
The output of the model is a vector of logits, which is given to a softmax
to convert to probabilities for the next token.

```math
P(x_t | x_1 \ldots x_{t-1}) = \mathrm{model}(x_1 \ldots x_{t-1}) = \underset{V}{\mathrm{softmax}}\left( \mathrm{logits}(x_1 \ldots x_{t-1}) \right)
```

*Auto-regressive* inference follows this chain rule.
If done with greedy search:

```math
\hat{x}_{t} = \underset{x_t \in V}{\mathrm{argmax}} \ P(x_t | x_1 \ldots x_{t-1})
```

Beam search:

-   Beam search as used in NLP is described in Sutskever (2014).


## Transformer

-   Vaswani, A. et al. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762).

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

-   Zhang, B. & Sennrich, R. (2019). [Root mean square layer normalization](https://arxiv.org/abs/1910.07467).
-   Xiong, R. et al. (2020). [On layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745).
-   Phuong, M. & Hutter, M. (2022). [Formal algorithms for transformers](https://arxiv.org/abs/2207.09238).
-   Karpathy, A. (2024). Video: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU).
    -   [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
-   [karpathy/LLM101n](https://github.com/karpathy/LLM101n)

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
-   Self-supervised pretraining
-   GPT: Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
-   GPT-2: Radford, A. et al. (2019). [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).
-   GPT-3: Brown, T.B. et al. (2020). [Language models are few-shot learners](https://arxiv.org/abs/2005.14165).
-   InstructGPT: Ouyang, L. et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
    -   Instruction finetuning
    -   Reinforcement Learning from Human Feedback (RLHF)
-   ChatGPT based on GPT-3 initially released by OpenAI on November 30, 2022
-   GPT-4: OpenAI. (2023). [GPT-4 technical report](https://cdn.openai.com/papers/gpt-4.pdf).

![Development of ChatGPT (source: [2302.10724](https://arxiv.org/abs/2302.10724)).](img/development-of-chatgpt.png)


### Post-training

-   Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347). (PPO)
-   Direct Preference Optimization (DPO)
-   Rafailov, R. (2023). [Direct Preference Optimization: Your language model is secretly a reward model](https://arxiv.org/abs/2305.18290). 
-   SuperAnnotate. (2024). [Direct preference optimization (DPO): Complete overview](https://www.superannotate.com/blog/direct-preference-optimization-dpo).
-   Blog: [RLHF progress: Scaling DPO to 70B](https://www.interconnects.ai/p/rlhf-progress-scaling-dpo-to-70b).
-   Group Relative Policy Optimization (GRPO)
-   Shao, Z. (2024). [DeepSeekMath: Pushing the limits of mathematical reasoning in open language models](https://arxiv.org/abs/2402.03300).
-   Lambert, N. (2024). [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com).
-   Databricks. (2025). [TAO: Using test-time compute to train efficient LLMs without labeled data](https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data).


### Mixture of Experts (MoE)

-   Jordan, M.I. & Jacobs, R.A. (1994). [Hierarchical mixtures of experts and the EM algorithm](https://doi.org/10.1162/neco.1994.6.2.181).
-   Bengio, Y., L&eacute;onard, N., & Courville, A. (2013). [Estimating or propagating gradients through stochastic neurons for conditional computation](https://arxiv.org/abs/1308.3432).
-   Eigen, D., Ranzato, M., & Sutskever, I. (2013). [Learning factored representations in a deep mixture of experts](https://arxiv.org/abs/1312.4314).


### Decoder-only models like GPT

#### Closed source

-   GPT-4 (OpenAI)
-   Chinchilla (DeepMind)
-   Gemini (Google)
-   Claude (Anthropic)

#### Open source

-   Falcon (TII)
-   Llama (Meta)
    -   Meta. (2024). [Introducing Llama 3.1: Our most capable models to date](https://ai.meta.com/blog/meta-llama-3-1/).
    -   Dubey, A. et al. (2024). [The Llama 3 herd of models](https://arxiv.org/abs/2407.21783).
    -   Ainslie, J. (2023). [GQA: Training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245). - GQA
    -   Zhang, B. & Sennrich, R. (2019). [Root mean square layer normalization](https://arxiv.org/abs/1910.07467). - RMSNorm
    -   Ramachandran et al. (2017). [Searching for activation functions](https://arxiv.org/abs/1710.05941). - SiLU/Swish activation function used in FFN
    -   Meta. (2025). [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/). - First Llamas to use MoE
-   Nemotron (Nvidia)
    -   Nvidia. (2024). [Nemotron-4 340B technical report](https://arxiv.org/abs/2406.11704).
-   Teuken (OpenGPT-X, Fraunhofer IAIS, Fraunhofer IIS)
    -   OpenGPT-X. (2024). [Data processing for the OpenGPT-X model family](https://arxiv.org/abs/2410.08800).
    -   OpenGPT-X. (2024). [Teuken-7B-Base & Teuken-7B-Instruct: Towards European LLMs](https://arxiv.org/abs/2410.03730).
    -   OpenGPT-X. (2024). [Towards multilingual LLM evaluation for European languages](https://arxiv.org/abs/2410.08928).
-   Qwen (Alibaba)
    -   Qwen. (2024). [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115).
-   Gemma (Google)
    -   Google. (2025). [Introducing Gemma 3: The most capable model you can run on a single GPU or TPU](https://blog.google/technology/developers/gemma-3/).
-   Command A (Cohere)
    -   Cohere. (2025). [Command A: An enterprise-ready large language model](https://cohere.com/research/papers/command-a-technical-report.pdf).


## Computational complexity of transformers

-   Chen, C. (2022). [Transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/).
-   Bahdanau, D. (2022). [The FLOPs calculus of language model training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).
-   Sanger, A. (2023). [Inference characteristics of Llama-2](https://cursor.sh/blog/llama-inference). 
-   Shenoy, V. & Kiely, P. (2023). [A guide to LLM inference and performance](https://www.baseten.co/blog/llm-transformer-inference-guide/).
-   Anthony, Q., Biderman, S., & Schoelkopf, H. (2023). [Transformer math 101](https://blog.eleuther.ai/transformer-math/).
-   Ouyang, A. (2023). [*Understanding the Performance of Transformer*](https://dspace.mit.edu/handle/1721.1/151543). (MS thesis)
-   Casson, A. (2023). [Transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops).

```
FLOPs ~ n_layers * [4 * d_model**2 + 2 * sequence_length * d_model + 2 * d_model * d_ff] + d_model * vocab_size
```


## Efficient transformers: Inference optimizations

-   Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Ré, C. (2022). [FlashAttention: Fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135).
-   Pope, R. et al. (2022). [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102). - *KV cache*
-   Leviathan, Y., Kalman, M., & Matias, Y. (2022). [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192).
-   Dao, T. (2023). [FlashAttention-2: Faster attention with better parallelism and work partitioning](https://arxiv.org/abs/2307.08691).
-   Kim, S. et al. (2023). [Full stack optimization of transformer inference: A survey](https://arxiv.org/abs/2302.14017).
-   PyTorch. (2023). [Accelerating generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/).
-   Nvidia. (2023). [Mastering LLM techniques: Inference optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization).
-   Weng, L. (2023). [Large transformer model inference optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/).
-   Kwon, W. et al. (2023). [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180).
-   Zhang, L. (2023). [Dissecting the runtime performance of the training, fine-tuning, and inference of large language models](https://arxiv.org/abs/2311.03687).
-   Fu, Y. (2023). [Towards 100x speedup: Full stack transformer inference optimization](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c#5e3b9489c0204f8c8d70d014a9e88b28).
-   Fu, Y. (2024). [Challenges in deploying long-context transformers: A theoretical peak performance analysis](https://arxiv.org/abs/2405.08944).
-   Fu, Y. et al. (2024). [Data engineering for scaling language models to 128K context](https://arxiv.org/abs/2402.10171).
-   Kwon, W. et al. (2023). [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180). ([vLLM](https://github.com/vllm-project/vllm))
-   Nvidia. (2023). [Mastering LLM techniques: Inference optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization).
-   Chng, P. (2024). [What is the transformer KV cache?](https://peterchng.com/blog/2024/06/11/what-is-the-transformer-kv-cache/)
-   Shah, J. et al. (2024). [FlashAttention-3: Fast and accurate attention with asynchrony and low-precision](https://arxiv.org/abs/2407.08608).
-   Shi, L. et al. (2024). [Keep the cost down: A review on methods to optimize LLM' s KV-cache consumption](https://arxiv.org/abs/2407.18003).


## Efficient transformers: Architecture modifications

-   Shazeer, N. (2019). [Fast transformer decoding: One write-head is all you need](https://arxiv.org/abs/1911.02150). - MQA
-   Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). [Efficient transformers: A survey](https://arxiv.org/abs/2009.06732).
-   Leviathan, Y., Kalman, M., & Matias, Y. (2022). [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192).
-   Ainslie, J. (2023). [GQA: Training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245). - GQA
-   DeepSeek. (2024). [DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model](https://arxiv.org/abs/2405.04434). - Multi-head Latent Attention (MLA)
-   Meng, F., Yao, Z., & Zhang, M. (2025). [TransMLA: Multi-head latent attention is all you need](https://arxiv.org/abs/2502.07864).


## What comes after transformer?

-   The return of recurrence?
-   SSMs and Mamba
-   Gu, A., Goel, K., & R&eacute;, C. (2021). [Efficiently modeling long sequences with structured state spaces](https://arxiv.org/abs/2111.00396).
-   Merrill, W. & Sabharwal, A. (2022). [The parallelism tradeoff: Limitations of log-precision transformers](https://arxiv.org/abs/2207.00729).
-   Bulatov, A., Kuratov, Y., & Burtsev, M.S. (2022). [Recurrent memory transformer](https://arxiv.org/abs/2207.06881).
-   Raffel, C. (2023). [A new alchemy: Language model development as a subfield?](https://colinraffel.com/blog/language-model-development-as-a-new-subfield.html).
-   Bulatov, A., Kuratov, Y., & Burtsev, M.S. (2023). [Scaling transformer to 1M tokens and beyond with RMT](https://arxiv.org/abs/2304.11062).
-   Bertsch, A., Alon, U., Neubig, G., & Gormley, M.R. (2023). [Unlimiformer: Long-range transformers with unlimited length input](https://arxiv.org/abs/2305.01625).
-   Peng, B. et al. (2023). [RWKV: Reinventing RNNs for the transformer era](https://arxiv.org/abs/2305.13048).
-   Sun, Y. et al. (2023). [Retentive network: A successor to transformer for large language models](https://arxiv.org/abs/2307.08621).
-   Gu, A. & Dao, T. (2023). [Mamba: Linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752).
-   Wang, H. et al. (2023). [BitNet: Scaling 1-bit transformers for large language models](https://arxiv.org/abs/2310.11453).
-   Ma, S. et al. (2024). [The era of 1-bit LLMs: All large language models are in 1.58 bits](https://arxiv.org/abs/2402.17764).
-   Ma, X. et al. (2024). [Megalodon: Efficient LLM pretraining and inference with unlimited context length](https://arxiv.org/abs/2404.08801).
-   Sun, Y. et al. (2024). [Learning to (learn at test time): RNNs with expressive hidden states](https://arxiv.org/abs/2407.04620).
-   Dao, T. & Gu, A. (2024). [Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality](https://arxiv.org/abs/2405.21060).
-   Beck, M. et al. (2025). [xLSTM 7B: A recurrent LLM for fast and efficient inference](https://arxiv.org/abs/2503.13427).


## Reasoning models

-   Chain of Thought (CoT) prompting
    -   Kojima, T. et al. (2022). [Large language models are zero-shot reasoners](https://arxiv.org/abs/2205.11916).
-   Augmented language models
    -   Mialon, G. et al. (2023). [Augmented language models: A survey](https://arxiv.org/abs/2302.07842).
    -   Bhargava, A., Witkowski, C., Shah, M., & Thomson, M. (2023). [What's the magic word? A control theory of LLM prompting](https://arxiv.org/abs/2310.04444).
-   Scaling test-time compute
    -   See talks and work by Noam Brown in poker and other games.

#### Closed source

-   o3 (OpenAI)
    -   OpenAI. (2024). [OpenAI o1 system card](https://cdn.openai.com/o1-system-card-20241205.pdf).
    -   OpenAI. (2025). [OpenAI o3-mini system card](https://cdn.openai.com/o3-mini-system-card-feb10.pdf).
-   Claude 3.7 Sonnet (Anthropic)
    -   Anthropic. (2025). [Claude 3.7 Sonnet and Claude Code](https://www.anthropic.com/news/claude-3-7-sonnet).

#### Open source

-   R1 (DeepSeek)
    -   DeepSeek. (2025). [DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning](https://arxiv.org/abs/2501.12948).
-   s1 (Stanford)
    -   Muennighoff, N. et al. (2025). [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393).
-   QwQ-32B (Alibaba)
    -   Qwen. (2025). [QwQ-32B: Embracing the power of reinforcement learning](https://qwenlm.github.io/blog/qwq-32b/).
-   Deep Cogito
    -   Deep Cogito. (2025). [Cogito v1 Preview: Introducing IDA as a path to general superintelligence](https://www.deepcogito.com/research/cogito-v1-preview).


## Conclusion

![Evolutionary tree of LLMs (source: [2304.13712](https://arxiv.org/abs/2304.13712)).](img/evolutionary-tree-of-LLMs.png)


### Surveys

2022

-   Bommasani, R. et al. (2022). [On the opportunities and risks of foundation models](https://arxiv.org/abs/2108.07258).

2023

-   Yang, J. et al. (2023). [Harnessing the power of LLMs in practice: A survey on ChatGPT and beyond](https://arxiv.org/abs/2304.13712).
-   Raschka, S. (2023). [Understanding large language models](https://magazine.sebastianraschka.com/p/understanding-large-language-models).
-   Mohamadi, S. et al. (2023). [ChatGPT in the age of generative AI and large language models: A concise survey](https://arxiv.org/abs/2307.04251v1).
-   Zhao, W.X. et al. (2023). [A survey of large language models](https://arxiv.org/abs/2303.18223).
-   Bowman, S.R. (2023). [Eight things to know about large language models](https://arxiv.org/abs/2304.00612).
-   Timbers, F. (2023). [Five years of GPT progress](https://finbarr.ca/five-years-of-gpt-progress/)
-   Chen, C. (2023). [Transformer taxonomy](https://kipp.ly/transformer-taxonomy/).
-   Naveed, H. (2023). [A comprehensive overview of large language models](https://arxiv.org/abs/2307.06435).

2024

-   [Anti-hype LLM reading list](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)
-   Willison, S. (2024). [Things we learned about LLMs in 2024](https://simonwillison.net/2024/Dec/31/llms-in-2024/).


### Misc

-   Banerjee, S., Agarwal, A., & Singla, S. (2024). [LLMs will always hallucinate, and we need to live with this](https://arxiv.org/abs/2409.05746).


--------

-   Up next: [Parallelism and hardware](parallelism-and-hw.md)
-   Previous: [Computer vision](computer-vision.md)

