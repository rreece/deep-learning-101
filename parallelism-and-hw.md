# Parallelism and hardware


### Contents

1.  [Introduction](#introduction)
2.  [Performance and bandwidth](#performance-and-bandwidth)
3.  [Model parallelism](#model-parallelism)
4.  [Computational complexity of transformers](#computational-complexity-of-transformers)
5.  [Efficient transformers](#efficient-transformers)
6.  [Accelerators](#accelerators)
7.  [Conclusion](#conclusion)


## Introduction

-   Single Instruction/Multiple Data (SIMD) and GPUs
-   [FLOPs vs FMACs](https://medium.com/@pashashaik/a-guide-to-hand-calculating-flops-and-macs-fa5221ce5ccc)
-   [Data parallel vs model parallel vs tensor parallel](https://colossalai.org/docs/concepts/paradigms_of_parallelism/)
-   SRAM vs DRAM

![Examples of GPU memory usage. source: https://arxiv.org/abs/2403.03507](img/galore-memory-usage.png)

-   Hooker, S. (2020). [The hardware lottery](https://arxiv.org/abs/2009.06489).
-   Sevilla, J. et al. (2022). [Compute trends across three eras of machine learning](https://arxiv.org/abs/2202.05924).
-   He, H. (2022). [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html).
-   Geiping, J. & Goldstein, T. (2022). [Cramming: Training a language model on a single GPU in one day](https://arxiv.org/abs/2212.14034).
-   Spector, B. (2024). [GPUs go brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk).


## Performance and bandwidth

Roofline plots:

![Example of a Roofline plot. source: https://commons.wikimedia.org/wiki/File:Example_of_a_Roofline_model.svg](img/roofline_model.png)

-   Williams, S., Waterman, A., & Patterson, D. (2009). [Roofline: an insightful visual performance model for multicore architectures](https://dl.acm.org/doi/pdf/10.1145/1498765.1498785).


## Model parallelism

![Model parallelism (source: https://huggingface.co/docs/transformers/v4.17.0/en/parallelism)](img/parallelism-gpipe-bubble.png)

-   [Model parallelism](https://huggingface.co/docs/transformers/v4.17.0/en/parallelism) - HuggingFace


## Computational complexity of transformers

-   kipply (2022). [Transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/).
-   Bahdanau, D. (2022). [The FLOPs calculus of language model training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).
-   Sanger, A. (2023). [Inference characteristics of Llama-2](https://cursor.sh/blog/llama-inference). 
-   Shenoy, V. & Kiely, P. (2023). [A guide to LLM inference and performance](https://www.baseten.co/blog/llm-transformer-inference-guide/).
-   Anthony, Q., Biderman, S., & Schoelkopf, H. (2023). [Transformer math 101](https://blog.eleuther.ai/transformer-math/).


## Efficient transformers

-   Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Ré, C. (2022). [FlashAttention: Fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135).
-   Pope, R. et al. (2022). [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102).
-   Dao, T. (2023). [FlashAttention-2: Faster attention with better parallelism and work partitioning](https://arxiv.org/abs/2307.08691).
-   Kim, S. et al. (2023). [Full stack optimization of transformer inference: A survey](https://arxiv.org/abs/2302.14017).
-   PyTorch. (2023). [Accelerating generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/).
-   Nvidia. (2023). [Mastering LLM techniques: Inference optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization).
-   Weng, L. (2023). [Large transformer model inference optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/).
-   Kwon, W. et al. (2023). [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180).
-   Zhang, L. (2023). [Dissecting the runtime performance of the training, fine-tuning, and inference of large language models](https://arxiv.org/abs/2311.03687).
-   Dettmers, T. (2023). [Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#Raw_Performance_Ranking_of_GPUs).
-   Fu, Y. (2023). [Towards 100x speedup: Full stack transformer inference optimization](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c#5e3b9489c0204f8c8d70d014a9e88b28).
-   Fu, Y. (2024). [Challenges in deploying long-context transformers: A theoretical peak performance analysis](https://arxiv.org/abs/2405.08944).
-   Fu, Y. et al. (2024). [Data engineering for scaling language models to 128K context](https://arxiv.org/abs/2402.10171).
-   Kwon, W. et al. (2023). [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180). ([vLLM](https://github.com/vllm-project/vllm))
-   Sun, Y. et al. (2024). [Learning to (learn at test time): RNNs with expressive hidden states](https://arxiv.org/abs/2407.04620).
-   Shah, J. et al. (2024). [FlashAttention-3: Fast and accurate attention with asynchrony and low-precision](https://arxiv.org/abs/2407.08608).


## Accelerators

-   Cerebras. (2020). [Fast stencil-code computation on a wafer-scale processor](https://arxiv.org/abs/2010.03660).
-   Cerebras. (2022). [Wafer-scale fast fourier transforms](https://arxiv.org/abs/2209.15040).
-   Cerebras. (2023). [Cerebras architecture deep dive: First look inside the hardware/software co-design for deep learning](https://8968533.fs1.hubspotusercontent-na1.net/hubfs/8968533/IEEE%20Micro%202023-03%20Hot%20Chips%2034%20Cerebras%20Architecture%20Deep%20Dive.pdf).
-   Cerebras. (2023). [Training giant neural networks using weight streaming on cerebras wafer-scale systems](https://8968533.fs1.hubspotusercontent-na1.net/hubfs/8968533/Virtual%20Booth%20Docs/CS%20Weight%20Streaming%20White%20Paper.pdf)
-   SambaNova. (2024). [SambaNova SN40L: Scaling the AI memory wall with dataflow and composition of experts](https://arxiv.org/abs/2405.07518).
-   Th&uuml;ning, M. (2024). [Attention in SRAM on Tenstorrent Grayskull](https://arxiv.org/abs/2407.13885).


## Conclusion

TODO


--------

-   Up next: [Misc](misc.md)
-   Previous: [Natural language](natural-language.md)

