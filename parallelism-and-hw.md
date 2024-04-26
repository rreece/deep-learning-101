# Parallelism and hardware


### Contents

1.  [Introduction](#introduction)
2.  [Model parallelism](#model-parallelism)
3.  [Misc](#misc)
4.  [Conclusion](#conclusion)


## Introduction

-   Single Instruction/Multiple Data (SIMD) and GPUs
-   [FLOPs vs FMACs](https://medium.com/@pashashaik/a-guide-to-hand-calculating-flops-and-macs-fa5221ce5ccc)
-   [Data parallel vs model parallel vs tensor parallel](https://colossalai.org/docs/concepts/paradigms_of_parallelism/)
-   SRAM vs DRAM

![Examples of GPU memory usage. source: https://arxiv.org/abs/2403.03507](img/galore-memory-usage.png)

-   Hooker, S. (2020). [The hardware lottery](https://arxiv.org/abs/2009.06489).
-   He, H. (2022). [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html).
-   Geiping, J. & Goldstein, T. (2022). [Cramming: Training a language model on a single GPU in one day](https://arxiv.org/abs/2212.14034).


## Model parallelism

![Model parallelism (source: https://huggingface.co/docs/transformers/v4.17.0/en/parallelism)](img/parallelism-gpipe-bubble.png)

-   [Model parallelism](https://huggingface.co/docs/transformers/v4.17.0/en/parallelism) - HuggingFace


## Misc

-   Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Ré, C. (2022). [FlashAttention: Fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135).


## Conclusion

TODO


--------

-   Up next: [Misc](misc.md)
-   Previous: [Natural language](natural-language.md)

