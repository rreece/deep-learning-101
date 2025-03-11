# Parallelism and hardware


### Contents

1.  [Introduction](#introduction)
2.  [Performance and bandwidth](#performance-and-bandwidth)
3.  [Model parallelism](#model-parallelism)
4.  [Computational complexity of transformers](#computational-complexity-of-transformers)
5.  [Efficient transformers: Inference optimizations](#efficient-transformers-inference-optimizations)
5.  [Efficient transformers: Architecture modifications](#efficient-transformers-architecture-modifications)
6.  [Kernel programming](#kernel-programming)
7.  [Accelerators - Big Tech](#accelerators---big-tech)
8.  [Accelerators - Startups](#accelerators---startups)
9.  [Scaling](#scaling)
10. [Conclusion](#conclusion)


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
-   Chen, L. (2023). [Dissecting batching effects in GPT inference](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/).
-   Chng, P. (2024). [The naive roofline model in performance modeling](https://peterchng.com/blog/2024/08/29/the-naive-roofline-model-in-performance-modeling/).
-   Kao, S.C. et al. (2022). FRAME: Fast Roofline Analytical Modeling and Estimation. https://github.com/maestro-project/frame
-   Yuan, Z. et al. (2024). LLM inference unveiled: Survey and roofline model insights. https://arxiv.org/abs/2402.16363
    -   <https://github.com/hahnyuan/LLM-Viewer>
    -   <http://llm-viewer.com>


## Model parallelism

![Model parallelism (source: https://huggingface.co/docs/transformers/v4.17.0/en/parallelism)](img/parallelism-gpipe-bubble.png)

-   [Model parallelism](https://huggingface.co/docs/transformers/v4.17.0/en/parallelism) - HuggingFace
-   Pipeline parallelism
-   Tensor parallelism


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

-   Shazeer, N. (2019). [Fast transformer decoding: One write-head is all you need](https://arxiv.org/abs/1911.02150).  - MQA
-   Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). [Efficient transformers: A survey](https://arxiv.org/abs/2009.06732).
-   Leviathan, Y., Kalman, M., & Matias, Y. (2022). [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192).
-   Ainslie, J. (2023). [GQA: Training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245). - GQA


## Kernel programming

### Nvidia: CUDA

-   Nvidia. (2024). [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
-   Harris, M. (2017). Nvidia blog: [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/).
-   Boehm, S. (2022). [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM).
-   [github.com/ANSANJAY/KernelDev101](https://github.com/ANSANJAY/KernelDev101)
-   [github.com/cupy/cupy](https://github.com/cupy/cupy)
-   [github.com/NVIDIA/cuda-python](https://github.com/NVIDIA/cuda-python)
-   [github.com/NVIDIA/nvmath-python](https://github.com/NVIDIA/nvmath-python)

### AMD: ROCm

-   [ROCm Documentation](https://rocm.docs.amd.com/en/latest/)


## Accelerators - Big Tech

### Nvidia

Products: 

-    H100 and B100-based systems (DGX)

Whitepapers:

-   Nvidia. (2023). [GPU performance background user's guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html).

Media:

-   Dettmers, T. (2023). [Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/).

Research:

-   Volkov, V. (2016). [*Understanding Latency Hiding on GPUs*](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf). (PhD thesis)

### Amazon

Products: TODO

Whitepapers:

-    [AWS Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/)

Media: TODO

Research:

-    Zheng, H. et al. (2020). [Optimizing memory-access patterns for deep learning accelerators](https://arxiv.org/pdf/2002.12798).

### AMD

Products: TODO

-   [tinybox](https://tinygrad.org/#tinybox), red uses 6x 7900XTX

TODO: Whitepapers, Media, Research

### Apple

Products: M4

TODO: Whitepapers, Media, Research

### Intel (Habana)

Products: Gaudi 2

Whitepapers:

-   [Gaudi 2](https://habana.ai/products/gaudi2/)

TODO: Media, Research

### Meta

Products: MTIA v2

Whitepapers:

-   [Meta Training and Inference Accelerator (MTIA)](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)

TODO: Media, Research

## Accelerators - Startups

### Blaize

-   First AI chip startup to go public, in 2025 via SPAC

Products: TODO       
Whitepapers: TODO

Media:

-   Blaize. (2025). [S-1 filing with the SEC](https://ir.blaize.com/sec-filings/sec-filing/s-1/0001193125-25-008689). 2025/01/21.

Research: TODO

### Cerebras

-   Andrew Feldman, CEO

Products: CS-3

Whitepapers:

-   Cerebras. (2021). [The path to successful wafer-scale integration: The cerebras story](https://8968533.fs1.hubspotusercontent-na1.net/hubfs/8968533/IEEE%20Micro%202021-11%20Path%20to%20Wafer-Scale%20Integration.pdf).
-   Cerebras. (2023). [Cerebras architecture deep dive: First look inside the hardware/software co-design for deep learning](https://8968533.fs1.hubspotusercontent-na1.net/hubfs/8968533/IEEE%20Micro%202023-03%20Hot%20Chips%2034%20Cerebras%20Architecture%20Deep%20Dive.pdf).
-   Cerebras. (2023). [Training giant neural networks using weight streaming on cerebras wafer-scale systems](https://8968533.fs1.hubspotusercontent-na1.net/hubfs/8968533/Virtual%20Booth%20Docs/CS%20Weight%20Streaming%20White%20Paper.pdf).

Media:

-   Cerebras. (2024). [S-1 filing with the SEC](https://www.sec.gov/Archives/edgar/data/2021728/000162828024041596/cerebras-sx1.htm). 2024/09/30.

Research:

-   Cerebras. (2020). [Fast stencil-code computation on a wafer-scale processor](https://arxiv.org/abs/2010.03660).
-   Cerebras. (2022). [Wafer-scale fast fourier transforms](https://arxiv.org/abs/2209.15040).
-   Santos, K. et al. (2024). [Breaking the molecular dynamics timescale barrier using a wafer-scale system](https://arxiv.org/pdf/2405.07898).

### d-Matrix

-    Sid Sheth, CEO

Products: Corsair

Whitepapers:

-   d-Matrix. (2023). [d-Matrix Total Cost of Ownership White Paper](https://www.d-matrix.ai/wp-content/uploads/2023/09/d-Matrix-WhitePaper-Approved-w-cover.pdf).
-   d-Matrix. (2024). [d-Matrix Corsair redefines performance and efficiency for AI inference at scale](https://www.d-matrix.ai/wp-content/uploads/2024/11/d-Matrix-WhitePaper-Technical-FINAL.pdf).

Media:

-   EETimes. (2025). [d-Matrix targets fast LLM inference for 'real world scenarios’](https://www.eetimes.com/d-matrix-targets-fast-llm-inference-for-real-world-scenarios/). 2025/01/13.

Research: TODO

### Furiosa

-   June Paik, CEO

Products: TODO

Whitepapers:

-   Furiosa. (2024). [TCP: A Tensor Contraction Processor for AI workloads industrial product](https://ieeexplore.ieee.org/document/10609575).

Media:

-   Forbes. (2025). [Meta in talks to buy Korean AI chip startup founded by Samsung engineer](https://www.forbes.com/sites/johnkang/2025/02/11/meta-in-talks-to-buy-korean-ai-chip-startup-founded-by-samsung-engineer/). 2025/02/11.

Research: TODO

### Groq

-   Jonathan Ross, CEO

Products: TODO

Whitepapers:

-   Groq. (2020). [Think Fast: A Tensor Streaming Processor (TSP) for accelerating deep learning workloads](https://groq.com/wp-content/uploads/2020/06/ISCA-TSP.pdf).
-   Groq. (2022). [A software-defined tensor streaming multiprocessor for large-scale machine learning](https://wow.groq.com/wp-content/uploads/2023/05/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning-1.pdf).

Media:

-   Linley Group. (2020). [Groq rocks neural networks](https://groq.com/wp-content/uploads/2023/05/GROQ-ROCKS-NEURAL-NETWORKS.pdf).

Research:

-   Groq. (2024). [Optimized simulation methodology of warpage and localized stress hotspot prediction for assembly risk assessment](https://groq.com/wp-content/uploads/2024/06/Zhi_ECTC_Optimized-Simulation-Methodology-of-Warpage-and-Localized-Stress-Hotspot-Prediction-for-Assembly-Risk-Assessment_Mar1.pdf).

### Rebellions

-   Sunghyun Park, CEO

Products: TODO

Whitepapers:

-   Rebellions. (2024). [ATOM Architecture: Finding the Sweet Spot for GenAI](https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf).

TODO: Media, Research

### SambaNova

-   Rodrigo Liang, CEO

Products: SN40L-based systems

Whitepapers:

-   SambaNova. (2024). [SambaNova SN40L: Scaling the AI memory wall with dataflow and composition of experts](https://arxiv.org/abs/2405.07518).
-   SambaNova. (2024). [Why SambaNova's SN40L chip is the best for inference](https://sambanova.ai/blog/sn40l-chip-best-inference-solution).

TODO: Media, Research

### Tenstorrent

-   Jim Keller, CEO

Products: n150 and n300-based systems

Whitepapers:

-   Tenstorrent. (2024). [Onepager with Wormhole and Grayskull](https://cdn.sanity.io/files/jpb4ed5r/production/eefe2b42c1423b693c3a8eaf66c0015157e930a1.pdf).
-   Tenstorrent. (2024). [Wormhole Tensix Processor](https://cdn.sanity.io/files/jpb4ed5r/production/7c465635048500a3399800ab432f2e32d89d73a2.pdf).

Media: TODO

Research:

-   Th&uuml;ning, M. (2024). [Attention in SRAM on Tenstorrent Grayskull](https://arxiv.org/abs/2407.13885).
-   Brown, N. & Barton, R. (2024). [Accelerating stencils on the Tenstorrent Grayskull RISC-V accelerator](https://arxiv.org/abs/2409.18835).

### Others

-   Etched
    -   Gavin Uberti, CEO
-   Graphcore
    -   In July 2024, Softbank Group agreed to acquire Graphcore for around $500 million. The deal is under review by the UK's Business Department's investment security unit. [Wikipedia]
-   Lightmatter
-   MatX
    -   Reiner Pope, CEO
-   Taalas
    -   Ljubisa Bajic, CEO & Lejla Bajic, COO
-   Untether AI


## Scaling

-   Tazi, N. et al. (2025). [The ultra-scale playbook: Training LLMs on GPU clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook).


## Conclusion

TODO


--------

-   Up next: [Misc](misc.md)
-   Previous: [Natural language](natural-language.md)

