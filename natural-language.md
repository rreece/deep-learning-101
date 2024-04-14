# Natural language

## Introduction

TODO


## Tokenization

Tokenization is the process of taking a sequence of text
and breaking it into units called *tokens*.
You can think of tokens as being words,
but in general they can be parts of words.

Tokens are generally then converted to "token IDs" that
are integer encodings of the tokens.

Example:

```
text = "Hello world! This is tokenization."
tokens = ["<start>", ""Hello", " ", "world", "!", " ", "This", " ", "is", " ", "token", "iz", "ation", ".", "<end>"]
token_ids = [0, 123, 2223, 10, 335, 556, 10, ... ]
```

-   Tokenization is basically a map from word parts to integers.
-   It is important to note that tokenization is dependent on a *vocabulary* used to make the map.
-   So note that a certain tokenization may not support any language. The language needs to map the vocabulary.

See also:

-   Tutorial video by Andrej Karpathy: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)


## word2vec

After tokenization, the next step in a language model is to *embed* the tokens,
which is a map from the token IDs to a vector in some large space,
with dimension called the `embedding_size`.

After the embedding parameters are trained end-to-end with a model,
remarkably, you can give some semantic interpretations to some basis
vectors in the embedding space.  Famously, for example

$$ \vec{\mathrm{king}} - \vec{\mathrm{man}} + \vec{\mathrm{woman}} \approx \vec{\mathrm{queen}} $$

![word2vec visualization (source: https://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html/).](img/word2vec-viz.png)


## seq2seq

-   RNNs
-   LSTMs
-   Watershed moment in NLP with deep learning


## Transformer

TODO


## BERT

TODO


## T5

-   Raffel, C. et al. (2019). [Exploring the limits of transfer learning with a unified text-to-text transformer](https://arxiv.org/abs/1910.10683).


## Conclusion

![Evolutionary tree of LLMs (source: [2304.13712](https://arxiv.org/abs/2304.13712)).](img/evolutionary-tree-of-LLMs.png)

-   Yang, J. et al. (2023). [Harnessing the power of LLMs in practice: A survey on ChatGPT and beyond](https://arxiv.org/abs/2304.13712).
-   Raschka, S. (2023). [Understanding large language models](https://magazine.sebastianraschka.com/p/understanding-large-language-models).

--------

-   Up next: [Parallelism and hardware](parallelism-and-hw.md)
-   Previous: [Computer vision](computer-vision.md)


