# 📝 AIRA NLP & Tokenization Engine (`ai/nlp`)

[![Language: Aira](https://img.shields.io/badge/Language-Aira%20100%25-ff007f.svg)](https://github.com/adameehanairagroup/Airalang)
[![Domain: ai/nlp](https://img.shields.io/badge/Domain-Tokenizer%20%26%20Embeddings-00e5ff.svg)](https://github.com/adameehanairagroup/Airalang)

The `ai/nlp` package implements text tokenization, byte-pair encoding (BPE), vocabulary mapping, and dense semantic embedding search.

```aira
import "ai/nlp";

let tokenizer = new BPETokenizer();
tokenizer.load_vocab("vocab.json", "merges.txt");

let token_ids = tokenizer.encode("Sovereign AI Engineering in AIRA");
say token_ids; // [1284, 9942, 381, 2901]

let text = tokenizer.decode(token_ids);
say text;
```
