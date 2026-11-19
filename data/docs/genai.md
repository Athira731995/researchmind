# Generative AI Basics

Transformers rely on self-attention, which lets every token weigh the relevance of every other token in a sequence. This replaced recurrence and enables parallel training.

Retrieval-augmented generation (RAG) grounds a language model by retrieving relevant documents from an external knowledge base and placing them in the prompt. This reduces hallucination and allows knowledge updates without retraining.

LoRA (low-rank adaptation) fine-tunes a model by freezing the original weights and training small low-rank matrices added to selected layers. This cuts trainable parameters by over 99 percent.

Embeddings map text to dense vectors so semantically similar text lies close together. Vector databases such as FAISS use nearest-neighbour search to find similar vectors quickly.

Agents combine a language model with tools. The model decides which tool to call, observes the result, and repeats until it can answer.