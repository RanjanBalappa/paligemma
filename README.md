# PaliGemma VLLM - Pytorch

+------------------+     +------------------+
|    Input Image   |     |    Input Text    |
+--------+---------+     +--------+---------+
         |                       |
         v                       v
+--------+---------+     +--------+---------+
| Vision Encoder   |     | Text Tokenizer/  |
| (ViT)            |     | Embedding        |
+--------+---------+     +--------+---------+
         |                       |
         v                       v
+------------------+     +------------------+
| Image Tokens     |     | Text Tokens      |
| (e.g., L_img x D)|     | (e.g., L_text x D)|
+--------+---------+     +--------+---------+
         |           (Concatenate/Prepend)
         v
+---------------------------------+
| Multi-modal Transformer Decoder |
| (Gemma-like LLM with            |
|  Cross-Attention)               |
+---------------------------------+
         |
         v
+---------------------------------+
|   Generated Text/Answer         |
+---------------------------------+
