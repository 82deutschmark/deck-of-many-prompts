# Tokenization Research for Modern LLMs (2024-2025)

## Overview
This document summarizes the latest tokenization approaches used by major language models as of late 2024 and early 2025.

## Major Model Families

### OpenAI Models

#### GPT-4o (Released 2024)
- **Tokenizer**: o200k_base encoding (NEW - different from GPT-4)
- **Library**: tiktoken
- **Details**: GPT-4o introduced a new tokenization algorithm optimized for efficiency
- **Xenova Support**: `Xenova/gpt-4o`

#### GPT-4
- **Tokenizer**: cl100k_base encoding
- **Library**: tiktoken
- **Xenova Support**: `Xenova/gpt-4`

#### o1, o3, o3-mini Models (December 2024 - January 2025)
- **o3-mini**: Released January 31, 2025 (three variants: low, medium, high)
- **Context Window**: o3 expanded to 200K tokens (up from GPT-4's 128K)
- **Performance**: Significant improvements on AIME-2024 (96.7% vs 83.3%) and GPQA Diamond (87.7% vs 78%)
- **Tokenizer**: Likely uses same tokenization as GPT-4o (o200k_base)

### Anthropic Claude Models

#### Claude 3.5 Sonnet (June 2024)
- **Context Window**: 200K tokens
- **Output Tokens**: Initially 4,096, expanded to 8,192 (with beta header), latest supports 64K output tokens
- **Pricing**: $3 per million input tokens, $15 per million output tokens
- **Xenova Support**: `Xenova/claude-tokenizer` (Note: May not be accurate for Claude 3+)
- **Important**: Anthropic has not released the v3 tokenizer officially

#### Claude 3.7 Sonnet (February 2025)
- **Features**: Hybrid AI reasoning model with rapid or step-by-step reasoning
- **Context Window**: 200,000 tokens
- **Output Tokens**: Up to 32,000 tokens

#### Claude Opus 4.1 (August 2025)
- **Focus**: Agentic tasks, real-world coding, reasoning
- **Context Window**: 200,000 tokens
- **Output Tokens**: Up to 32,000 tokens

### Meta Llama Models

#### Llama 3, 3.1, 3.2, 3.3
- **Major Change**: Switched from SentencePiece (Llama 2) to tiktoken
- **Tokenizer**: TikToken-based with 100k base tokens + 28k non-English tokens
- **Llama 3.2**: Released September 25, 2024 (1B and 3B models)
- **Llama 3.3**: Released December 4, 2024 (70B model)
- **Context Length**: 128K tokens
- **Xenova Support**: `Xenova/Meta-Llama-3.1-Tokenizer`, `Xenova/llama3-tokenizer`
- **Compatibility**: Prompts designed for Llama 3 work unchanged across 3.1, 3.2, 3.3

### Google Gemini Models

#### Gemini 1.5 and 2.0 (December 2024)
- **Tokenizer**: SentencePiece (BPE + unigram algorithms)
- **Shared with**: Gemma models (same tokenizer)
- **Vocabulary**: ~250,000 tokens
- **Token Ratio**: ~4 characters per token, 100 tokens ≈ 60-80 English words
- **Context Windows**:
  - Gemini 1.0: 32,000 tokens
  - Gemini 1.5 Pro: 128,000 tokens standard, up to 1M tokens
  - Tested: 10M token window
- **Image Tokenization** (Gemini 2.0):
  - Images ≤384px: 258 tokens
  - Larger images: 258 tokens per 768x768 tile
- **Xenova Support**: `Xenova/gemini-nano`, `Xenova/gemma-tokenizer`, `Xenova/gemma2-tokenizer`
- **Note**: Use Gemma tokenizer for Gemini models (same tokenization scheme)

#### Gemini 2.5 Flash
- **Latest Release**: Advanced features for the agentic era
- **Pricing**: Token-based (models using character-based pricing are being retired)

### Alibaba Qwen Models

#### Qwen 2.5 (2024)
- **Tokenizer**: Byte-level Byte-Pair-Encoding (BPE)
- **Models**: 0.5B to 72B parameters
- **HuggingFace**: Available in transformers library
- **Xenova Support**: `Xenova/Qwen1.5-0.5B-Chat`, ONNX models via `onnx-community/Qwen2.5-*`
- **Integration**: Qwen2 is supported in Transformers.js

### DeepSeek Models

#### DeepSeek-V3 (December 2024)
- **Architecture**: Mixture-of-Experts (MoE)
- **Parameters**: 671B total, 37B activated per token
- **Tokenizer**: Available on HuggingFace (`deepseek-ai/DeepSeek-V3`)
- **Usage**: Works with `transformers.AutoTokenizer`

#### DeepSeek-V3.1
- **Update**: Introduces additional `</think>` token
- **Special Tokens**: `<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`, `<｜end▁of▁sentence｜>`
- **Chat Template**: Updated from V3

### Microsoft Phi Models

#### Phi-4 (December 2024)
- **Tokenizer**: OpenAI's tiktoken
- **Vocabulary**: 100,352 tokens (padded to multiple of 64)
- **Format**: ChatML format (different from Phi-3's Llama format)
- **Features**: Better tokenization performance for multiple languages
- **HuggingFace**: `microsoft/phi-4`
- **Note**: Includes unused "dummy" tokens for future functionality

## Tokenization Libraries

### Transformers.js / Xenova
- **Version**: v3 (October 2024)
- **Models**: 120+ architectures, 1200+ pre-converted models
- **New Package**: `@huggingface/transformers` (moved from `@xenova/transformers`)
- **Platform**: Runs in browser with WebGPU support
- **Notable 2024 Additions**: Phi-3, Gemma & Gemma 2, Qwen2, and many others

### tiktoken (OpenAI)
- **Purpose**: Fast BPE tokenizer for OpenAI models
- **Encodings**:
  - `cl100k_base`: GPT-4, GPT-3.5-turbo
  - `o200k_base`: GPT-4o (NEW)
- **Usage**: Pre-tokenize to estimate API costs (billing is per token)

### SentencePiece
- **Used by**: Gemini, Gemma, older Llama models (pre-Llama 3)
- **Algorithms**: BPE and unigram language models
- **Note**: Llama 3+ switched away from SentencePiece to tiktoken

## Key Trends

1. **Context Window Expansion**: Models moving from 32K → 128K → 200K → 1M+ tokens
2. **Tiktoken Adoption**: More models switching to tiktoken (Llama 3, Phi-4)
3. **Multimodal Tokenization**: Improved handling of images and audio
4. **Output Token Increases**: From 4K → 8K → 32K+ output tokens
5. **Vocabulary Optimization**: Strategic padding (e.g., Phi-4 to multiples of 64)
6. **Browser-Native**: Transformers.js enabling client-side tokenization

## Recommendations for Project Updates

### Current Tokenizers in Project
The project currently includes many tokenizers via Xenova but may be missing:
- Explicit o1/o3 tokenizers (though GPT-4o tokenizer likely compatible)
- DeepSeek-V3/V3.1 tokenizers
- Qwen 2.5 tokenizers
- Phi-4 tokenizer
- Latest Gemini 2.0/2.5 identifiers

### Suggested Additions
Consider adding entries for:
- `gpt-o1` / `gpt-o3` (using Xenova/gpt-4o)
- `qwen-2.5` (using Xenova/Qwen1.5 or ONNX community models)
- `phi-4` (using microsoft/phi-4)
- `deepseek-v3` (using deepseek-ai/DeepSeek-V3)
- `gemini-2.0` (using Xenova/gemma2-tokenizer)

### Important Notes
- Claude tokenizer may not be accurate for Claude 3.5+
- Gemini models should use Gemma tokenizers (same tokenization)
- Llama 3+ uses tiktoken, not SentencePiece
- Check Xenova/HuggingFace regularly for ONNX-converted models

## References
- OpenAI tiktoken: https://github.com/openai/tiktoken
- Transformers.js: https://github.com/huggingface/transformers.js
- Xenova Models: https://huggingface.co/Xenova
- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers

## Last Updated
November 21, 2025

Based on web research conducted on November 21, 2025, covering models and tokenizers released through early 2025.
