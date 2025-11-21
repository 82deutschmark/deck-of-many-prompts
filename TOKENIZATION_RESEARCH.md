# Tokenization Research for Modern LLMs (2025)

## Overview
This document summarizes the latest tokenization approaches used by major language models as of November 2025.

## Major Model Families (2025)

### OpenAI Models

#### GPT-5 (Released August 7, 2025) ⭐ LATEST
- **Tokenizer**: o200k_base encoding
- **Library**: tiktoken
- **Context Window**: 400K tokens (272K input + 128K output including reasoning tokens)
- **Models**: Three sizes available
  - `gpt-5`: $1.25/M input, $10/M output
  - `gpt-5-mini`: $0.25/M input, $2/M output
  - `gpt-5-nano`: $0.05/M input, $0.40/M output
- **Token Caching**: 90% discount on cached tokens (within previous few minutes)
- **Details**: Best AI system yet with SOTA performance across coding, math, writing, health, and visual perception

#### GPT-5.1 (Released November 12, 2025) ⭐ LATEST
- **Tokenizer**: o200k_base encoding
- **Updates**: Warmer, more capable models with customizable tone and style
- **Variants**:
  - GPT-5.1-Codex-Max (Nov 19, 2025) - faster, more intelligent agentic coding
  - GPT-5-Codex-Mini (Nov 10, 2025) - token-efficient with compaction across context windows

#### o200k_harmony (2025)
- **New Encoding**: For open-weight Harmony models (gpt-oss-20b, gpt-oss-120b)
- **Released**: Alongside GPT-5
- **Purpose**: Open-source model tokenization

#### o1, o3, o3-mini Models (December 2024 - January 2025)
- **o3-mini**: Released January 31, 2025 (three variants: low, medium, high)
- **o1-pro**: Released December 5, 2024 (ChatGPT Pro)
- **Context Window**: o3 expanded to 200K tokens
- **Tokenizer**: o200k_base encoding

#### GPT-4o (Released 2024)
- **Tokenizer**: o200k_base encoding (first to use this encoding)
- **Library**: tiktoken
- **Xenova Support**: `Xenova/gpt-4o`

#### GPT-4
- **Tokenizer**: cl100k_base encoding
- **Library**: tiktoken
- **Xenova Support**: `Xenova/gpt-4`

### Anthropic Claude Models

#### Claude Sonnet 4.5 (Released September 29, 2025) ⭐ LATEST
- **Context Window**: 200K (expandable to 1M tokens)
- **Performance**: 77.2% on SWE-bench Verified (leading coding model)
- **Output Tokens**: Up to 64K
- **Extended Context Pricing**: $6/M input beyond 200K (doubled from $3/M)
- **Prompt Caching**: 5-min TTL (standard) or 1-hour TTL (extended, additional cost)

#### Claude Opus 4 and Claude Sonnet 4 (Released May 2025)
- **Opus 4**:
  - Context Window: 200K tokens
  - Output: 32K max output
  - Pricing: $15/M input, $75/M output
- **Sonnet 4**:
  - Context Window: 200K (expandable to 1M tokens via API since August 12, 2025)
  - Output: 64K max output
  - Pricing: $3/M input, $15/M output (standard), $6/M input beyond 200K
- **Xenova Support**: `Xenova/claude-tokenizer` (Note: May not be accurate for Claude 4+)
- **Important**: Anthropic has not released official tokenizer for Claude 4 series

#### Claude 3.7 Sonnet (February 2025)
- **Features**: Hybrid AI reasoning model
- **Context Window**: 200,000 tokens
- **Output Tokens**: Up to 32,000 tokens

#### Claude 3.5 Sonnet (June 2024)
- **Context Window**: 200K tokens
- **Output Tokens**: Initially 4,096, expanded to 8,192, then 64K
- **Pricing**: $3/M input, $15/M output

### Meta Llama Models

#### Llama 4 (Released April 2025) ⭐ LATEST
- **Architecture**: Mixture-of-Experts (MoE), natively multimodal
- **Tokenizer**: TikToken-based (continued from Llama 3)
- **Training Data**: 30+ trillion tokens (2x Llama 3)
- **Multilingual**: 200 languages, 100+ with 1B+ tokens each (10x more than Llama 3)
- **Context Windows**:
  - **Llama 4 Scout**: 10 million tokens (industry-leading)
  - **Llama 4 Maverick**: 512,000 tokens
- **Token Estimation**: ~4 characters per token
- **Multimodality**: Early fusion of text and vision tokens
- **Xenova Support**: Likely `Xenova/Meta-Llama-4-Tokenizer` (check HuggingFace)

#### Llama 3.x (2024)
- **Major Change**: Switched from SentencePiece to tiktoken
- **Tokenizer**: TikToken-based with 100k base + 28k non-English tokens
- **Context Length**: 128K tokens
- **Xenova Support**: `Xenova/Meta-Llama-3.1-Tokenizer`, `Xenova/llama3-tokenizer`

### Google Gemini Models

#### Gemini 3 (Released November 18, 2025) ⭐ LATEST
- **Tokenizer**: SentencePiece (same as Gemini 2.x and Gemma)
- **Description**: Google's "most intelligent model"
- **Context Window**: 1M input tokens, 64K output tokens
- **Knowledge Cutoff**: January 2025
- **Token Ratio**: ~4 characters per token, 100 tokens ≈ 60-80 English words
- **Pricing**: $2/$12 per M tokens (up to 200K), $4/$18 per M (beyond 200K)
- **Features**: Generative UI responses, vibe-coding
- **Xenova Support**: `Xenova/gemma-tokenizer`, `Xenova/gemma2-tokenizer`

#### Gemini 2.5 Pro (2025)
- **Tokenizer**: SentencePiece (shared with Gemma)
- **Context Window**: 1M tokens
- **Vocabulary**: ~250,000 tokens
- **Image Tokenization**:
  - Images ≤384px: 258 tokens
  - Larger images: 258 tokens per 768x768 tile
- **Video**: 1-min 720p ≈ 10-15K tokens

#### Gemini 2.0 (December 2024)
- **Tokenizer**: SentencePiece
- **Context Window**: Up to 1M tokens
- **Note**: Character-based pricing models being retired, all using token-based

#### Gemini 1.5
- **Context Windows**: 128K standard, tested up to 10M tokens
- **Xenova Support**: `Xenova/gemini-nano`

### xAI Grok Models

#### Grok 4.1 (Released November 17, 2025) ⭐ LATEST
- **Updates**: Improved reasoning, multimodal understanding, personality, reduced hallucinations
- **Context Window**: 2M tokens
- **Knowledge Cutoff**: November 2024

#### Grok 4 Fast (Released September 2025)
- **Context Window**: 2M tokens (110-160 hours of conversation)
- **Architecture**: Unified reasoning and non-reasoning modes
- **Efficiency**: Uses 40% fewer thinking tokens than Grok 4

#### Grok 4 and 4 Heavy (Released July 9, 2025)
- **Context Window**: 2M tokens
- **Tokenizer**: Details not publicly disclosed
- **Knowledge Cutoff**: November 2024

### Alibaba Qwen Models

#### Qwen 3 (2025)
- **Tokenizer**: Byte-level BPE (same as Qwen 2.5)
- **Integration**: DeepSeek-R1-0528-Qwen3-8B uses Qwen3 architecture with DeepSeek tokenizer
- **HuggingFace**: `Qwen/Qwen3-*` models
- **Xenova Support**: Check `onnx-community` for ONNX conversions

#### Qwen 2.5 (2024)
- **Tokenizer**: Byte-level Byte-Pair-Encoding (BPE)
- **Models**: 0.5B to 72B parameters
- **Xenova Support**: `Xenova/Qwen1.5-0.5B-Chat`, `onnx-community/Qwen2.5-*`
- **Integration**: Qwen2 supported in Transformers.js

### DeepSeek Models

#### DeepSeek-V3.1 (Released August 2025)
- **Tokenizer**: Updated from V3 with additional `</think>` token
- **Special Tokens**: `<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`, `<｜end▁of▁sentence｜>`
- **Chat Template**: Updated
- **HuggingFace**: `deepseek-ai/DeepSeek-V3.1`

#### DeepSeek-V3 (December 2024)
- **Architecture**: MoE (671B total, 37B activated per token)
- **Tokenizer**: Available on HuggingFace
- **Usage**: `transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")`

#### DeepSeek-R1 Series (2025)
- **Integration**: DeepSeek-R1-0528-Qwen3-8B combines Qwen3 architecture with DeepSeek tokenizer

### Microsoft Phi Models

#### Phi-4 (December 2024)
- **Tokenizer**: OpenAI's tiktoken
- **Vocabulary**: 100,352 tokens (padded to multiple of 64)
- **Format**: ChatML format (changed from Phi-3's Llama format)
- **Features**: Better multilingual tokenization
- **HuggingFace**: `microsoft/phi-4`
- **Dummy Tokens**: Unused tokens reserved for future functionality

## Tokenization Libraries (2025)

### tiktoken (OpenAI)
- **Purpose**: Fast BPE tokenizer for OpenAI models
- **Encodings Available**:
  - `r50k_base`: GPT-3, GPT-2
  - `p50k_base`: Code models
  - `p50k_edit`: Edit models
  - `cl100k_base`: GPT-4, GPT-3.5-turbo
  - `o200k_base`: GPT-4o, GPT-5, o1, o3, o4-mini ⭐
  - `o200k_harmony`: GPT-OSS open-weight models ⭐ NEW
- **Usage**: Pre-tokenize to estimate API costs
- **Repository**: https://github.com/openai/tiktoken

### Transformers.js / Xenova
- **Version**: v3 (October 2024)
- **Models**: 120+ architectures, 1200+ pre-converted models
- **Package**: `@huggingface/transformers` (moved from `@xenova/transformers`)
- **Platform**: Browser-native with WebGPU support
- **2024-2025 Additions**: Phi-3, Phi-4, Gemma 2, Qwen2, Llama 3.x, and more

### SentencePiece
- **Used by**: Gemini (all versions), Gemma, older Llama models (pre-Llama 3)
- **Algorithms**: BPE and unigram language models
- **Vocabulary**: ~250,000 tokens (Gemini/Gemma)
- **Note**: Llama 4 continues using tiktoken (not SentencePiece)

### gpt-tokenizer (JavaScript/TypeScript)
- **Support**: GPT-5, GPT-4o, o1, o3, o4, and all OpenAI models
- **Platform**: Fastest, smallest footprint for JavaScript environments
- **Repository**: https://github.com/niieani/gpt-tokenizer
- **NPM**: `gpt-tokenizer`

## Key Trends (2025)

1. **Massive Context Window Expansion**: 32K → 128K → 1M → 2M → **10M tokens** (Llama 4 Scout)
2. **Tiktoken Dominance**: o200k_base now standard for GPT-5, maintained for Llama 4, adopted by Phi-4
3. **Multimodal Tokenization**: Native fusion of text, vision, and audio tokens (Llama 4)
4. **Output Token Increases**: Now 32K → 64K → 128K max output
5. **Token Caching**: Aggressive caching (90% discounts, 5-min to 1-hour TTL options)
6. **Open-Weight Models**: o200k_harmony encoding for GPT-OSS models
7. **Training Data Scale**: 30+ trillion tokens (Llama 4)
8. **Token Efficiency**: Models optimized for fewer thinking tokens (Grok 4 Fast: 40% reduction)

## Context Window Leaderboard (November 2025)

1. **Llama 4 Scout**: 10,000,000 tokens 🥇
2. **Grok 4/4 Fast/4.1**: 2,000,000 tokens 🥈
3. **Claude Sonnet 4/4.5**: 1,000,000 tokens 🥉
4. **Gemini 3**: 1,000,000 tokens 🥉
5. **Llama 4 Maverick**: 512,000 tokens
6. **GPT-5**: 400,000 tokens (272K input + 128K output)
7. **Claude Opus 4**: 200,000 tokens
8. **o3**: 200,000 tokens

## Recommendations for Project Updates

### Missing from Current Project (as of Nov 2025)

**Critical Additions Needed:**
- `gpt-5` / `gpt-5-mini` / `gpt-5-nano` (Xenova/gpt-4o or wait for Xenova/gpt-5)
- `gpt-5.1` (same tokenizer as gpt-5)
- `claude-opus-4` / `claude-sonnet-4` / `claude-sonnet-4.5`
- `gemini-3` (use Xenova/gemma2-tokenizer)
- `llama-4-scout` / `llama-4-maverick` (check for Xenova/Meta-Llama-4-Tokenizer)
- `grok-4` / `grok-4-fast` / `grok-4.1`
- `qwen-3` (check onnx-community)
- `deepseek-v3.1`

### Suggested Tokenizer List Update

```python
world['tokenizers'] = {
    # Latest OpenAI (2025)
    'gpt-5': 'Xenova/gpt-4o',  # Uses o200k_base
    'gpt-5-mini': 'Xenova/gpt-4o',
    'gpt-5.1': 'Xenova/gpt-4o',
    'gpt-4o': 'Xenova/gpt-4o',
    'o3': 'Xenova/gpt-4o',
    'o1': 'Xenova/gpt-4o',

    # Latest Claude (2025)
    'claude-sonnet-4.5': 'Xenova/claude-tokenizer',  # Not official
    'claude-sonnet-4': 'Xenova/claude-tokenizer',
    'claude-opus-4': 'Xenova/claude-tokenizer',
    'claude': 'Xenova/claude-tokenizer',

    # Latest Gemini (2025)
    'gemini-3': 'Xenova/gemma2-tokenizer',
    'gemini-2.5': 'Xenova/gemma2-tokenizer',
    'gemini-2.0': 'Xenova/gemma2-tokenizer',
    'gemini-nano': 'Xenova/gemini-nano',

    # Latest Llama (2025)
    'llama-4': 'Xenova/Meta-Llama-3.1-Tokenizer',  # Check for Llama-4 when available
    'llama-3.3': 'Xenova/Meta-Llama-3.1-Tokenizer',
    'llama-3.1': 'Xenova/Meta-Llama-3.1-Tokenizer',

    # Grok (2025) - tokenizer not publicly available via Xenova yet
    # 'grok-4': 'TBD',
    # 'grok-4-fast': 'TBD',

    # Other 2025 models
    'qwen-3': 'Xenova/Qwen1.5-0.5B-Chat',  # Check onnx-community
    'qwen-2.5': 'Xenova/Qwen1.5-0.5B-Chat',
    'deepseek-v3.1': 'deepseek-ai/DeepSeek-V3.1',
    'phi-4': 'microsoft/phi-4',

    # ... keep existing older models ...
}
```

### Important Notes (November 2025)

- **GPT-5 uses o200k_base** (NOT o300k_base - that doesn't exist)
- **Claude 4+ tokenizer** is not officially released by Anthropic
- **Gemini 3 uses same tokenizer** as Gemini 2.x and Gemma (SentencePiece)
- **Llama 4 continues tiktoken** (not reverting to SentencePiece)
- **Grok tokenizer** details not publicly disclosed
- **Check onnx-community** on HuggingFace for latest ONNX conversions
- **Monitor Xenova's profile** for new model conversions

## References

- OpenAI tiktoken: https://github.com/openai/tiktoken
- OpenAI GPT-5: https://openai.com/index/introducing-gpt-5/
- Transformers.js: https://github.com/huggingface/transformers.js
- Xenova Models: https://huggingface.co/Xenova
- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers
- gpt-tokenizer: https://github.com/niieani/gpt-tokenizer
- Claude 4: https://www.anthropic.com/news/claude-4
- Gemini 3: https://blog.google/products/gemini/gemini-3/
- Llama 4: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Grok 4: https://x.ai/news/grok-4-fast

## Document History

- **Last Updated**: November 21, 2025
- **Research Date**: November 21, 2025
- **Coverage**: Models released through November 2025
- **Previous Version**: Covered through early 2025 (outdated)

---

**Note**: This research is current as of November 21, 2025. AI models and tokenizers evolve rapidly. Check official sources and HuggingFace for the latest updates.
