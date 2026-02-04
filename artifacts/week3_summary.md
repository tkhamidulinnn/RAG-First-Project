# Week 3 — Prompt Engineering Summary

## Setup
- LLM: Ollama via LangChain
- Model: gemma3:4b
- Templates: qa_strict, summary, reasoning
- Temperatures tested: [0.0, 0.2]

## Quick metrics (proxy)
 template  temperature  n  avg_latency_s  avg_answer_len  idk_rate  citation_rate
qa_strict          0.0  5         0.6948            43.0       1.0            0.0
qa_strict          0.2  5         0.4374            43.0       1.0            0.0
reasoning          0.0  5         5.8930          1210.2       0.0            0.8
reasoning          0.2  5         5.0254          1058.4       0.0            1.0
  summary          0.0  5         1.4898           267.0       0.0            0.0
  summary          0.2  5         1.3916           262.6       0.0            0.0
