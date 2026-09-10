# September 10 full-cohort benchmark

[Current README comparison](../../README.md#subtitle-translation-leaderboard) · [Historical runs](history.md)

<!-- smartfast-full-quality-benchmark:start -->
### Full 16-model benchmark and sampled translation quality, September 10, 2026

**10/16 models in the full cohort returned a structurally complete file.** All 16 jobs launched together with a 600-second cap each, using the same 1,340-cue English Matrix subtitle file for Hungarian translation through the actual 2.0.0 RC encrypted queued-content API. Each job used 100-cue batches, four parallel batches, configured temperature 0.3 (omitted for Luna because unsupported), unspecified reasoning and a stable opaque session. Each encrypted request explicitly set requestTimeout to 600 seconds, and the benchmark service used REQUEST_TIMEOUT=600. The external 600-second job cap bounded work even though the root recovery budget was 1,800 seconds. The observer allowed 605 seconds of read inactivity within the absolute remaining job window. This is the Bazarr+ wire format and authentication flow; the harness did not run inside Bazarr.

Completion requests now use the [official OpenRouter Python SDK](https://openrouter.ai/docs/client-sdks/python/overview), pinned to 1.1.133. The SDK constructs and dispatches requests with its retries disabled; the translator retains its response parsing, recovery rules and cancellation. The migration preserves reasoning fields missing from this SDK schema. Prices and authenticated endpoint metrics were captured before completion admission. Latency scoring retains the earlier milliseconds-to-seconds correction.

SmartFast kept its median +50% price cutoff, sparse-pool 3x limit, 20% estimated speed band and $1 input / $3 output per million token ceilings. Free variants retained exact zero price caps. Affinity can retain an eligible provider until three successful slow observations, so this is not an absolute-fastest guarantee on every request. This run includes the entire original 16-model cohort on commit f80a190 with the final SmartFast and SDK changes. Prior tables retain their original builds and conditions; they are not used to fill missing results in this run.

| Model | Result | Progress cues | Returned cues | Seconds | Requests | Service tokens | Cache-read tokens | Observed cost, USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemini-3.5-flash-lite` | Complete | 1340 | 1340 | 40.496 | 14 | 80964 | 0 | $0.120194 |
| `google/gemini-3.1-flash-lite` | Complete | 1340 | 1340 | 41.820 | 17 | 83575 | 0 | $0.083729 |
| `google/gemini-2.5-flash-lite` | Complete | 1340 | 1340 | 45.534 | 14 | 80679 | 0 | $0.021060 |
| `openai/gpt-5.6-luna` | Complete | 1340 | 1340 | 45.581 | 14 | 62512 | 0 | $0.046563 |
| `inception/mercury-2.5` | Complete | 1340 | 1340 | 50.856 | 16 | 121188 | 0 | $0.013707 |
| `meta/muse-spark-1.2-contributor` | Complete | 1340 | 1340 | 126.772 | 14 | 106138 | 0 | $0.017770 |
| `openai/gpt-4o-mini` | Complete | 1340 | 1340 | 131.258 | 14 | 68210 | 0 | $0.027982 |
| `meta/muse-spark-1.3-contributor` | Complete | 1340 | 1340 | 376.741 | 14 | 112989 | 2500 | $0.018895 |
| `z-ai/glm-5.3-flash` | Complete | 1340 | 1340 | 501.726 | 17 | 259301 | 512 | $0.117077* |
| `qwen/qwen3.7-flash` | Complete | 1340 | 1340 | 564.744 | 17 | 208704 | 5120 | $0.020150 |
| `dots-studio/dots-3-note-preview:free` | Failed | 0 | 0 | 6.684 | 0 | unknown | unknown | $0.000000 (no requests) |
| `inclusionai/ling-3.0-flash-fin:free` | Failed | 0 | 0 | 96.436 | 14 | unknown | unknown | unknown |
| `deepseek/deepseek-v4-flash-0731` | 600s cap | 400 | 0 | 600.007 | 9 | 164395 | 1024 | $0.081432* |
| `liquid/lfm-2.5-2.6b:free` | 600s cap | 700 | 0 | 600.094 | 91 | 165530 | 55744 | $0.000000* |
| `nvidia/nemotron-3.5-lightning:free` | 600s cap | 0 | 0 | 600.099 | 4 | unknown | unknown | unknown |
| `qwen/qwen3.8-flash` | 600s cap | 1000 | 0 | 600.125 | 15 | 166788 | 2688 | $0.069066* |

Observed response charges totaled **$0.63762675**, with **39 requests missing cost values**. An asterisk marks a lower bound. Service counters include finished roots, while observed usage includes returned responses from work still in progress. Missing usage is unknown, including for an interrupted free request. Cache counts are actual response usage, not inferred savings.

The observer forwarded upstream Retry-After headers unchanged. Receipts record their presence and normalized delay, while response status and body bytes pass through unchanged. Diagnostic failures are isolated from response delivery and recorded separately.

Progress cues count service-reported translated work. Returned cues describe result structure and may include original-text fallbacks for partial jobs. A capped running job can report progress without exposing result lines. Complete structure and preserved timestamps do not establish translation quality. HTTP 200 headers and keepalive bytes do not establish a completed response.

Raw results: [JSON](2026-09-10-smartfast-full-quality.json), [CSV](2026-09-10-smartfast-full-quality.csv), [price preflight](2026-09-10-smartfast-full-quality-prices.json), [authenticated endpoint metrics](2026-09-10-smartfast-full-quality-endpoints.json), [routing and stall diagnosis](2026-09-10-smartfast-full-quality-diagnosis.md). Prior numeric artifacts remain unchanged.

**Sampled Hungarian quality:** Blinded automated editorial review of a fixed purposive 180-cue sample: 12 evenly spaced 12-cue scene blocks plus 36 early non-overlapping thematic vocabulary cues. Frozen before live output. Meaning 50%, semantic completeness 20%, Hungarian fluency/register 20%, terminology 10%. Scores are subjective rubric judgments, not accuracy percentages; small gaps are not statistically established. Each model has one primary reviewer, with targeted contextual checks. No full-file semantic review or professional certification is implied. Fragment scores use only available sampled cues and are not comparable rankings.

| Model | Quality /100 | Reviewed /180 | Eligible complete file |
| --- | ---: | ---: | --- |
| `openai/gpt-5.6-luna` | 93.0 | 180 | Yes |
| `google/gemini-3.1-flash-lite` | 93.0 | 180 | Yes |
| `meta/muse-spark-1.3-contributor` | 91.0 | 180 | Yes |
| `meta/muse-spark-1.2-contributor` | 89.0 | 180 | Yes |
| `google/gemini-3.5-flash-lite` | 84.0 | 180 | Yes |
| `z-ai/glm-5.3-flash` | 81.0 | 180 | Yes |
| `openai/gpt-4o-mini` | 81.0 | 180 | Yes |
| `google/gemini-2.5-flash-lite` | 80.0 | 180 | Yes |
| `inception/mercury-2.5` | 70.0 | 180 | Yes |
| `qwen/qwen3.7-flash` | 58.0 | 180 | Yes |
| `deepseek/deepseek-v4-flash-0731` | 87.0 provisional | 50 | No |
| `qwen/qwen3.8-flash` | 79.0 provisional | 144 | No |
| `liquid/lfm-2.5-2.6b:free` | 29.0 provisional | 125 | No |
| `inclusionai/ling-3.0-flash-fin:free` | N/A | 0 | No |
| `nvidia/nemotron-3.5-lightning:free` | N/A | 0 | No |
| `dots-studio/dots-3-note-preview:free` | N/A | 0 | No |

Quality evidence: [findings and method](2026-09-10-smartfast-full-quality-quality.md), [raw assessments and mechanical checks](2026-09-10-smartfast-full-quality-quality.json), [numeric CSV](2026-09-10-smartfast-full-quality-quality.csv). Original subtitle and translated dialogue are excluded from public artifacts.
**Quality leaders among completed files:** Luna and Gemini 3.1 Flash Lite tied at 93/100; Muse Spark 1.3 followed at 91/100. Luna took 45.581 seconds and $0.046563, Gemini 3.1 took 41.820 seconds and $0.083729, and Muse 1.3 took 376.741 seconds and $0.018895. These are subjective sampled scores, and a two-point gap is not a proven quality difference. For this run, Luna offers the strongest combination of the leading sampled quality, near-fastest completion and lower cost than Gemini 3.1. Muse 1.2 is a lower-cost alternative at 89/100, 126.772 seconds and $0.017770. Mercury's 50.856-second, $0.013707 completion scored only 70/100 because of frequent malformed Hungarian; its speed alone does not justify the earlier recommendation.

<!-- smartfast-full-quality-benchmark:end -->
