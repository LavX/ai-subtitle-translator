# Ten-minute SmartFast routing and stall diagnosis

The latency-unit correction was verified before this eight-model run. OpenRouter catalog latency is milliseconds; the parser now converts it to seconds before adding output tokens divided by throughput. Three regressions failed before the correction and pass after it. All 1067 tests pass on Python 3.12, 3.13 and 3.14, and the corrected image passed 65 installed encrypted-API checks. See the [official schema](https://openrouter.ai/openapi.json).

## Endpoint selection

An independent calculation using the captured authenticated metadata, literal request workload estimates, price caps, cold-cost outlier thresholds and 20% speed band matched all eight initial cold choices. This validates the initial selection against advertised data. Later requests can change after rate limits, cooldowns, learned speed and metadata refresh; cache affinity can retain an eligible endpoint until three successful slow observations. Advertised speed is not a guarantee of future completion time.

| Model | First endpoint | Advertised latency, ms | Throughput, tokens/s | Estimated first-request seconds | Input / output USD per million |
| --- | --- | ---: | ---: | ---: | --- |
| `z-ai/glm-5.3-flash` | `modal/fp8` | 430 | 129 | 30.942 | 0.149985 / 0.499950 |
| `deepseek/deepseek-v4-flash-0731` | `reka/fp4` | 640 | 74 | 53.829 | 0.110000 / 0.660000 |
| `qwen/qwen3.7-flash` | `alibaba` | 741 | 35 | 113.198 | 0.030000 / 0.130000 |
| `qwen/qwen3.8-flash` | `makora/fp4` | 647.5 | 113 | 35.479 | 0.150000 / 0.470000 |
| `inclusionai/ling-3.0-flash-fin:free` | `novita` | 1315 | 129 | 35.331 | 0.000000 / 0.000000 |
| `dots-studio/dots-3-note-preview:free` | `atlas-cloud/fp8` | 1396.5 | 51 | 78.573 | 0.000000 / 0.000000 |
| `liquid/lfm-2.5-2.6b:free` | `liquid/fp8` | 658 | 110 | 36.440 | 0.000000 / 0.000000 |
| `nvidia/nemotron-3.5-lightning:free` | `nvidia/nvfp4` | 13484.5 | 10 | 407.084 | 0.000000 / 0.000000 |

For DeepSeek, the selected Reka estimate was 53.829 seconds versus the fastest eligible estimate of 47.737 seconds, within the allowed 20% band and at a lower estimated cold cost. All actual DeepSeek responses came from Reka. Nemotron had only one eligible free endpoint, whose initial estimate was 407.085 seconds. That exceeds this run's unchanged 120-second per-request deadline; a 600-second job cap alone does not give an individual request that much time.

## What happened while progress looked stuck

| Model | Reported translated cues | HTTP 429 | Interrupted requests | Interrupted with only whitespace | Content-filter responses | Smallest requested batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inclusionai/ling-3.0-flash-fin:free` | 0 | 0 | 0 | 0 | 0 | 40 |
| `deepseek/deepseek-v4-flash-0731` | 1340 | 0 | 0 | 0 | 0 | 40 |
| `z-ai/glm-5.3-flash` | 400 | 46 | 4 | 4 | 0 | 50 |
| `qwen/qwen3.7-flash` | 200 | 0 | 16 | 16 | 0 | 50 |
| `qwen/qwen3.8-flash` | 400 | 18 | 16 | 16 | 0 | 50 |
| `dots-studio/dots-3-note-preview:free` | 417 | 0 | 9 | 9 | 12 | 1 |
| `liquid/lfm-2.5-2.6b:free` | 501 | 16 | 5 | 5 | 0 | 1 |
| `nvidia/nemotron-3.5-lightning:free` | 0 | 0 | 20 | 20 | 0 | 50 |

Whitespace-only interruptions mean upstream headers and keepalive bytes arrived but no non-whitespace response bytes arrived before cancellation or timeout. These receipts establish a response-body wait rather than an idle local queue. They do not distinguish upstream queuing from model generation before a buffered response. HTTP 429 responses establish rate limiting; the service also waits through bounded endpoint cooldowns.

Dots returned 12 content_filter finish reasons; nine of its completed responses had empty translation text. Liquid and Dots also returned incomplete output, triggering smaller recovery requests. Top-level cue progress updates when a root batch finishes, so it can stay flat while child requests recover cues. Repeated small requests and root-only progress reporting can therefore look stalled. The root recovery deadline is 360 seconds; cooldown/provider errors can prevent the special all-timeout cohort stop from classifying a group as pure timeouts, though the external 600-second cap still bounds this run.

Ling was submitted without response_format or ignore and with exact zero price caps. Its repeated HTTP 404 responses stated that all providers had been ignored. The internal OpenRouter reason is not established by this request. Free-endpoint capability fallback was exercised, and no paid fallback was allowed.

## Reasoning and measurement limits

Every job left reasoning unspecified to preserve the previous request settings. That permits model-default reasoning. DeepSeek reported 112,403 reasoning tokens out of 182,044 completion tokens, affecting elapsed time and price. Reasoning tokens from other models are included in the diagnostic JSON. These are sums of reported usage, with unknown values kept null when no usage was received. The JSON includes response counts and completeness flags; these totals do not include unobserved reasoning or completion tokens.

The observer forwarded completion request bodies unchanged and preserved upstream response bodies/status. It did not capture or forward Retry-After headers. The product normally reads that header, so these receipts do not prove identical retry delays to a direct Bazarr+ connection when an upstream value is supplied. Rate-limit counts and costs are observations of this harness. The initial endpoint-selection audit and DeepSeek's no-429 completion are unaffected by this gap. No unobserved Retry-After value is assumed.

One run with eight concurrent jobs cannot isolate the contribution of the routing correction from changing provider load. The previous run used sixteen jobs and a 300-second cap. Structural completion does not establish translation quality. Reported response costs are lower bounds whenever usage is missing.

Numeric evidence: [results](2026-09-09-smartfast-tenminute.json), [diagnostic counts](2026-09-09-smartfast-tenminute-diagnosis.json), [endpoint metadata](2026-09-09-smartfast-tenminute-endpoints.json), [independent first-choice calculations](2026-09-09-smartfast-tenminute-routing-audit.json).
