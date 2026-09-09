# SmartFast SDK routing and response diagnosis

Completion requests used official OpenRouter Python SDK 1.1.133 in the actual 2.0.0 RC. The installed image matched all 33 application files. Before this run, all 1,083 tests passed on each of Python 3.12, 3.13 and 3.14, and 65 installed encrypted-API checks passed. Independent wire checks verified 600-second SDK timeouts, nonzero Retry-After, zero free-endpoint caps, optional JSON omission, raw usage and cancellation. The SDK adds no retry loop. Its unused synchronous client interface owns no network resources, avoiding synchronous TLS setup during async requests.

Both requestTimeout and the candidate service request timeout were 600 seconds. Each job had a separate 600-second external cap; its 1,800-second root recovery budget could not extend that cap. The observer used a 605-second read-inactivity setting within the absolute remaining job window. It preserved upstream Retry-After values, status and response-body bytes. Receipt processing could not interrupt delivery, and the final manifest recorded no diagnostic errors.

## Endpoint selection

An independent calculation using captured authenticated metadata, literal request workload estimates, price caps, cold-cost outlier thresholds and the 20% speed band matched all eight initial cold choices. This checks initial selection against advertised data. Later requests may change after rate limits, cooldowns, learned speed or metadata refresh; affinity can retain an eligible endpoint until three successful slow observations. Advertised speed does not guarantee future response time. Latency remains converted from milliseconds to seconds.

| Model | First endpoint | Advertised latency, ms | Throughput, tokens/s | Estimated first-request seconds | Input / output USD per million |
| --- | --- | ---: | ---: | ---: | --- |
| `z-ai/glm-5.3-flash` | `modal/fp8` | 436 | 155 | 28.746 | 0.149985 / 0.499950 |
| `deepseek/deepseek-v4-flash-0731` | `deepseek` | 875.5 | 81 | 49.468 | 0.220000 / 0.660000 |
| `qwen/qwen3.7-flash` | `alibaba` | 726 | 43 | 92.261 | 0.030000 / 0.130000 |
| `qwen/qwen3.8-flash` | `makora/fp4` | 905 | 110 | 36.687 | 0.150000 / 0.470000 |
| `inclusionai/ling-3.0-flash-fin:free` | `novita` | 1331 | 131 | 31.377 | 0.000000 / 0.000000 |
| `dots-studio/dots-3-note-preview:free` | `atlas-cloud/fp8` | 1331.5 | 52 | 85.716 | 0.000000 / 0.000000 |
| `liquid/lfm-2.5-2.6b:free` | `liquid/fp8` | 653 | 111 | 37.842 | 0.000000 / 0.000000 |
| `nvidia/nemotron-3.5-lightning:free` | `nvidia/nvfp4` | 15721 | 9 | 474.388 | 0.000000 / 0.000000 |

## What happened while progress looked stuck

| Model | Reported translated cues | HTTP 429 | Interrupted requests | Interrupted with only whitespace | Content-filter responses | Smallest requested batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inclusionai/ling-3.0-flash-fin:free` | 0 | 0 | 0 | 0 | 0 | 40 |
| `qwen/qwen3.7-flash` | 1340 | 0 | 0 | 0 | 0 | 1 |
| `nvidia/nemotron-3.5-lightning:free` | 0 | 0 | 4 | 3 | 0 | 50 |
| `dots-studio/dots-3-note-preview:free` | 0 | 0 | 4 | 4 | 10 | 5 |
| `liquid/lfm-2.5-2.6b:free` | 484 | 14 | 4 | 4 | 0 | 1 |
| `qwen/qwen3.8-flash` | 800 | 5 | 4 | 4 | 0 | 100 |
| `deepseek/deepseek-v4-flash-0731` | 500 | 0 | 4 | 4 | 0 | 100 |
| `z-ai/glm-5.3-flash` | 600 | 20 | 4 | 4 | 0 | 100 |

Whitespace-only interruptions establish that upstream headers and keepalive bytes arrived, but no non-whitespace response bytes arrived before cancellation. They show a response-body wait rather than an idle local queue. They do not distinguish upstream queuing from generation before a buffered response. HTTP 429 establishes rate limiting. Cue progress updates when a root batch finishes, so it may stay flat while child requests recover output.

- `inclusionai/ling-3.0-flash-fin:free`: 0 complete HTTP 200 response bodies, 0 responses with fewer output cues than requested, 0 empty-content responses, and 0 completed responses taking over 120 seconds. Allowed endpoint attempts: novita (14).
- `qwen/qwen3.7-flash`: 24 complete HTTP 200 response bodies, 2 responses with fewer output cues than requested, 0 empty-content responses, and 7 completed responses taking over 120 seconds. Allowed endpoint attempts: alibaba (24).
- `nvidia/nemotron-3.5-lightning:free`: 3 complete HTTP 200 response bodies, 0 responses with fewer output cues than requested, 1 empty-content responses, and 2 completed responses taking over 120 seconds. Allowed endpoint attempts: nvidia/nvfp4 (7).
- `dots-studio/dots-3-note-preview:free`: 27 complete HTTP 200 response bodies, 1 responses with fewer output cues than requested, 8 empty-content responses, and 1 completed responses taking over 120 seconds. Allowed endpoint attempts: atlas-cloud/fp8 (32).
- `liquid/lfm-2.5-2.6b:free`: 68 complete HTTP 200 response bodies, 6 responses with fewer output cues than requested, 0 empty-content responses, and 0 completed responses taking over 120 seconds. Allowed endpoint attempts: liquid/fp8 (86).
- `qwen/qwen3.8-flash`: 8 complete HTTP 200 response bodies, 0 responses with fewer output cues than requested, 0 empty-content responses, and 5 completed responses taking over 120 seconds. Allowed endpoint attempts: makora/fp4 (6), alibaba (11).
- `deepseek/deepseek-v4-flash-0731`: 5 complete HTTP 200 response bodies, 0 responses with fewer output cues than requested, 0 empty-content responses, and 5 completed responses taking over 120 seconds. Allowed endpoint attempts: deepseek (9).
- `z-ai/glm-5.3-flash`: 6 complete HTTP 200 response bodies, 0 responses with fewer output cues than requested, 0 empty-content responses, and 6 completed responses taking over 120 seconds. Allowed endpoint attempts: modal/fp8 (8), makora (6), baseten/fp8 (4), friendli (3), phala/fp8 (1), together (1), relace/fp4 (8).

Nemotron returned a 50-cue child response after 552.718 seconds, with unique indices and no blank cues. Top-level translated progress remained zero because its root had not finished. Zero root progress therefore does not mean no child output arrived.

Ling was submitted without native response_format and with exact zero price caps. Its HTTP 404 error envelopes reported that all providers had been ignored. The request did not specify ignore. These receipts establish an upstream rejection but not OpenRouter's internal reason. No paid fallback was allowed.

## Reasoning and measurement limits

All jobs left reasoning unspecified, preserving the previous request settings and allowing model-default reasoning. Reported reasoning and completion totals are below. Counts identify responses with usage; missing values remain unknown.

| Model | Reasoning tokens | Responses with reasoning usage | Completion tokens | Responses with completion usage |
| --- | ---: | ---: | ---: | ---: |
| `inclusionai/ling-3.0-flash-fin:free` | unknown | 0 / 14 | unknown | 0 / 14 |
| `qwen/qwen3.7-flash` | 146223 | 24 / 24 | 190259 | 24 / 24 |
| `nvidia/nemotron-3.5-lightning:free` | 20181 | 3 / 7 | 25059 | 3 / 7 |
| `dots-studio/dots-3-note-preview:free` | 65403 | 27 / 32 | 79484 | 27 / 32 |
| `liquid/lfm-2.5-2.6b:free` | 123298 | 68 / 86 | 154295 | 68 / 86 |
| `qwen/qwen3.8-flash` | 95199 | 8 / 17 | 122784 | 8 / 17 |
| `deepseek/deepseek-v4-flash-0731` | 223904 | 5 / 9 | 237749 | 5 / 9 |
| `z-ai/glm-5.3-flash` | 73388 | 6 / 31 | 112128 | 6 / 31 |

23 responses included Retry-After. Observed normalized delay values in seconds: [1.0, 7.0, 22.0, 28.0, 36.0, 38.0, 50.0, 53.0]. The observer forwarded the original header unchanged; it recorded only presence and normalized numeric delay.

This repeats the previous eight-model cohort with a changed SDK, request deadline and Retry-After handling. Changing provider conditions and these combined changes prevent isolating an SDK speed effect. Structural completion and preserved timestamps do not establish translation quality. Costs are lower bounds whenever usage is missing. A session creates a caching opportunity, while only reported cache usage establishes actual cache reads.

Numeric evidence: [results](2026-09-09-smartfast-sdk.json), [diagnostic counts](2026-09-09-smartfast-sdk-diagnosis.json), [endpoint metadata](2026-09-09-smartfast-sdk-endpoints.json), [independent first-choice calculations](2026-09-09-smartfast-sdk-routing-audit.json).
