# Historical translation benchmarks

These September 7 and 9 measurements used earlier builds, routes or smaller samples. Recommendations and estimates below are historical. Use the [current README comparison](../../README.md#subtitle-translation-leaderboard) for the September 10 full-cohort run. Original numeric artifacts are unchanged.

## Subtitle translation leaderboard

In our September 7, 2026 English-to-Hungarian smoke test, **Luna offered the strongest quality/value** and **Gemini 3.1 Flash Lite was the fastest practical option**. DeepSeek V4 Flash 0731 produced readable output with longer waits, but cost more than Luna. This is a qualitative judgment from 20 synthetic cues per pass, not a general translation-quality ranking.

| Model ID | Result | Time / 20 cues | Mean cost / 20 cues | Main finding |
|---|---|---:|---:|---|
| `openai/gpt-5.6-luna:floor` | 2/2 initial passes | 14.81s | $0.000455 | Best quality/value in this sample |
| `google/gemini-3.1-flash-lite:floor` | 2/2 initial passes | 3.19s | $0.000823 | Fast; one negation error |
| `deepseek/deepseek-v4-flash-0731:floor` | 2/2 selected retries | 170.23s | $0.000877 | Readable, with some literal idioms |
| `z-ai/glm-5.3-flash:floor` | 2/2 initial passes | 11.38s | $0.000258 | Cheapest of these, but made meaning errors |

Times are medians. Initial requests had a 100-second limit; DeepSeek's two initial requests timed out, then succeeded at 148 and 193 seconds with a 600-second limit. Its row uses successful retries only and excludes unknown charges from the original timeouts. Completion means all cue IDs returned, not correct language or meaning.

**Service compatibility:** the benchmark used a direct request harness. The service now omits temperature when OpenRouter's cached model catalog lists it as unsupported, including Luna. A September 7 live GUI API check translated 20 synthetic Hungarian-to-English cues with Luna on `:nitro` in 3.73 seconds for $0.000577. Two direct 20-cue `:floor` probes timed out with a 120-second request timeout, with model-default and low reasoning respectively. These small checks establish the tested route behavior, not full-movie reliability. Provider and prompt differences still apply.

For DeepSeek or other slow `:floor` providers, consider a longer HTTP timeout in `.env` (forwarded by Compose):

```dotenv
OPENROUTER_DEFAULT_MODEL=deepseek/deepseek-v4-flash-0731:floor
REQUEST_TIMEOUT=600
```

Apply changed settings by recreating the container (`docker compose up -d`) or restarting a manual process. This is an HTTP timeout, not a whole-job deadline, and differs from the benchmark's hard request cutoff.

### Episode, movie and season cost estimates

A **cue** is one timed subtitle entry, which can contain one or two lines of text. For budgeting, use **15 cues per minute**, with **10–20 cues per minute** as illustrative lower/higher dialogue-density scenarios. These are planning assumptions, not a measured universal average or a statistical confidence interval. Count the actual cues in your subtitle file when available.

For context, a [study of Swedish subtitles for selected reality and documentary TV episodes](https://jatjournal.org/index.php/jat/article/download/195/84) found 12.9 cues per minute in its 2020s sample (section 6.1.1). We round upward to 15 for the examples below; applying that rate to other genres and movies is our budgeting assumption. The movie example uses two hours, not a claimed average film runtime. Quiet films can fall below the range; fast dialogue and accessibility captions can exceed it.

At 15 cues/minute, budget **300 cues for 20 minutes**, **600 for 40 minutes** and **1,800 for a two-hour movie**. “Full season” below means 24 episodes.

#### Estimated API cost in USD, one target language

These examples use the planning cue counts below and the measured mean cost per 20 cues. All four model routes use `:floor`. DeepSeek 0731 uses its two successful longer-timeout retries; the other models use their two initial completed passes. GLM is included for price comparison despite the meaning errors noted in the leaderboard.

| What you translate | Cues | Luna | DeepSeek V4 Flash 0731 | Gemini 3.1 Flash Lite | GLM 5.3 Flash |
|---|---:|---:|---:|---:|---:|
| 20-minute episode | 300 | $0.0068 | $0.0132 | $0.0124 | $0.0039 |
| 40-minute episode | 600 | $0.0137 | $0.0263 | $0.0247 | $0.0077 |
| Movie (120-minute example) | 1,800 | $0.0410 | $0.0789 | $0.0741 | $0.0232 |
| Mini-series: 8 × 20-minute episodes | 2,400 | $0.0546 | $0.1052 | $0.0988 | $0.0309 |
| Mini-series: 8 × 40-minute episodes | 4,800 | $0.1092 | $0.2104 | $0.1976 | $0.0618 |
| Full season: 24 × 20-minute episodes | 7,200 | $0.1639 | $0.3156 | $0.2964 | $0.0927 |
| Full season: 24 × 40-minute episodes | 14,400 | $0.3277 | $0.6313 | $0.5928 | $0.1854 |

For scale, the two-hour movie estimate is about **4.10 US cents with Luna** or **7.89 cents with DeepSeek 0731**. A 24-episode season of 40-minute episodes is about **$0.33 with Luna** or **$0.63 with DeepSeek 0731**.

The 10–20 cues/minute scenarios make each central estimate approximately **0.67×–1.33×** as large, holding text length per cue and all other factors constant. For example, the two-hour movie is roughly **$0.0273–$0.0546 with Luna**, or **$0.0526–$0.1052 with DeepSeek 0731**, from cue-density variation alone.

```text
estimated cues = minutes per episode × episode count × 15
estimated cost = mean measured cost per 20 cues × estimated cues / 20
Luna, 24 × 40 min = $0.000455175 × 14,400 / 20 = $0.327726
```

Use the same formula with any model's cost per 20 cues in the leaderboard. For an actual file, substitute its cue count. For multiple target languages or seasons, sum their estimates; multiplying by their count is a rough shortcut when the volumes are similar.

These are **API-cost extrapolations, not measured full-episode or movie bills**. They assume similar text length per cue, 20-cue batches, unchanged provider prices and reasoning settings, and no failed attempts or retries. Unknown charges from timed-out requests are excluded. Cue length, target language, batch size, reasoning volume, retries, caching and promotions can change the bill beyond the density range. Hosting and any account-level fees or taxes are not included. Request latency does not scale directly into movie runtime because parallelism and batch sizing change it.



<!-- smartfast-sdk-benchmark:start -->
### SmartFast with the official SDK and ten-minute requests, September 9, 2026

**1/8 models in the repeated eight-model cohort returned a structurally complete file.** These eight jobs launched together with a 600-second cap each, using the same 1,340-cue English Matrix subtitle file for Hungarian translation through the actual 2.0.0 RC encrypted queued-content API. Each job used 100-cue batches, four parallel batches, temperature 0.3, unspecified reasoning and a stable opaque session. Each encrypted request explicitly set requestTimeout to 600 seconds, and the service used the same 600-second default. The external 600-second job cap bounded work even though the root recovery budget was 1,800 seconds. The observer allowed 605 seconds of read inactivity within the absolute remaining job window. This is the Bazarr+ wire format and authentication flow; the harness did not run inside Bazarr.

Completion requests now use the [official OpenRouter Python SDK](https://openrouter.ai/docs/client-sdks/python/overview), pinned to 1.1.133. The SDK constructs and dispatches requests with its retries disabled; the translator retains its response parsing, recovery rules and cancellation. The migration preserves reasoning fields missing from this SDK schema. Prices and authenticated endpoint metrics were captured before completion admission. Latency scoring retains the earlier milliseconds-to-seconds correction.

SmartFast kept its median +50% price cutoff, sparse-pool 3x limit, 20% estimated speed band and $1 input / $3 output per million token ceilings. Free variants retained exact zero price caps. Affinity can retain an eligible provider until three successful slow observations, so this is not an absolute-fastest guarantee on every request. This repeats the preceding eight-model cohort with a changed SDK, request deadline and Retry-After handling. Changing endpoint conditions and these combined changes prevent attributing timing differences to the SDK alone.

| Model | Result | Progress cues | Returned cues | Seconds | Requests | Service tokens | Cache-read tokens | Observed cost, USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3.7-flash` | Complete | 1340 | 1340 | 590.412 | 24 | 235127 | 2560 | $0.026018 |
| `inclusionai/ling-3.0-flash-fin:free` | Failed | 0 | 0 | 92.788 | 14 | unknown | unknown | unknown |
| `nvidia/nemotron-3.5-lightning:free` | 600s cap | 0 | 0 | 600.012 | 7 | unknown | 0 | $0.000000* |
| `dots-studio/dots-3-note-preview:free` | 600s cap | 0 | 0 | 600.014 | 32 | unknown | 7680 | $0.000000* |
| `liquid/lfm-2.5-2.6b:free` | 600s cap | 484 | 0 | 600.017 | 86 | 147871 | 46000 | $0.000000* |
| `qwen/qwen3.8-flash` | 600s cap | 800 | 0 | 600.020 | 17 | 145303 | 2048 | $0.060812* |
| `deepseek/deepseek-v4-flash-0731` | 600s cap | 500 | 0 | 600.022 | 9 | 251175 | 0 | $0.159868* |
| `z-ai/glm-5.3-flash` | 600s cap | 600 | 0 | 600.023 | 31 | 127921 | 0 | $0.043054* |

Observed response charges totaled **$0.28975240**, with **79 requests missing cost values**. An asterisk marks a lower bound. Service counters include finished roots, while observed usage includes returned responses from work still in progress. Missing usage is unknown, including for an interrupted free request. Cache counts are actual response usage, not inferred savings.

The observer forwarded upstream Retry-After headers unchanged. Receipts record their presence and normalized delay, while response status and body bytes pass through unchanged. Diagnostic failures are isolated from response delivery and recorded separately.

Progress cues count service-reported translated work. Returned cues describe result structure and may include original-text fallbacks for partial jobs. A capped running job can report progress without exposing result lines. Complete structure and preserved timestamps do not establish translation quality. HTTP 200 headers and keepalive bytes do not establish a completed response.

Raw results: [JSON](2026-09-09-smartfast-sdk.json), [CSV](2026-09-09-smartfast-sdk.csv), [price preflight](2026-09-09-smartfast-sdk-prices.json), [authenticated endpoint metrics](2026-09-09-smartfast-sdk-endpoints.json), [routing and stall diagnosis](2026-09-09-smartfast-sdk-diagnosis.md). Prior numeric artifacts remain unchanged.
<!-- smartfast-sdk-benchmark:end -->

<!-- smartfast-tenminute-benchmark:start -->
### Ten-minute SmartFast retry after latency correction, September 9, 2026

**1/8 previously unfinished models returned a structurally complete file.** These eight jobs launched together with a 600-second cap each, using the same 1,340-cue English Matrix subtitle file for Hungarian translation through the actual 2.0.0 RC encrypted queued-content API. Each job used 100-cue batches, four parallel batches, temperature 0.3, unspecified reasoning and a stable opaque session. Individual request deadlines remained 120 seconds, and each root retained its existing 360-second recovery budget. This is the Bazarr+ wire format and authentication flow; the harness did not run inside Bazarr.

Before this retry, a confirmed speed-scoring defect was corrected: catalog latency is milliseconds and now converts to seconds before adding output-token generation time. The earlier tables retain measurements made before that correction. Their speed selections are not validated by this run. See the [official endpoint schema](https://openrouter.ai/openapi.json). Prices and authenticated endpoint metrics were captured, and the production request body passed unchanged through an observer that records header/body timing, keepalive bytes, output counts and reported usage.

SmartFast kept its median +50% price cutoff, sparse-pool 3x limit, 20% estimated speed band and $1 input / $3 output per million token ceilings. Free variants retained exact zero price caps. Affinity can retain an eligible provider until three successful slow observations, so this is not an absolute-fastest guarantee on every request. The smaller eight-model cohort, longer deadline and corrected ranking make this a separate diagnostic run, not a controlled timing comparison.

| Model | Result | Progress cues | Returned cues | Seconds | Requests | Service tokens | Cache-read tokens | Observed cost, USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek/deepseek-v4-flash-0731` | Complete | 1340 | 1340 | 218.023 | 14 | 218281 | 0 | $0.124135 |
| `inclusionai/ling-3.0-flash-fin:free` | Failed | 0 | 0 | 92.002 | 14 | unknown | unknown | unknown |
| `z-ai/glm-5.3-flash` | 600s cap | 400 | 0 | 600.002 | 54 | 40044 | 0 | $0.016347* |
| `qwen/qwen3.7-flash` | 600s cap | 200 | 0 | 600.118 | 22 | 51808 | 0 | $0.002737* |
| `qwen/qwen3.8-flash` | 600s cap | 400 | 0 | 600.120 | 38 | 30807 | 640 | $0.010860* |
| `dots-studio/dots-3-note-preview:free` | 600s cap | 417 | 0 | 600.123 | 73 | 166860 | 18432 | $0.000000* |
| `liquid/lfm-2.5-2.6b:free` | 600s cap | 501 | 0 | 600.125 | 103 | 157041 | 55536 | $0.000000* |
| `nvidia/nemotron-3.5-lightning:free` | 600s cap | 0 | 0 | 600.126 | 20 | unknown | unknown | unknown |

Observed response charges totaled **$0.15407905**, with **164 requests missing cost values**. An asterisk marks a lower bound. Service counters include finished roots, while observed usage includes returned responses from work still in progress. Missing usage is unknown, including for an interrupted free request. Cache counts are actual response usage, not inferred savings.

The observer did not preserve upstream Retry-After headers. Rate-limit responses are recorded, but exact retry timing can differ from a direct Bazarr+ connection. The diagnosis details this limitation.

Progress cues count service-reported translated work. Returned cues describe result structure and may include original-text fallbacks for partial jobs. A capped running job can report progress without exposing result lines. Complete structure and preserved timestamps do not establish translation quality. HTTP 200 headers and keepalive bytes do not establish a completed response.

Raw results: [JSON](2026-09-09-smartfast-tenminute.json), [CSV](2026-09-09-smartfast-tenminute.csv), [price preflight](2026-09-09-smartfast-tenminute-prices.json), [authenticated endpoint metrics](2026-09-09-smartfast-tenminute-endpoints.json), [routing and stall diagnosis](2026-09-09-smartfast-tenminute-diagnosis.md). Prior numeric artifacts remain unchanged.
<!-- smartfast-tenminute-benchmark:end -->

<!-- smartfast-recovery-benchmark:start -->
### Full-file SmartFast rerun after routing correction, September 9, 2026

**8/16 models returned a structurally complete file within five minutes.** All 16 jobs launched together, each receiving the same 1,340-cue English Matrix subtitle file for Hungarian translation. The corrected 2.0.0 candidate ran 100-cue batches with four parallel batches per job, temperature 0.3 and reasoning unspecified. Requests used the encrypted API-key override, `X-Auth-Token`, zero-based cue positions, Movie context, `POST /api/v1/jobs/translate/content` and status polling, matching the Bazarr+ wire flow. This harness did not execute inside Bazarr or change its settings.

Every model used `:smartfast`. Provider prices were queried first. Defaults were median +50%, up to 20% slower estimated request time, 3x cheapest for sparse pools, and $1 input / $3 output per million token ceilings. Free variants required zero prices. A transparent observer preserved the production request body and recorded routing and usage. A 300-second watchdog stopped the isolated candidate; upstream response deadlines also enforced the cap. Qwen 3.5 remained excluded, and DeepSeek V4 Flash used the requested 0731 ID.

| Model | Result | Progress cues | Returned cues | Seconds | Requests | Service tokens | Cache-read tokens | Observed cost, USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemini-3.5-flash-lite` | Complete | 1340 | 1340 | 36.587 | 14 | 80831 | 0 | $0.119861 |
| `google/gemini-3.1-flash-lite` | Complete | 1340 | 1340 | 36.618 | 19 | 83992 | 0 | $0.075068 |
| `google/gemini-2.5-flash-lite` | Complete | 1340 | 1340 | 41.525 | 17 | 82846 | 0 | $0.021324 |
| `inception/mercury-2.5` | Complete | 1340 | 1340 | 44.680 | 14 | 114725 | 0 | $0.012936 |
| `openai/gpt-4o-mini` | Complete | 1340 | 1340 | 72.734 | 15 | 68891 | 0 | $0.025552 |
| `openai/gpt-5.6-luna` | Complete | 1340 | 1340 | 75.553 | 14 | 62753 | 0 | $0.042619 |
| `meta/muse-spark-1.2-contributor` | Complete | 1340 | 1340 | 137.173 | 14 | 105740 | 0 | $0.017691 |
| `meta/muse-spark-1.3-contributor` | Complete | 1340 | 1340 | 166.383 | 14 | 109940 | 1875 | $0.018347 |
| `inclusionai/ling-3.0-flash-fin:free` | Failed | 0 | 0 | 91.944 | 14 | unknown | unknown | unknown |
| `z-ai/glm-5.3-flash` | 300s cap | 0 | 0 | 300.001 | 17 | unknown | unknown | unknown |
| `dots-studio/dots-3-note-preview:free` | 300s cap | 200 | 0 | 300.001 | 22 | 30467 | 3072 | $0.000000* |
| `deepseek/deepseek-v4-flash-0731` | 300s cap | 100 | 0 | 300.002 | 17 | 15656 | 0 | $0.002039* |
| `qwen/qwen3.8-flash` | 300s cap | 200 | 0 | 300.002 | 22 | 17716 | 0 | $0.008751* |
| `liquid/lfm-2.5-2.6b:free` | 300s cap | 400 | 0 | 300.003 | 63 | 120513 | 40176 | $0.000000* |
| `qwen/qwen3.7-flash` | 300s cap | 200 | 0 | 300.004 | 12 | 31920 | 5120 | $0.002592* |
| `nvidia/nemotron-3.5-lightning:free` | 300s cap | 0 | 0 | 300.009 | 8 | unknown | unknown | unknown |

Observed charges totaled **$0.34677928**, with **87 forwarded requests missing a cost value**. An asterisk marks a lower-bound cost, excluding unknown charges from interrupted or missing-usage responses. Service token/cost counters and observed response usage are separate measurements. A free endpoint can still fail capability, health or rate-limit checks.

Progress cues are the service's reported translated count; returned cues measure recovered result structure. A job stopped while processing can report progress without exposing result lines. Partial results may contain original-text fallbacks. Complete structure and unchanged timings do not establish translation accuracy. This was one parallel run, not repeated trials or an isolated provider-speed test; shared load and account rate limits affect results. Cache-read counts are actual returned usage, not an inferred cache guarantee.

This run uses the corrected router: partial, invalid and blank output can recover without quarantining a healthy endpoint, while temporary provider cooldowns retry within existing limits. Native JSON mode is optional: otherwise eligible endpoints without it receive the same translation instructions without the unsupported response_format field. The original run below is preserved. This is a fresh parallel measurement, so differences also reflect provider load, endpoint availability and model output variability.

The JSON-mode fallback was exercised live: all 14 Ling requests and all eight Nemotron requests omitted `response_format` and kept zero price caps. Ling received OpenRouter HTTP 404 responses stating "All providers have been ignored"; the request routing did not contain an `ignore` field. Nemotron returned HTTP 200 headers but no completed response bodies before the request/job deadlines. These are recorded separately from the resolved capability rejection.

Raw numeric receipts: [JSON](2026-09-09-smartfast-recovery.json), [CSV](2026-09-09-smartfast-recovery.csv), [provider price preflight](2026-09-09-smartfast-recovery-prices.json). The JSON includes exact timing, routing, provider attribution, cost-completeness flags and source/image hashes. Earlier tables below retain their original routes and settings and are not directly comparable.
<!-- smartfast-recovery-benchmark:end -->

<!-- smartfast-parallel-benchmark:start -->
### Historical full-file SmartFast benchmark, September 9, 2026

**5/16 models returned a structurally complete file within five minutes.** All 16 jobs launched together, each receiving the same 1,340-cue English Matrix subtitle file for Hungarian translation. The actual 2.0.0 candidate ran 100-cue batches with four parallel batches per job, temperature 0.3 and reasoning unspecified. Requests used the encrypted API-key override, `X-Auth-Token`, zero-based cue positions, Movie context, `POST /api/v1/jobs/translate/content` and status polling, matching the Bazarr+ wire flow. This harness did not execute inside Bazarr or change its settings.

Every model used `:smartfast`. Provider prices were queried first. Defaults were median +50%, up to 20% slower estimated request time, 3x cheapest for sparse pools, and $1 input / $3 output per million token ceilings. Free variants required zero prices. A transparent observer preserved the production request body and recorded routing and usage. A 300-second watchdog stopped the isolated candidate; upstream response deadlines also enforced the cap. Qwen 3.5 remained excluded, and DeepSeek V4 Flash used the requested 0731 ID.

| Model | Result | Progress cues | Returned cues | Seconds | Requests | Service tokens | Cache-read tokens | Observed cost, USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemini-2.5-flash-lite` | Complete | 1340 | 1340 | 36.383 | 14 | 80580 | 0 | $0.021021 |
| `openai/gpt-4o-mini` | Complete | 1340 | 1340 | 74.613 | 14 | 68246 | 0 | $0.025460 |
| `openai/gpt-5.6-luna` | Complete | 1340 | 1340 | 84.654 | 14 | 62515 | 0 | $0.042333 |
| `meta/muse-spark-1.2-contributor` | Complete | 1340 | 1340 | 133.031 | 14 | 105451 | 1250 | $0.017510 |
| `meta/muse-spark-1.3-contributor` | Complete | 1340 | 1340 | 177.210 | 14 | 113619 | 1875 | $0.019083 |
| `nvidia/nemotron-3.5-lightning:free` | Failed | 0 | 0 | 2.403 | 0 | unknown | unknown | $0.000000 |
| `inclusionai/ling-3.0-flash-fin:free` | Failed | 0 | 0 | 2.406 | 0 | unknown | unknown | $0.000000 |
| `google/gemini-3.5-flash-lite` | Partial | 310 | 1340 | 10.311 | 4 | 20701 | 0 | $0.027878 |
| `inception/mercury-2.5` | Partial | 1098 | 1340 | 33.500 | 12 | 104002 | 0 | $0.011802 |
| `google/gemini-3.1-flash-lite` | Complete API, 1 blank cue(s) | 1340 | 1340 | 44.551 | 14 | 81214 | 0 | $0.075132 |
| `liquid/lfm-2.5-2.6b:free` | Partial | 261 | 1340 | 44.630 | 4 | 36576 | 1376 | $0.000000 |
| `qwen/qwen3.8-flash` | Failed | 0 | 0 | 121.844 | 4 | unknown | unknown | unknown |
| `qwen/qwen3.7-flash` | Failed | 0 | 0 | 121.847 | 4 | unknown | unknown | unknown |
| `dots-studio/dots-3-note-preview:free` | Partial | 100 | 1340 | 121.996 | 4 | 15753 | 0 | $0.000000* |
| `z-ai/glm-5.3-flash` | 300s cap | 1240 | 0 | 300.002 | 17 | 61169 | 0 | $0.013151* |
| `deepseek/deepseek-v4-flash-0731` | 300s cap | 200 | 0 | 300.002 | 14 | 30863 | 0 | $0.017433* |

Observed charges totaled **$0.27080201**, with **24 forwarded requests missing a cost value**. An asterisk marks a lower-bound cost, excluding unknown charges from interrupted or missing-usage responses. Service token/cost counters and observed response usage are separate measurements. A free endpoint can still fail capability, health or rate-limit checks.

Progress cues are the service's reported translated count; returned cues measure recovered result structure. A job stopped while processing can report progress without exposing result lines. Partial results may contain original-text fallbacks. Complete structure and unchanged timings do not establish translation accuracy. This was one parallel run, not repeated trials or an isolated provider-speed test; shared load and account rate limits affect results. Cache-read counts are actual returned usage, not an inferred cache guarantee.

**Historical SmartFast limitation exposed by this run:** the measured candidate could quarantine an endpoint after partial HTTP 200 output and fail immediately when no other endpoint remained. This affected several partial results, including Mercury. These numbers describe that implementation and are not evidence that the models cannot translate the full file.

The subsequent recovery correction keeps incomplete or invalid output in bounded recovery, retries missing or blank cues, and reserves endpoint quarantine for transport failures and upstream errors. A sole endpoint, including a free endpoint, can be retried after its cooldown within the existing deadline. Native JSON response mode is optional: endpoints without it receive the same JSON translation instructions and output validation, with the unsupported field omitted. Free variants retain zero price caps on every retry. Local regression tests verify this behavior; these benchmark numbers precede the correction. The separate corrected rerun above records the later live measurement.

Raw numeric receipts: [JSON](2026-09-09-smartfast-parallel.json), [CSV](2026-09-09-smartfast-parallel.csv), [provider price preflight](2026-09-09-smartfast-parallel-prices.json). The JSON includes exact timing, routing, provider attribution, cost-completeness flags and source/image hashes. Earlier tables below retain their original routes and settings and are not directly comparable.
<!-- smartfast-parallel-benchmark:end -->

### Full benchmark results

<details>
<summary>All initial completions, 100-second cutoff</summary>

### Completed both initial passes with `:floor`

Every model below returned all 20 cues in both passes without truncation. **Completion measures structure, not translation correctness.** Costs are actual reported API charges in USD, including billed reasoning tokens. Time is median request latency for one 20-cue pass.

| Model ID | Time / 20 cues | Mean cost / 20 cues | Estimated cost / 600 cues | Review |
|---|---:|---:|---:|---|
| `openai/gpt-5.6-luna:floor` | 14.81s | $0.000455 | $0.0137 | Best quality/value in this sample; natural idioms, minor awkward phrasing |
| `google/gemini-3.1-flash-lite:floor` | 3.19s | $0.000823 | $0.0247 | Best speed option; one negation error reversed meaning |
| `openai/gpt-5.6-luna-pro:floor` | 28.65s | $0.001962 | $0.0588 | Strong output; no demonstrated quality gain over Luna |
| `z-ai/glm-5.3-flash:floor` | 11.38s | $0.000258 | $0.0077 | Cheapest; fire alarm became starting a fire in one pass |
| `tencent/hy3:floor` | 41.28s | $0.002572 | $0.0772 | Readable, but literal idioms and lost detail |
| `tencent/hy3-preview:floor` | 63.47s | $0.003041 | $0.0912 | Readable, but awkward literal phrasing |
| `nex-agi/nex-n2-pro:floor` | 47.62s | $0.006434 | $0.1930 | Mostly readable; costly, with some literal idioms |
| `arcee-ai/trinity-large-thinking:floor` | 4.31s | $0.000799 | $0.0240 | Fast, but meaning errors and invented words |
| `minimax/minimax-m2.7:floor` | 20.99s | $0.002145 | $0.0643 | Poor idioms and person/tense errors |
| `google/gemma-4-31b-it:floor` | 77.18s | $0.001213 | $0.0364 | Broken Hungarian and mixed-language output |
| `google/gemini-3.5-flash-lite:floor` | 9.05s | $0.001233 | $0.0370 | Readable in places, but malformed Hungarian words in both passes |
| `openai/gpt-4o-mini:floor` | 11.42s | $0.000593 | $0.0178 | Readable, but literal translations of close call and low profile |
| `deepseek/deepseek-v4-flash-vision-exp:floor` | 68.71s | $0.003255 | $0.0976 | Mostly readable; no demonstrated benefit for text subtitles |

Vision Exp is a separate model and is not evidence for DeepSeek V4 Flash 0731.

</details>

<details>
<summary>Initial failures and selected longer-timeout retries</summary>

### Initial incomplete or timed-out `:floor` runs

| Model ID | Complete passes | Observed limitation |
|---|---:|---|
| `stepfun/step-3.7-flash:floor` | 1/2 | One 100-second timeout; grammatical and meaning errors in the completed pass |
| `qwen/qwen3.6-35b-a3b:floor` | 1/2 | One output truncated at the token cap; completed pass reversed who forgives whom |
| `minimax/minimax-m3:floor` | 0/2 | Returned only one untranslated English cue per pass |
| `xiaomi/mimo-v2.5-pro:floor` | 0/2 | Both requests exceeded 100 seconds; translation quality not assessed |
| `deepseek/deepseek-v4-flash-0731:floor` | 0/2 | Both requests exceeded 100 seconds; no output available for quality assessment |
| `deepseek/deepseek-v4-flash:floor` (0423) | 1/2 | One 100-second timeout; completed pass added profanity and used a literal idiom |
| `google/gemini-2.5-flash-lite:floor` | 0/2 | One timeout; other pass shifted all cue indices from 1–20 to 0–19 |
| `qwen/qwen3.5-flash-02-23:floor` | 0/2 | Both responses contained only a number, no translations |
| `qwen/qwen3.7-flash:floor` | 1/2 | One 429; completed pass changed 9:30 to 9:15 and reversed who forgives whom |

These are the original 100-second results, preserved separately from the longer-timeout follow-up below. Timeouts do not establish a translation-quality failure. No movie-cost estimate is made from these failed attempts.

### Longer-timeout follow-up with `:floor`

The 100-second cutoff was too restrictive for some routes. Only the four timed-out requests from the top-ten follow-up were repeated with a **600-second limit**. Prompts, low reasoning request, 8,192-token cap, price filters and parsing stayed the same. These are selected retries, not a fresh independent benchmark; the other pass results were retained.

| Model ID | Completed retries | Time per retry | Mean cost / 20 cues | Estimated cost / 600 cues | Review |
|---|---:|---:|---:|---:|---|
| `deepseek/deepseek-v4-flash-0731:floor` | 2/2 | 147.90s / 192.57s | $0.000877 | $0.0263 | Readable Hungarian, some literal idioms; no added profanity in these retries |
| `google/gemini-2.5-flash-lite:floor` | 1/1 | 8.21s | $0.001107 | $0.0332 | Correct indices on retry; separate original pass with shifted indices remains unsuccessful |
| `deepseek/deepseek-v4-flash:floor` | 1/1 | 325.01s | $0.000542 | $0.0163 | Older 0423 version; readable with some awkward phrasing |

All four retries completed and reported $0.00340 in charges. These cost projections use only the successful retries, **excluding unknown charges from the original timed-out requests**. They are not the total cost of obtaining the successful output across all attempts. DeepSeek 0731 now has two complete results across four total attempts; 0423 has two across three; Gemini 2.5 Flash Lite has one across three. The original 100-second series and these selected 600-second retries should not be compared as equal success-rate samples. No timeout setting in the service was changed.

</details>

<details>
<summary>Earlier throughput comparisons and screening</summary>

### Earlier comparisons, different settings

These measurements used throughput routing before the `:floor` series. They are kept separate because neither latency nor cost is directly comparable across routes and reasoning settings.

| Model | Successful sample | Time / 20 cues | Mean cost / 20 cues | Estimated cost / 600 cues | Review |
|---|---|---:|---:|---:|---|
| `deepseek/deepseek-v4-flash-0731` | One low-reasoning follow-up | 6.31s | $0.000389 | $0.0117 | Mostly natural, but added profanity; two earlier attempts with default reasoning and a smaller output cap were truncated |
| `google/gemini-2.5-flash` | Two passes with default reasoning | 5.10s | $0.002429 | $0.0729 | Mostly natural; one overly literal idiom |

Other models from the earlier throughput screening are excluded from the recommendations above. These outcomes apply to that configuration, not every possible provider or prompt:

| Earlier model | Complete passes | Screening result |
|---|---:|---|
| `inception/mercury-2.5-preview` | 2/2 | Poor Hungarian phrasing and idioms |
| `nvidia/nemotron-3.5-lightning` | 2/2 | Severe Hungarian language errors |
| `tencent/hy-mt2-7b` | 2/2 | Poor Hungarian and mixed-language output |
| `tencent/hy-mt2-30b-a3b` | 1/2 | Missing cue in one pass; awkward wording |
| `poolside/laguna-s-2.1` | 1/2 | One truncated pass; poor Hungarian |
| `tencent/hy-mt2-1.8b` | 0/2 | Repetition and truncated output |
| `poolside/laguna-xs-2.1` | 0/2 | Empty content |
| `inclusionai/ling-3.0-flash-fin` | 0/2 | Unusable structured output |
| `ibm-granite/granite-4.2-8b` | 0/2 | Unusable structured output; no translated cues |
| `qwen/qwen3.8-flash` | 0/2 | Truncated output; follow-up hit a rate limit |
| `meta/muse-spark-1.2-contributor` | 0/2 | Access denied (403); quality not assessed |
| `meta/muse-spark-1.3-contributor` | 0/2 | Access denied (403); quality not assessed |

</details>

<details>
<summary>Method, routing, sample limits and total charges</summary>

### Test method and limits

The `:floor` series used the service's translation prompt, formatter and tolerant parser at source commit `16b32b3`, with direct OpenRouter requests rather than the live job queue. Each request used low reasoning effort, an 8,192-token output cap and JSON-object response format. Temperature was 0.3 where supported, omitted for Luna and Luna Pro. The initial 44 requests ran with concurrency three, randomized order, no retries and a 100-second timeout. The four selected follow-up retries used concurrency three, a fixed submission order and a 600-second timeout. Provider price filters capped eligible routes at $1 per million input tokens and $3 per million output tokens; the earlier throughput screening used $0.50 and $3 respectively. The harness sent JSON mode even to routes without advertised support. Production requests can differ, including omission of JSON mode when reasoning is enabled.

Latest aliases were deduplicated, and listings marked batch were tested as their base model with `:floor`, not as asynchronous batch jobs. Hungarian output was reviewed qualitatively for meaning, idioms, grammar and completeness; there was no blinded panel or formal accuracy scoring. The original 30 `:floor` attempts reported $0.06226, excluding unknown charges for three timeouts. Seven additional models from the translation top ten received two passes each with the same settings. These 14 attempts reported $0.00632, with four timeouts and one 429 returning no cost. Before the longer-timeout retries, combined reported charges were $0.06858. Including those four retries, reported charges across the 48 floor attempts total $0.07199; seven original timeouts and one 429 returned no cost. Provider selection and prices can change between requests.

All ten models in OpenRouter's translation request-share ranking for the seven days ending September 6, 2026 are now covered by two `:floor` attempts each. GLM 5.3 Flash, Luna and Gemini 3.1 Flash Lite reuse the original measurements; the other seven were added without repeating those three. The catalog maps DeepSeek 0423 to `deepseek/deepseek-v4-flash` and Qwen3.5 Flash to `qwen/qwen3.5-flash-02-23`. Popularity is not a translation-quality score. Source: [OpenRouter Data API](https://openrouter.ai/docs/cookbook/administration/data-api#task-classifications), as of 2026-09-06, licensed under CC BY 4.0.

The service's recommended model list with metadata is available at `GET /api/v1/models`. It is a curated list, not the full OpenRouter catalog or this leaderboard.

</details>

