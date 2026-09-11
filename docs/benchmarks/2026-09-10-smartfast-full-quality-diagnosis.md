# Full-cohort routing and deadline diagnosis

Current source: f80a190d7cb13845bc8a6b7564bb164f9dd83ae2. Actual 2.0.0 RC image with OpenRouter SDK 1.1.133; all 33 installed application files matched. All 16 encrypted queued jobs launched within 0.079 seconds. Ten returned all 1,340 cues. The original source and 38 existing containers were preserved.

All 284 forwarded requests passed price, session, payload and deadline checks. Fifteen sessions were observed upstream; Dots had zero forwarded requests. Independent first-request arithmetic passed for all 15 admitted models using authenticated metadata, price filtering, a 20% estimated speed band and cost preference. This establishes policy compliance at admission, not that advertised speed predicted actual latency. Session affinity and changing conditions mean later requests are not guaranteed the absolute fastest endpoint. Temperature was configured as 0.3; Luna omits the unsupported parameter.

| Model | Endpoint attempts | Complete HTTP 200 bodies | Interrupted whitespace-only | HTTP statuses | Observed reasoning tokens |
| --- | --- | ---: | ---: | --- | ---: |
| `dots-studio/dots-3-note-preview:free` | {} | 0 | 0 | {} | unknown |
| `google/gemini-3.5-flash-lite` | {"google-ai-studio": 14} | 14 | 0 | {"200": 14} | 0 |
| `google/gemini-3.1-flash-lite` | {"google-vertex/eu": 17} | 17 | 0 | {"200": 17} | 0 |
| `google/gemini-2.5-flash-lite` | {"google-ai-studio": 14} | 14 | 0 | {"200": 14} | 2 |
| `openai/gpt-5.6-luna` | {"amazon-bedrock/us-east-1": 14} | 14 | 0 | {"200": 14} | 3294 |
| `inception/mercury-2.5` | {"inception": 16} | 16 | 0 | {"200": 16} | 37864 |
| `inclusionai/ling-3.0-flash-fin:free` | {"novita": 14} | 0 | 0 | {"404": 14} | unknown |
| `meta/muse-spark-1.2-contributor` | {"meta": 14} | 14 | 0 | {"200": 14} | 34760 |
| `openai/gpt-4o-mini` | {"azure/swedencentral": 14} | 14 | 0 | {"200": 14} | 0 |
| `meta/muse-spark-1.3-contributor` | {"meta": 14} | 14 | 0 | {"200": 14} | 42224 |
| `z-ai/glm-5.3-flash` | {"baseten/fp8": 11, "crusoe/fp4": 1, "makora": 1, "fireworks": 4} | 14 | 0 | {"200": 14, "429": 3} | 54007 |
| `qwen/qwen3.7-flash` | {"alibaba": 17} | 17 | 0 | {"200": 17} | 120074 |
| `deepseek/deepseek-v4-flash-0731` | {"coreweave/fp8": 9} | 5 | 4 | {"200": 9} | 223453 |
| `liquid/lfm-2.5-2.6b:free` | {"liquid/fp8": 91} | 82 | 3 | {"200": 85, "429": 6} | 157139 |
| `nvidia/nemotron-3.5-lightning:free` | {"nvidia/nvfp4": 4} | 0 | 4 | {"200": 4} | unknown |
| `qwen/qwen3.8-flash` | {"makora/fp4": 6, "alibaba": 9} | 10 | 4 | {"200": 14, "429": 1} | 104298 |

- Dots: metadata offered no healthy endpoint within the configured limits, so routing failed without a completion request. Completion charges are known zero.
- Ling: Novita was admitted with free caps and optional native JSON, but all 14 upstream requests returned HTTP 404. No usable translation returned.
- Nemotron: four admitted requests received HTTP 200 headers, then whitespace keepalives without completed JSON bodies until the absolute deadline. No usable content or usage returned.
- DeepSeek: CoreWeave was the fastest eligible initial estimate. Five complete response bodies reported 223,453 reasoning tokens; four other requests were interrupted while only whitespace had arrived. Service progress reached 400 cues. The job did not deliver a file.
- Qwen 3.8: ten complete bodies, one 429, four interrupted whitespace-only responses;1,000 service progress cues. Completed responses reached 468.496 seconds. No file delivered before the cap.
- Liquid:82 complete bodies and repeated small recovery batches, six 429 responses, three interrupted whitespace-only requests;700 service progress cues. Captured fragments exceed settled service progress but are not a delivered file.
- GLM and Qwen 3.7 completed within the cap, at 501.726 and 564.744 seconds respectively. Their long response waits were real upstream behavior.

Reasoning was left unspecified for every model. This run therefore compares that common configuration, not each model after individual tuning. Headers and keepalives do not prove translation progress. Captured response fragments are used only for provisional quality diagnosis. Returned usage charges are lower bounds when requests lack usage. Stable sessions were verified; cache savings are reported only when upstream usage provides counts.

The accompanying quality report reviews 180 fixed source positions for complete files and only available positions for fragments. It does not establish full-file semantic correctness.
