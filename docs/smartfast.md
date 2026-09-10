# SmartFast routing

SmartFast chooses an OpenRouter endpoint for the model you selected using price limits, estimated request time and affinity for each translation job. It is opt-in. It does not change the GUI's Lowest price default or the API's existing routing behavior when SmartFast is absent.

## Enable it

In the GUI, choose **Provider routing > SmartFast (price + speed)**. Open **Request options** to adjust its thresholds and price limits. The routing selector takes precedence over a pasted routing suffix. Switching away from SmartFast removes its suffix and leaves any underlying `:free` or `:thinking` variant intact. Inactive SmartFast settings are not sent on other routes.

For an API request, set `config.provider.sort` to `"smartfast"`. Alternatively, append `:smartfast` to the exact model ID, for example `openai/gpt-5.6-luna:smartfast`. Bazarr+ clients can use that suffix in the translator model field when a custom model ID is accepted. Ensure any accompanying provider sort is absent or `smartfast`.

The suffix belongs to this translator. It is removed before OpenRouter model discovery and translation requests. To use an available free or thinking variant, append it after the variant: `author/model:free:smartfast` or `author/model:thinking:smartfast`. The underlying variant must exist and retains its capability requirements. SmartFast does not make a paid model free or disable mandatory reasoning.

Do not combine SmartFast with `:floor`, `:nitro`, another routing sort, or a nonempty `config.provider.order`. Those conflicts fail before a completion request. `only` and `ignore` restrictions narrow the eligible endpoints, and `allowFallbacks: false` remains enforced. A manually selected provider must still pass every SmartFast check.

## Controls

The API tuning object is `config.provider.smartFast`. Omitted fields use these defaults:

| Field | Default | Allowed values | Meaning |
| --- | --- | --- | --- |
| `medianPremiumPercent` | 50 | 0 to 1000 | With at least five eligible endpoints, permit estimated cold cost up to this percentage above their median. |
| `speedTolerancePercent` | 20 | 0 to 1000 | Permit estimated request time up to this percentage above the fastest known estimate in the price pool. |
| `sparsePremiumMultiplier` | 3 | 1 to 100 | With fewer than five eligible endpoints, permit cold cost up to this multiple of the cheapest eligible endpoint. |
| `maxPromptPrice` | 1 | 0 to 1000 | Maximum quoted input rate, in USD per million tokens. |
| `maxCompletionPrice` | 3 | 0 to 1000 | Maximum quoted output rate, in USD per million tokens. |

Values must be finite numbers. Zero is valid except for the sparse multiplier, whose minimum is one. Unknown policy fields are rejected.

**Price limits are quoted unit rates, not a total bill budget.** Output length, reasoning tokens, retries and the number of batches affect the bill. SmartFast checks quoted cache read/write rates against the input limit and quoted reasoning rates against the output limit. It disallows nonzero per-request charges and sends `max_price` on every completion request, including requests that permit fallback. Free variants force input, output and request ceilings to zero regardless of the configured token limits. OpenRouter's upstream price filter also uses per-million token rates. See [provider routing and maximum prices](https://openrouter.ai/docs/guides/routing/provider-selection).

SmartFast uses standard endpoints by default. It never automatically upgrades to priority or switches to Flex. The API can explicitly request `config.serviceTier: "flex"` or `"priority"`; a tier-specific `only` restriction can also select that tier when `serviceTier` is omitted. In the GUI, such a restriction requires **Follow routing**. Every tier remains subject to the same price and capability checks.

## How selection works

1. Load the endpoint catalog for the exact model variant. Deduplicate endpoint tags and exclude unhealthy endpoints, unusable prices, incompatible tiers, unsupported required request parameters, insufficient context limits and endpoints outside your explicit restrictions. Pricing rules that the parser cannot evaluate safely make that endpoint ineligible. Catalogs are cached for 300 seconds. If refresh fails and no unexpired snapshot is available, SmartFast fails before sending a completion.
2. Estimate the same input and output workload for all eligible endpoints and apply absolute price caps. Input includes the translation instructions and cue text. Output allows for text expansion and an explicit reasoning budget. These are estimates, not tokenizer measurements or output limits.
3. Remove cold-cost outliers. At the defaults, a pool of five or more allows up to 1.5 times median cold cost; a smaller pool allows up to three times the cheapest cold cost. Cold costs keep cache expectations from distorting this comparison.
4. Estimate request time from recent successful local measurements, normalized for output size, or usable catalog latency/throughput data. Catalog latency is converted from milliseconds to seconds before combining it with output tokens divided by tokens per second. At the default tolerance, keep candidates estimated to take no more than 1.2 times the fastest estimate. If no usable speed data exists, use OpenRouter throughput routing inside the filtered price pool and mark the decision as bootstrap. Unknown speed is not evidence of the globally fastest endpoint, and an unknown estimate cannot displace known fast candidates merely by being absent.
5. For a new session, choose among fast candidates within 5% of the lowest effective cost, using current in-flight load to break ties. Keep an existing eligible endpoint pinned through small performance fluctuations. Three consecutive successful observations outside the speed band permit a new selection. Prices, restrictions and capabilities are checked again for each request.

Native JSON response mode is optional. Endpoints without `response_format` support remain eligible, including a sole paid or free endpoint. When reasoning is omitted or disabled, SmartFast sends `response_format: {"type": "json_object"}` only if every endpoint allowed for that request supports it. A mixed bootstrap pool omits the field, and each later request checks the selected route again, including after a provider switch. Translation instructions still request JSON, and the same cue validation and bounded output recovery apply. Explicit reasoning and other required parameters remain mandatory.

An attributed transport failure, upstream error or rate limit temporarily excludes that endpoint for 30 seconds and releases its affinity. Partial, blank, malformed or repaired HTTP 200 translation output stays in bounded output recovery. Useful cues are retained, blank cues remain missing, and incomplete or repaired output does not train successful routing speed. Reported token usage, cost and cache counts remain available.

Existing translation retries may select another eligible endpoint. If every otherwise eligible endpoint is cooling down, the router supplies the earliest expiry so the existing retry loop can wait within its retry and wall-clock limits. This also works with a single endpoint, including a free endpoint. Price caps, capabilities and explicit restrictions are checked before cooldowns, and a cooldown cannot promote a price outlier. A free variant keeps zero price caps on every retry and cannot fall through to paid capacity. Permanent ineligibility still fails immediately. SmartFast adds no separate completion retry loop.

Completion requests use the official OpenRouter Python SDK, pinned to version 1.1.133, with SDK retries disabled. The translator retains its shared asynchronous connection pool, raw response parsing and retry policy. Provider price caps are serialized as decimal strings required by the SDK. A narrow compatibility adapter preserves the existing reasoning `enabled` and `max_tokens` fields that this SDK schema omits.

`config.requestTimeout` sets the wall-clock limit for an individual provider call, including a response that keeps sending whitespace. It also configures the SDK's transport timeout. A benchmark job cap is a separate outer limit; increasing it alone does not extend provider calls. The ten-minute SDK benchmark sets both limits to 600 seconds.

## Sessions and cache behavior

The translator creates an opaque session for each translation operation and sends it as OpenRouter's `session_id`. A queued job retains the same identity through batches, retries, adaptive splits and recovery after a restart. Separate synchronous or streaming operations receive separate identities. Local affinity and observations are scoped by account, model and translation context. Their maps are bounded, expire after inactivity and are not a persistent cache guarantee.

Keeping requests on one provider improves the opportunity to reuse a prompt cache. Actual hits require provider/model support, a sufficiently long matching prefix and an unexpired cache. Session identity alone does not create cache hits. Manual `provider.order` takes precedence over upstream sticky routing, which is why SmartFast does not send it. See [OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching).

The translator keeps stable instructions ahead of changing cue data and preserves its existing supported Anthropic cache breakpoints. It does not pad prompts or add previous translations to manufacture cache hits. Warm-cost estimates require observed cache reads and a quoted cache-read price. Missing cache usage remains unknown. The README includes separate full-file runs before and after the recovery and JSON-capability corrections, alongside earlier routing comparisons. Original measurements remain unchanged. Each run reports its own provider conditions, actual usage and completeness limits; structural completion does not establish translation quality.

## API examples

These JSON bodies go to the translator, not directly to OpenRouter. Use your existing `X-Auth-Token` and API-key encryption setup described in [Authentication](../README.md#authentication). The examples omit credentials and use the service's configured provider key. Select a model available to your account; a valid model may still have no endpoint within the chosen policy.

For `POST /api/v1/translate/content` or `POST /api/v1/jobs/translate/content`:

```json
{
  "sourceLanguage": "en",
  "targetLanguage": "hu",
  "lines": [{"position": 1, "line": "The train leaves at noon."}],
  "config": {
    "model": "openai/gpt-5.6-luna",
    "serviceTier": "default",
    "provider": {
      "sort": "smartfast",
      "smartFast": {
        "medianPremiumPercent": 50,
        "speedTolerancePercent": 20,
        "sparsePremiumMultiplier": 3,
        "maxPromptPrice": 1,
        "maxCompletionPrice": 3
      }
    }
  }
}
```

For `POST /api/v1/translate/file` or `POST /api/v1/jobs/translate/file`, the suffix alone enables the default SmartFast policy:

```json
{
  "sourceLanguage": "en",
  "targetLanguage": "hu",
  "fileName": "sample.srt",
  "content": "1\n00:00:01,000 --> 00:00:03,000\nThe train leaves at noon.\n",
  "config": {
    "model": "openai/gpt-5.6-luna:smartfast",
    "serviceTier": "default"
  }
}
```

The free-variant form is `model:free:smartfast`. In this content-request template, replace `example/model` with a model that exposes an available `:free` variant before sending:

```json
{
  "sourceLanguage": "en",
  "targetLanguage": "hu",
  "lines": [{"position": 1, "line": "The train leaves at noon."}],
  "config": {"model": "example/model:free:smartfast"}
}
```

OpenRouter receives `example/model:free` for that final template, with zero input, output and request price ceilings. A free endpoint that fails the other eligibility checks is still excluded. OpenRouter documents model IDs, variants, capabilities and quoted pricing in its [model catalog guide](https://openrouter.ai/docs/guides/overview/models).

## Diagnostics and troubleshooting

At INFO level, server logs include sanitized `SmartFast routing:` records after routed attempts. They report the price and speed pools, selected or observed endpoint, bootstrap status, exclusion reasons, elapsed time, reported cost, cache reads/writes, endpoint health, output success and failure status when available. Unknown or ambiguous measurements remain unknown. These records contain no raw credentials, subtitle text, media titles or session IDs. They are also attached to internal provider results and errors; queued API responses and GUI job cards retain their existing schema.

If no endpoint is eligible, check the model variant, price ceilings, provider restrictions, tier and required reasoning/capabilities. Unsupported pricing rules or unavailable catalog data can also prevent selection. Raising a limit permits higher quoted rates; it does not guarantee completion. A bootstrap record means the router lacks a usable speed estimate. A missing cache-read count is not proof that caching is supported or that a hit occurred.
