# September 15 translation-cohort benchmark

[Current README comparison](../../README.md#subtitle-translation-leaderboard) · [Historical runs](history.md)

<!-- translation-cohort-benchmark:start -->
### 24 models on a full feature film, September 15, 2026

**11 of 24 models returned a complete file inside ten minutes.** Every model translated the same 1,338-cue English subtitle for *The Matrix* into Hungarian, through the queued encrypted content API, with 100-cue batches, four parallel batches, SmartFast routing at its default policy and a 600-second cap per job. Models ran in waves of five against a container started fresh for each wave.

The cohort is OpenRouter's ten most-used models for the `translation` task classification as of 2026-09-14, plus the free and low-cost candidates worth testing for subtitle work.

#### What finished

| Model | Result | Seconds | Requests | Observed cost, USD | Defect cues |
| --- | --- | ---: | ---: | ---: | ---: |
| `inception/mercury-2.5` | Complete | 70.7 | 17 | $0.0142 | 0 |
| `deepseek/deepseek-v4-flash` | Complete | 142.8 | 21 | $0.0145 | 0 |
| `tencent/hy-mt2-30b-a3b` | Complete | 93.2 | 41 | $0.0165 | 234 |
| `tencent/hy-mt2-7b` | Complete | 103.1 | 42 | $0.0167 | 3 |
| `meta/muse-spark-1.2-contributor` | Complete | 114.2 | 17 | $0.0176 | 0 |
| `tencent/hy-mt2-1.8b` | Partial, 1333/1338 | 113.0 | 202 | $0.0178 | 22 |
| `google/gemma-4-26b-a4b-it` | Complete | 290.1 | 23 | $0.0182 | 15 |
| `meta/muse-spark-1.3-contributor` | Complete | 192.0 | 17 | $0.0184 | 0 |
| `inclusionai/ling-3.0-flash-vl` | Complete | 577.2 | 19 | $0.0201 | 0 |
| `google/gemini-2.5-flash-lite` | Complete | 39.7 | 18 | $0.0212 | 0 |
| `openai/gpt-5.6-luna` | Complete | 85.4 | 17 | $0.0446 | 0 |
| `deepseek/deepseek-v4.1-flash` | Complete | 256.3 | 21 | $0.1893 | 0 |

Total observed charge for the run was **$0.8726**, including the models that did not finish.

#### What did not finish inside the cap

| Model | Progress at 600s |
| --- | ---: |
| `deepseek/deepseek-v4-flash-0731` | 94% |
| `tencent/hy3` | 88% |
| `poolside/laguna-s-2.1:free` | 70% |
| `z-ai/glm-5.3-flash` | 64% |
| `poolside/laguna-xs-2.1:free` | 58% |
| `dots-studio/dots-3-note-preview:free` | 47% |
| `qwen/qwen3.8-flash` | 47% |
| `ibm-granite/granite-4.2-8b` | 29% |
| `liquid/lfm-2.5-2.6b:free` | 17% |
| `inclusionai/ling-3.0-flash-fin` | 17% |
| `nvidia/nemotron-3.5-lightning:free` | 0% |
| `nex-agi/nex-n2.5-pro:free` | Failed: no endpoint passed the routing limits |

**Not one free model finished the film.** That held in the September 10 run as well. A capped job reports no usage, so its cost is unknown rather than zero.

Timing for the slower models varies substantially between runs. `deepseek/deepseek-v4-flash-0731` completed in 483 seconds on one run and capped at 94% on the next; `tencent/hy3` completed in 459 seconds and then capped at 88%. Treat anything above about 200 seconds as a range, not a measurement. `google/gemini-2.5-flash-lite` is the opposite case, landing within a few seconds and a fraction of a cent across four separate runs.

#### Defects, counted over every delivered cue

These are mechanical counts, not judgements. Each detector corresponds to something found while reading the output, promoted to a full-file count so the ranking does not rest on a sample.

| Model | Foreign-script cues | Invented markup tags | English left in |
| --- | ---: | ---: | ---: |
| Eight models, listed below | 0 | 0 | 0 |
| `tencent/hy-mt2-7b` | 3 | 0 | 0 |
| `google/gemma-4-26b-a4b-it` | 9 | 6 | 0 |
| `tencent/hy-mt2-1.8b` | 7 | 6 | 9 |
| `tencent/hy-mt2-30b-a3b` | 0 | 234 | 0 |

Clean on all three counts: `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4.1-flash`, `google/gemini-2.5-flash-lite`, `inception/mercury-2.5`, `inclusionai/ling-3.0-flash-vl`, `meta/muse-spark-1.2-contributor`, `meta/muse-spark-1.3-contributor`, `openai/gpt-5.6-luna`.

Foreign-script cues carry characters from a writing system neither English nor Hungarian uses, mid-sentence:

```
google/gemma-4-26b-a4b-it   cue 379   'Ez a mag的核心...'
google/gemma-4-26b-a4b-it   cue 513   'a로hol még meleg'
google/gemma-4-26b-a4b-it   cue 206   'hogy megmutatom ඔබට a középszellememet'
tencent/hy-mt2-7b           cue 1286  'Ismerem. Ne担心.'
tencent/hy-mt2-1.8b         cue 90    '他们看着你，Neo.'
```

`tencent/hy-mt2-30b-a3b` writes clean Hungarian but inserted 234 `<i>` and `<br>` tags into a source file that contains none.

#### Sampled reading

Fourteen cues were chosen for translation difficulty before any output existed: idiom, register, wordplay, proper nouns and the film's own terminology. This is a small sample and supports a grouping, not a score.

**Reliable.** `openai/gpt-5.6-luna`, `deepseek/deepseek-v4.1-flash`, `meta/muse-spark-1.2-contributor`, `meta/muse-spark-1.3-contributor`, `deepseek/deepseek-v4-flash`. Idiomatic, consistent terminology, correct register. All render "the One" as *a Kiválasztott*, the established Hungarian rendering.

**Usable with reservations.** `google/gemini-2.5-flash-lite` replaced cue 1102 entirely, giving *Apa voltál* for "Morpheus, you were more than a leader to us". `inclusionai/ling-3.0-flash-vl` put the same line in the wrong person. `inception/mercury-2.5` is grammatically loose in places (*Morpheus hisz, hogy*) and translates "the One" literally as *az Egy*. `tencent/hy-mt2-30b-a3b` reads well but needs its invented markup stripped.

**Weak.** `google/gemma-4-26b-a4b-it` is mostly sound but renders "the One" as *az Úr*, drops foreign script into three cues, and replaced cue 1200 with an unrelated sentence. `tencent/hy-mt2-7b` picks the wrong verb for "know" throughout (*ismerem* where *tudom* is meant, and the reverse), and hallucinated cue 206 outright.

**Unusable.** `tencent/hy-mt2-1.8b`. *A szürke kettőzó elárta a megjövőt* for "A black cat went past us", *Mi az az?* for "Where is it?", 40 cues returned unchanged and 202 requests to get there.

The three Tencent Hy-MT2 models lead OpenRouter's translation-task usage ranking. On this workload they are the three weakest results in the cohort that finished.

#### How this was measured

Jobs went through `POST /api/v1/jobs/translate/content` with an encrypted per-request key, the same wire format Bazarr+ uses. Temperature was 0.3, omitted for Luna because it does not accept one; reasoning was unspecified. Each request carried a 600-second provider timeout and each job a 600-second wall-clock cap. SmartFast ran at its defaults: median +50%, sparse-pool 3x, a 20% estimated-speed band, and $1 input / $3 output per million token ceilings, with exact zero caps for free variants.

Blank cues were excluded from the submission. The reference file has two, and a cue with no text cannot come back translated.

Six models could not be benchmarked at all, for reasons outside the model:

- `inclusionai/ling-3.0-flash-vl:free`, `ling-3.0-flash-sante:free`, `ling-3.0-flash-fin:free`: HTTP 404, "All providers have been ignored", from the account's data-policy setting. Reproduced with a direct call and no sidecar involved.
- `thinkingmachines/inkling:free`, `inkling-small:free`: HTTP 403, available only to agentic harnesses.
- `nex-agi/nex-n2.5-mini:free`: its sole endpoint reports `status: -2`, so SmartFast declines it. A direct call succeeds.

Raw results: [CSV](2026-09-15-translation-cohort.csv), [defect counts](2026-09-15-translation-cohort-defects.csv). Translated subtitle text is not kept in the repository.

#### Five defects this run found

The first pass ran against v2.1.0 and its results were discarded, because they measured the sidecar's bugs as much as the models. All five are fixed, and the numbers above come from a build containing the fixes.

1. A running job could not be cancelled through the API, which answered 200 as though it had been. Capped jobs kept running, kept billing and held their workers until later jobs never started at all. (#23)
2. Blank cues were dispatched, split down to a single line, and failed, turning a complete file into `partial` on every model in the cohort. (#24)
3. One translated line over 2000 characters failed response validation and discarded a finished film. `tencent/hy-mt2-1.8b` lost all 1,338 translated cues this way. (#25)
4. Batches were planned from `OPENROUTER_MAX_TOKENS` without regard to the model's own output ceiling, so a 4096-token model was handed 80-cue batches and cascaded 80 to 40 to 20 to 10 to 5 to 1. (#26)
5. SmartFast pins one endpoint, and a hard rejection from it lost the batch rather than falling back to the pool it had already ranked. `openai/gpt-5.6-luna` lost 240 cues to three batches routed to an endpoint that cannot serve it. (#27)

With the fixes in place Luna returns a complete file where it previously delivered 1098 of 1338 cues.

<!-- translation-cohort-benchmark:end -->
