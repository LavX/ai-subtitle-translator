# Sampled Hungarian translation quality

Blinded automated editorial review of a fixed purposive 180-cue sample: 12 evenly spaced 12-cue scene blocks plus 36 early non-overlapping thematic vocabulary cues. Frozen before live output. Meaning 50%, semantic completeness 20%, Hungarian fluency/register 20%, terminology 10%. Scores are subjective rubric judgments, not accuracy percentages; small gaps are not statistically established. Each model has one primary reviewer, with targeted contextual checks. No full-file semantic review or professional certification is implied. Fragment scores use only available sampled cues and are not comparable rankings.

All available cues checked mechanically. Flags use >42 characters per line, >2 lines, and >20 non-whitespace characters per second, excluding tags. These are declared heuristics, not measured semantic errors. Unchanged names or brief utterances can be valid. Captured fragments are diagnostic output, not a delivered subtitle file.

| Model | Score /100 | Sample reviewed | Output cues available | Complete file | Critical / major / minor findings |
| --- | ---: | ---: | ---: | --- | ---: |
| `openai/gpt-5.6-luna` | 93.0 | 180/180 | 1340 | Yes | 0 / 1 / 4 |
| `google/gemini-3.1-flash-lite` | 93.0 | 180/180 | 1340 | Yes | 0 / 1 / 3 |
| `meta/muse-spark-1.3-contributor` | 91.0 | 180/180 | 1340 | Yes | 0 / 1 / 6 |
| `meta/muse-spark-1.2-contributor` | 89.0 | 180/180 | 1340 | Yes | 0 / 0 / 5 |
| `google/gemini-3.5-flash-lite` | 84.0 | 180/180 | 1340 | Yes | 0 / 2 / 3 |
| `z-ai/glm-5.3-flash` | 81.0 | 180/180 | 1340 | Yes | 0 / 2 / 14 |
| `openai/gpt-4o-mini` | 81.0 | 180/180 | 1340 | Yes | 0 / 2 / 22 |
| `google/gemini-2.5-flash-lite` | 80.0 | 180/180 | 1340 | Yes | 0 / 2 / 8 |
| `inception/mercury-2.5` | 70.0 | 180/180 | 1340 | Yes | 0 / 8 / 35 |
| `qwen/qwen3.7-flash` | 58.0 | 180/180 | 1340 | Yes | 0 / 14 / 18 |
| `deepseek/deepseek-v4-flash-0731` | 87.0 provisional | 50/180 | 400 | No | 0 / 1 / 5 |
| `qwen/qwen3.8-flash` | 79.0 provisional | 144/180 | 1000 | No | 0 / 3 / 21 |
| `liquid/lfm-2.5-2.6b:free` | 29.0 provisional | 125/180 | 915 | No | 4 / 30 / 7 |
| `inclusionai/ling-3.0-flash-fin:free` | N/A | 0/180 | 0 | No | 0 / 0 / 0 |
| `nvidia/nemotron-3.5-lightning:free` | N/A | 0/180 | 0 | No | 0 / 0 / 0 |
| `dots-studio/dots-3-note-preview:free` | N/A | 0/180 | 0 | No | 0 / 0 / 0 |

## inclusionai/ling-3.0-flash-fin:free

No translated sample is available. Translation wording and semantic quality cannot be assessed; all linguistic scores are null.

## openai/gpt-5.6-luna

Strong and mostly natural Hungarian on all180 reviewed cues. One local meaning error and a handful of small wording issues; no critical defect found in this sample.
- Major, zero-based positions 491, 492: The crew is described as searching for the Matrix itself, rather than searching within it for the person. The following cue partly repairs the intended object.
- Minor, zero-based positions 301, 302: The metaphor about hiding reality becomes a literal and awkward construction about a world drawn over the eyes and blindness toward truth.
- Minor, zero-based positions 370: The year expression uses an incomplete or incorrect verb form, making an otherwise clear sentence ungrammatical.
- Minor, zero-based positions 607: Passivity or inertia is narrowed to indifference. The following dependency explanation keeps the broader point intact.
- Minor, zero-based positions 852: The short question responding to an unfinished reservation uses the wrong Hungarian case.

## z-ai/glm-5.3-flash

The180-cue sample is understandable but needs substantial Hungarian editing. Two local meaning defects and repeated grammar or idiom problems are present; no critical plot reversal identified.
- Major, zero-based positions 0: The prohibition on one person replacing the speaker becomes a prohibition on that person abandoning duty. The action and participant roles change.
- Minor, zero-based positions 1: Volunteering for a shift is expressed with a verb normally describing an appetite or craving.
- Minor, zero-based positions 8: The echo question asking what sound was meant becomes a question about what the respondent personally heard.
- Minor, zero-based positions 275, 285, 301, 314: Forms of address change from formal to informal during the explanatory dialogue without a contextual signal.
- Minor, zero-based positions 301, 302: The concealment metaphor uses an unidiomatic Hungarian blindness construction.
- Minor, zero-based positions 319: The rabbit-hole image uses an awkward invented compound instead of the ordinary term already used nearby.
- Minor, zero-based positions 328: The relative clause about the swallowed pill has incorrect Hungarian verb agreement.
- Minor, zero-based positions 362: The sentence about sore eyes mixes a plural verb with a singular Hungarian subject.
- Minor, zero-based positions 372: The question of what year it is uses an unnatural ordinal and verb combination.
- Minor, zero-based positions 608: An extra detached case ending appears in the explanation of dependence on the system.
- Minor, zero-based positions 615: The second-person conditional ends without the required Hungarian personal verb, leaving an incomplete or shifted predicate.
- Minor, zero-based positions 728: The negative comparison uses unnatural word order and agreement in Hungarian.
- Minor, zero-based positions 845: The bodily-certainty idiom becomes an unnatural phrase about a top and bottom. Adjacent target context conveys some bodily emphasis, so this is not treated as a total meaning loss.
- Minor, zero-based positions 852: The short follow-up about an unfinished reservation uses an unnatural accusative question.
- Minor, zero-based positions 1094: A recurring character name is misspelled in the question about killing him.
- Major, zero-based positions 1096: A taunt about using a human instead of a machine for the same task becomes a warning about sending humans to machine work.

## meta/muse-spark-1.2-contributor

Readable and semantically dependable on all180 sampled cues, with several localized grammar issues and an inconsistent switch in address. No critical or major defect identified in the reviewed sample.
- Minor, zero-based positions 2: The sentence about enjoying watching a specific person uses indefinite verb agreement and awkward pronoun placement.
- Minor, zero-based positions 275, 285, 301, 314: The same explanatory dialogue changes from formal address to informal address without a contextual signal. This makes the speaker relationship feel inconsistent.
- Minor, zero-based positions 301: An extra first-person prefix is attached to the metaphor about covering the addressee's eyes, producing a malformed mixed-reference phrase.
- Minor, zero-based positions 315, 316: The belief clause has mismatched Hungarian object and verb agreement across the cue boundary.
- Minor, zero-based positions 335: The source emphasizes complete certainty that a dream was real; the translation retains belief but drops that strength of conviction.

## meta/muse-spark-1.3-contributor

Semantically solid and complete within the180-cue sample, with one local meaning problem and recurring small Hungarian wording issues. No critical defect identified.
- Minor, zero-based positions 0: The statement that this person is not supposed to take over becomes a statement that taking over is unnecessary. The modal force is softened.
- Minor, zero-based positions 130, 132, 275, 285, 301: Formal and informal address alternate for the same interlocutor, including within adjacent lines of the telephone exchange.
- Minor, zero-based positions 275: The question about belief in fate uses an unnatural Hungarian verb construction and case.
- Minor, zero-based positions 301, 302: The concealment metaphor becomes a redundant and unidiomatic construction about drawing a world onto someone and blindness from truth.
- Minor, zero-based positions 371: The approximate-year comparison combines an unsuitable comparative adverb with an incomplete Hungarian expression.
- Minor, zero-based positions 852: The short follow-up to an unfinished reservation uses an unnatural accusative question.
- Major, zero-based positions 1096: The taunt about choosing a human to do a task suited to a machine becomes a warning against sending people to machine work. This loses the local contrast about the performer's suitability.

## qwen/qwen3.7-flash

All 180 sampled cues are present, but repeated meaning errors and malformed Hungarian make this sample unsuitable without substantial editing. No confident critical plot reversal is assigned.
- Major, zero-based positions 0: The duty-relief instruction becomes a prohibition on moving toward the speaker, losing the intended replacement relationship.
- Major, zero-based positions 1: Taking a shift becomes taking or buying a round, using the wrong sense of the work assignment.
- Major, zero-based positions 14, 15, 22: The lieutenant is repeatedly promoted to lieutenant-colonel; one occurrence is also malformed.
- Major, zero-based positions 116: A statement that the employer is a leading software company becomes a statement about the name of such a company.
- Major, zero-based positions 247: The warning about being bugged uses a malformed listening construction that does not clearly convey planted surveillance.
- Major, zero-based positions 269, 319: The recurring rabbit-hole image becomes a rabbit-tail tunnel and later a nonsensical rabbit-related expression.
- Major, zero-based positions 610: The instructor asks about the pupil listening and looking, but the translation changes these actions into first-person actions and changes the object.
- Major, zero-based positions 615: The concluding group-membership rule has a corrupted predicate. The malformed initial word could suggest an unintended negative, but does not support a confident polarity-reversal classification.
- Major, zero-based positions 724: The food joke interprets soft or liquid eggs as eggs that run, using the wrong sense of the adjective.
- Major, zero-based positions 733, 734: The cereal comparison garbles the relationship between the taste the speaker imagines and the hypothetical actual flavors.
- Major, zero-based positions 1091: The appeal that the speakers must be able to do something becomes a malformed statement in which something should do something.
- Major, zero-based positions 1096: The taunt about using a human instead of a machine becomes a statement about machine work, losing the contrast between performers.
- Major, zero-based positions 1212: The call for police reinforcements uses a rescue or saving term instead of the intended request for backup.
- Major, zero-based positions 1213: The police command to stop becomes a command to freeze an object, using the wrong verbal sense and object structure.
- Minor, zero-based positions 2: A basic second-person verb contains a spelling error.
- Minor, zero-based positions 15, 114, 119, 275, 285, 301, 848, 856: Person and formal versus informal forms of address are inconsistent in several conversations.
- Minor, zero-based positions 22: The possessive relation identifying the addressee as responsible for the men is omitted.
- Minor, zero-based positions 61: A question about being awake uses an unnatural waking construction.
- Minor, zero-based positions 132: Searching for the person is rendered with a more threatening hunting verb than the neutral source requires.
- Minor, zero-based positions 301, 302: The metaphor about concealment uses an awkward blindness construction.
- Minor, zero-based positions 328: The swallowed-pill clause uses a malformed verb that resembles running out rather than consuming.
- Minor, zero-based positions 368, 369: The question about when the person is becomes a by-when deadline question. The following year comparison partly restores the intended temporal frame.
- Minor, zero-based positions 488, 489, 490: The explanation has inconsistent clause mood or subject agreement that makes its progression awkward.
- Minor, zero-based positions 604, 607, 608: The explanation of people dependent on the system contains malformed function words and agreement.
- Minor, zero-based positions 727, 731, 733: The cereal name is inconsistently copied with a spelling change.
- Minor, zero-based positions 845: An invented compound makes the bodily-certainty idiom unnatural, though its broad emphasis remains recoverable.
- Minor, zero-based positions 847: The verb form treats the addressee as a third-person object rather than using the normal first-person-to-second-person form.
- Minor, zero-based positions 855: The statement about possessing a gift uses an unnatural definite-noun construction.
- Minor, zero-based positions 1211: The exclamation is an unnatural repetition of divine terms.
- Minor, zero-based positions 1332: Ending a telephone call is expressed as physically hanging the telephone, an unnatural literal rendering in this context.
- Minor, zero-based positions 1336: The phrase about having no boundaries has incorrect Hungarian postposition placement.
- Minor, zero-based positions 1338: The question about where to proceed shifts toward where to proceed from. The next statement still retains the addressee choice.

## liquid/lfm-2.5-2.6b:free

Provisional review of 125 available sampled cues from 915 captured cues. No completed subtitle file was delivered. Severe recurring language and meaning failures make the fragment unsuitable without extensive retranslation. Missing coverage is separate from these linguistic scores; grouped defect records are not a count of every defective cue.
- Major, zero-based positions 0: Both the readiness check and the instruction about relieving a shift become unrelated or incomplete statements.
- Major, zero-based positions 1, 2: Taking a shift becomes a collective change; enjoyment of watching the man becomes merely watching him.
- Major, zero-based positions 3: The dismissal of a ridiculous suggestion becomes a malformed statement about being an extreme.
- Major, zero-based positions 4, 5, 6, 7: The early exchange about believing in the One loses the title and replaces belief with unrelated liking or bringing language.
- Major, zero-based positions 9: The line-security question adds thanks, drops the line referent and changes who expresses certainty.
- Major, zero-based positions 11: The intention to leave is replaced by an unintelligible expression.
- Major, zero-based positions 14, 15, 22: The police rank becomes an invented word; the warning that the lieutenant’s men are already dead is largely unintelligible. The profanity also becomes a positive exclamation.
- Major, zero-based positions 38: A real informant becomes real information, losing the person being discussed.
- Critical, zero-based positions 45: The plot-driving instruction to follow the white rabbit becomes nonsensical text, so the clue motivating the next action is lost.
- Major, zero-based positions 119, 121, 122, 123: The job-choice ultimatum contains invented language, loses the named addressee and replaces the comprehension check with looking at oneself.
- Major, zero-based positions 132: Having searched for the other person becomes a statement of being present.
- Major, zero-based positions 242, 244: The road is known by a different grammatical actor, and the destination becomes the speaker’s situation rather than the listener’s unwanted outcome.
- Major, zero-based positions 246: Both physical instructions for the examination become unintelligible and no longer describe reclining and lifting a shirt.
- Major, zero-based positions 247: The surveillance-device explanation is malformed and fails to identify the problem clearly.
- Major, zero-based positions 248, 249, 251, 252: The extraction dialogue substitutes behaving and glass for encouragement and abuse, and invents an unintelligible object that might be lost.
- Major, zero-based positions 269, 319: The rabbit-hole image is lost through displaced context and unrelated language about ground depth.
- Major, zero-based positions 301: The deceptive world pulled over the eyes becomes a world surprising the eyes, losing the metaphor’s action.
- Major, zero-based positions 310: Being told what the Matrix is changes to someone having shown it.
- Major, zero-based positions 314, 315, 316, 317, 328: The pill-choice verbs and pill noun are repeatedly malformed, and awakening in bed becomes awakening in illness. The colors remain but the choice explanation is damaged.
- Major, zero-based positions 335, 336, 337, 338: The dream thought experiment is severely malformed, replacing waking with standing and weakening certainty and conditional relations.
- Major, zero-based positions 362: The speaker’s hurting eyes become hurting legs, breaking the following explanation about unused eyes.
- Major, zero-based positions 372: Uncertainty about which year it is becomes uncertainty about an amount of years.
- Critical, zero-based positions 484, 485: The premise that humanity cannot be free while the Matrix exists becomes malformed conditional text about being consumable, losing the fundamental reason for liberation.
- Major, zero-based positions 483, 486, 487, 488, 490, 491, 492: The prophecy passage repeatedly changes actors, weakens the prediction and death sequence, and changes searching within the Matrix into searching for it. The end of war remains recognizable.
- Major, zero-based positions 606: Readiness to be disconnected becomes being late for an English-derived action, losing the intended readiness condition.
- Major, zero-based positions 609: Fighting to protect the system is replaced by an unrelated possibility involving clay.
- Major, zero-based positions 610: The question about attention to the red-dressed woman is replaced by an unrelated untranslated English command.
- Major, zero-based positions 612: Pausing the program becomes a malformed trampling command.
- Critical, zero-based positions 615: The survival rule identifying anyone outside the group with the opposing side changes its actor and ends in gibberish, losing the decisive allegiance condition.
- Major, zero-based positions 731, 733, 734: The simulated cereal-taste argument becomes confused language about pasta, and oatmeal and tuna become dust and generic sea fish.
- Major, zero-based positions 845: The emphatic certainty idiom becomes an unclear statement about bodily damage.
- Major, zero-based positions 848: An instruction to vocalize during an examination becomes an instruction to write.
- Major, zero-based positions 850: The quoted medical-style qualification remains in English, leaving part of the available cue untranslated.
- Critical, zero-based positions 854: The denial of being the One instead denies being Life, changing the central identity being discussed; the reply also introduces an unrelated diminutive animal address.
- Minor, zero-based positions 8, 61, 114, 116, 130, 196, 197, 243, 274, 363, 364, 365, 366, 368, 369, 370, 494, 604, 607, 608, 611, 614, 730, 732, 847, 851, 853, 855, 856: Recurring wrong case endings, verb forms, articles, agreement and literal constructions affect even lines whose broad meaning can still be reconstructed.
- Minor, zero-based positions 10: Certainty in this instance becomes always being certain.
- Minor, zero-based positions 100: The first-person search shifts to a third-person grammatical actor.
- Minor, zero-based positions 127: A positive acknowledgment becomes a remark that something is unusual.
- Minor, zero-based positions 203: An offer sounding attractive becomes a direct assertion that it is a good contract.
- Minor, zero-based positions 275: The question about present belief in fate becomes a past-tense question.
- Minor, zero-based positions 245, 366: Personal names use inconsistent alternative spellings.

## deepseek/deepseek-v4-flash-0731

Provisional quality of 50 available sampled cues from 400 captured cues, with no completed file delivered. The fragment is generally readable and complete within those cues, with one consequential perspective error and a few local weaknesses. Its limited coverage excludes overall ranking.
- Major, zero-based positions 0: The instruction that the listener should not relieve the speaker becomes the speaker saying they did not come to relieve someone, changing both perspective and intent.
- Minor, zero-based positions 8: The request to clarify what was heard becomes a question about what the responding speaker himself heard.
- Minor, zero-based positions 245: The lights request is a stiff bare plural noun rather than a natural concise command.
- Minor, zero-based positions 247: An ongoing implanted surveillance problem becomes a past act of eavesdropping, weakening the device explanation.
- Minor, zero-based positions 248, 254: The relaxation request and surprise about the device are comprehensible but slightly unnatural literal expressions.
- Minor, zero-based positions 491: The description of people who spent their lives searching uses a stiff relative construction; the following cue retains the person being sought.

## qwen/qwen3.8-flash

Provisional assessment of all 144 available sampled cues from 1000 captured cues, with no completed file delivered. Much of the fragment is understandable, but several consequential mistranslations and recurring language defects need editing. It cannot enter the overall ranking because delivery and sample coverage are incomplete.
- Minor, zero-based positions 0: A prohibition about relieving the shift becomes a prediction that the listener will not do so.
- Major, zero-based positions 4: The title identifying the expected savior is replaced by an unrelated English word, leaving the initial belief statement malformed.
- Minor, zero-based positions 8: The request to clarify what was heard becomes a question about what the responding speaker himself heard.
- Minor, zero-based positions 15, 22: Address to the lieutenant shifts from informal to formal within the same confrontation without an evident reason in the supplied text.
- Minor, zero-based positions 61: The contrast between waking and dreaming becomes waking and sleeping, weakening the dream-reality distinction.
- Minor, zero-based positions 245, 254: The lights request and surprise about the device use stiff literal noun phrasing.
- Minor, zero-based positions 251: The abusive form of address has an incorrect case ending.
- Major, zero-based positions 252: Accidentally losing the moving device becomes deliberately throwing it away, changing the extraction warning.
- Minor, zero-based positions 269, 319: The recurring rabbit-hole compound is malformed in two different ways.
- Minor, zero-based positions 274, 302: The truth-related statements contain unnatural word order or verb government.
- Minor, zero-based positions 328: The relative clause about the swallowed pill uses the wrong definite verb conjugation.
- Minor, zero-based positions 368: The contrast between what and when retains awkward source-language word order.
- Minor, zero-based positions 487, 488: The verbs or nouns for the prophecy and coming are misspelled.
- Minor, zero-based positions 491: The lifelong search is expressed through a clumsy nominal construction, although the search location and person remain present.
- Minor, zero-based positions 607: Inertia or resistance to change becomes indecision.
- Minor, zero-based positions 609: Defending the system is retained, but the explicit willingness to fight is lost.
- Minor, zero-based positions 612: The request to pause the program becomes a bare freezing command with no clear object.
- Major, zero-based positions 615: Belonging to the opposing side becomes merely being an obstruction, losing the specific allegiance relation needed for the following explanation.
- Minor, zero-based positions 725: The deliberately gross mucus comparison uses a more neutral term, slightly weakening the register.
- Minor, zero-based positions 728: The elliptical reply about never having eaten the cereal uses unnatural Hungarian word order.
- Minor, zero-based positions 730: Reflective wondering becomes being surprised.
- Minor, zero-based positions 733, 734: The alternative cereal-taste comparison has a tangled relative clause and lacks the natural comparison link.
- Minor, zero-based positions 845: The total-certainty idiom keeps its meaning but loses the source vulgarity.
- Minor, zero-based positions 852: The short request for a qualification uses an unsuitable accusative question form.

## openai/gpt-4o-mini

Fully delivered file with all 180 sampled cues reviewed. Usually understandable and mostly faithful, but literal phrasing, a few malformed constructions, two consequential local mistranslations and observed cue displacement need editing. The purposive sample does not certify the entire file.
- Minor, zero-based positions 1: The wish to take a shift becomes a vague, awkward wish to change.
- Minor, zero-based positions 8: The clarification about the heard sound becomes a question about what the speaker himself heard.
- Minor, zero-based positions 22: The dead officers become generic men, losing their explicit relationship to the lieutenant.
- Minor, zero-based positions 122: Asking whether the warning is understood uses a literal and unnatural self-description.
- Minor, zero-based positions 245: The request to operate the lights retains an awkward bare plural noun.
- Minor, zero-based positions 254: Surprise that the device is real is expressed with a stiff literal noun phrase.
- Minor, zero-based positions 301, 302: The deception metaphor becomes a world pulled over the listener, followed by an unnatural construction about being blinded from reality.
- Minor, zero-based positions 319: The rabbit-hole metaphor becomes a clumsy relative clause about an actual rabbit going into a burrow.
- Minor, zero-based positions 328: The verb for swallowing a pill is replaced by the similar-looking verb for deploying something.
- Minor, zero-based positions 335: Certainty that a dream was real is softened to the dream merely seeming very real.
- Minor, zero-based positions 365: The definite article before answers is incorrect.
- Minor, zero-based positions 368: The contrast between what and when retains awkward English-like word order.
- Major, zero-based positions 491: Searching within the Matrix becomes searching for the Matrix itself, changing the object of the lifelong search.
- Minor, zero-based positions 607: Resistance to change is narrowed toward simple inactivity.
- Minor, zero-based positions 727: The cereal question has awkward word order and inconsistent verb-object definiteness.
- Minor, zero-based positions 733: The remembered-taste comparison uses a tangled relative-clause construction.
- Minor, zero-based positions 845: The emphatic vulgar idiom loses its vulgarity and part of the bodily image.
- Minor, zero-based positions 852: Asking for the missing qualification becomes asking why.
- Minor, zero-based positions 853: The clause about what the speaker will say has malformed verb government.
- Minor, zero-based positions 1096: The warning about using a human for a machine’s job is grammatically compressed into an unnatural work assignment.
- Minor, zero-based positions 1210: A formal security instruction uses conspicuously informal address and an awkward expression for removing carried metal.
- Major, zero-based positions 1213: A command to stop moving becomes a command to freeze from cold.
- Minor, zero-based positions 1329: Knowing the future uses an unnatural verb choice.
- Minor, zero-based positions 1333, 1334: The final promise contains awkward viewing language and incorrect object agreement.

## inception/mercury-2.5

Complete 180-cue sample from a fully delivered file. Most narrative content is recoverable, but recurrent malformed Hungarian and several consequential local mistranslations require substantial editing. This purposive sample is not an exhaustive review of all 1340 cues.
- Minor, zero-based positions 1: Volunteering to take a shift is rendered with an unnatural expression suggesting trying out the shift.
- Major, zero-based positions 2: The specific pleasure in watching the man becomes a general liking for looking around, weakening the teasing implication.
- Minor, zero-based positions 6, 197: Belief constructions use incorrect Hungarian government or clause forms.
- Minor, zero-based positions 14, 251: Profanity is assembled into unnatural Hungarian exclamations or forms of address.
- Minor, zero-based positions 61: The word for being awake is misspelled into a malformed word.
- Minor, zero-based positions 121: The alternative of seeking another job is expressed through an awkward doubled choice construction.
- Minor, zero-based positions 196: Wasting time with someone becomes an unnatural nominal construction.
- Minor, zero-based positions 203: The Hungarian noun for a bargain is misspelled.
- Major, zero-based positions 247: Having an implanted surveillance device is replaced by an unclear claim about being recorded or fixed in place.
- Minor, zero-based positions 249: The imperative urging movement is misspelled.
- Minor, zero-based positions 269: The rabbit-hole metaphor contains a malformed Hungarian noun.
- Minor, zero-based positions 301, 302: The deception metaphor is comprehensible but expressed with unnatural word order and preposition-equivalent construction.
- Minor, zero-based positions 315: The second-person form of believing is misspelled.
- Minor, zero-based positions 321: The sentence identifying truth as the sole offer lacks a needed linking construction.
- Minor, zero-based positions 328: The verb describing the swallowed pill is misspelled.
- Minor, zero-based positions 336: A hypothetical inability to awaken becomes a past-tense possibility, weakening the thought experiment.
- Minor, zero-based positions 364: An instruction to rest becomes an instruction to calm down.
- Minor, zero-based positions 368: The greater importance of the date becomes an absolute claim that everything concerns the date.
- Minor, zero-based positions 484, 488, 491: The fictional system name switches from its accented Hungarian spelling to the unaccented form.
- Minor, zero-based positions 487: The word for the prophesied return has a malformed suffix.
- Minor, zero-based positions 493: The explanation of past actions lacks the object and linking punctuation needed for a natural Hungarian sentence.
- Minor, zero-based positions 604: The construction for making people enemies uses the wrong case ending.
- Major, zero-based positions 606: Being disconnected from the simulation becomes being switched off, suggesting deactivation of the people themselves.
- Major, zero-based positions 608: The line adds dependence on the speakers as well as the system, changing the allegiance being explained, and uses a malformed case ending.
- Minor, zero-based positions 727: Both the eating verb and the cereal-name inflection are malformed.
- Minor, zero-based positions 728: A singular reply about never having eaten the food shifts to a plural addressee and an awkward case form.
- Minor, zero-based positions 730, 735: Reflective wondering is repeatedly rendered as being surprised rather than considering an uncertainty.
- Minor, zero-based positions 732: The verb for getting something wrong is misspelled.
- Minor, zero-based positions 733: The remembered taste comparison becomes a comparison of what the speaker considered the cereal to be, blurring the argument.
- Minor, zero-based positions 734: The Hungarian word for oatmeal is truncated.
- Minor, zero-based positions 845: The emphatic vulgar idiom is reduced to a neutral expression of totality.
- Minor, zero-based positions 848, 850, 853, 854, 856: Unclosed quotation punctuation and literal escaped newline text disrupt the examination dialogue.
- Minor, zero-based positions 855: Singular and plural forms disagree in the statement about having the gift.
- Major, zero-based positions 974: Regret about the outcome becomes an incomplete, unintelligible clause, losing the statement.
- Major, zero-based positions 975: The accusation addresses multiple killers rather than the single man being confronted.
- Minor, zero-based positions 1087: The threatened destruction omits the explicit object referring to the speakers, leaving an awkward incomplete verb phrase.
- Minor, zero-based positions 1090: The continuation comparing Zion with Morpheus loses the comparative case ending.
- Minor, zero-based positions 1091: The possibility that some remedy exists becomes an obligation to act.
- Minor, zero-based positions 1210: Removing carried metal objects is narrowed toward taking off worn objects, despite the examples being pocket contents.
- Major, zero-based positions 1211: An expression of alarm becomes a direct insult involving the listener’s mother.
- Minor, zero-based positions 1213: The command to remain still uses the wrong verb form.
- Major, zero-based positions 1332: Hanging up the telephone becomes taking it off the hook, reversing the action.
- Minor, zero-based positions 1333, 1334, 1339: The final declaration has recurring object-agreement and verb-government errors, although its broad intent remains recoverable.

## nvidia/nemotron-3.5-lightning:free

No translation text was available for any of the 180 common sample positions. Quality is unassessed, with null scores rather than a zero linguistic rating.

## google/gemini-2.5-flash-lite

Mostly understandable and complete, but recurring literal phrasing and grammar errors need editing; a few important word choices lose the intended sense.
- Minor, zero-based positions 14, 15, 22: The specific police lieutenant rank is rendered as inspector, changing the rank terminology.
- Minor, zero-based positions 1, 121, 122, 245, 370: Several phrases retain English constructions instead of natural Hungarian idioms for taking a shift, choosing an alternative, checking understanding or asking for light.
- Minor, zero-based positions 22, 38, 365, 491, 614, 615: Recurring article, case and subject-predicate agreement errors make otherwise recoverable dialogue sound unedited.
- Minor, zero-based positions 38: The informant is described with a predicate for truth rather than a natural statement that the person is genuine or reliable.
- Minor, zero-based positions 301, 302: The next cue's explanation is added to the current cue and then repeated at its original position.
- Minor, zero-based positions 336: A hypothetical inability to wake becomes a past-tense inability, weakening the thought experiment.
- Minor, zero-based positions 615: Membership in the opposing group is generalized into being an enemy, losing the source's group-membership formulation.
- Major, zero-based positions 855: An innate gift or ability is rendered like a possessed present, obscuring the point of the assessment of the protagonist.
- Minor, zero-based positions 1211: The exclamation is translated word for word into an unnatural Hungarian expression.
- Major, zero-based positions 1212: A call for police reinforcements is rendered as a request for reserves, obscuring the operational meaning.

## dots-studio/dots-3-note-preview:free

No translation text was available for any of the 180 common sample positions. Quality is unassessed, with null scores rather than a zero linguistic rating.

## google/gemini-3.5-flash-lite

Largely natural and faithful Hungarian, but conspicuous textual corruption, duplicated cue material and one incorrect action command prevent a clean quality assessment.
- Major, zero-based positions 1, 11, 129, 1329, 1332: Several malformed words and a repeated article remain; the short departure cue is sufficiently garbled to obscure the intended statement.
- Minor, zero-based positions 3: An intention to kill is changed into an obligation to kill, adding a necessity absent from the source.
- Minor, zero-based positions 1088, 1089, 1097, 1098: Two cues repeat the following cue's content inside the earlier cue, disrupting the intended cue boundary.
- Minor, zero-based positions 1096: A statement about work belonging to a machine becomes a broader statement about anything a machine can also do.
- Major, zero-based positions 1213: An order to stop moving becomes an order to raise a weapon, changing the action requested in the confrontation.

## google/gemini-3.1-flash-lite

Strong sampled meaning and natural Hungarian overall, with one clear polarity error and a localized block of visible escape tokens requiring correction.
- Minor, zero-based positions 1: A voluntary desire to take a shift is phrased as a feeling of obligation.
- Minor, zero-based positions 372: The year question uses awkward ordinal phrasing rather than ordinary Hungarian date wording.
- Minor, zero-based positions 724, 727, 728, 731, 733, 735: Literal escape tokens appear in place of intended line breaks in six sampled cues.
- Major, zero-based positions 732: The possibility that the machines got the flavor wrong is changed to the possibility that they got it right, reversing the local claim.
