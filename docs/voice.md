# Voice commands: how they work and how well

What to say and how to use it is in [webots/README.md](../webots/README.md#voice-commands). This page
has how a recording is recognised, how the model for the rider's own words was chosen, and how well
exact commands are understood.

## How a recording is recognised

Each recording is recognised twice by Vosk's small US English model. One pass listens only for the
exact phrases that mean something right now (the commands, the places read, the markers up), the
other writes down whatever was said. An exact command is carried out at once when it is heard with at
least 70% confidence and the words written down are not a longer sentence around it or a "don't"
("I don't want to go to pharmacy"). Everything else goes to the model with the screen: the places
read with their wings, the numbered list, the markers, what is picked and who is driving. Its answer
is used only if the action is a real one and the place, marker or number is on the screen. Stopping
and paging are done at once, and a move is asked about first. The microphone keeps the 0.3 s before
**K** is pressed, so the first word is not cut, and all of this runs in the background, so the trip
does not wait for it.

## Choosing the model

Nothing is trained: the places come with every request, so the same model works in any building.
`python webots/interpret_eval.py all` chose it from every chat model on the gateway with 231 requests
in riders' own words (`voice_requests.jsonl`: other ways to say each command, places by name and by
what they are for, markers, list numbers, places not read yet, unclear requests, small talk and
typical mishearings), each with the screen it is said on and the answers that are right. Half of each
kind was used to choose and half to test. The rule was fixed before any model was tried: at most 1%
wrong moves (the chair sent somewhere it was not asked), at least 99% of answers readable, 90% of
answers within 2.5 s, then the most right answers.

All 11 chat models were screened on 40 requests and the 4 best were compared on the 117 validation
requests, twice (a slow answer, over 3 s, counts as none, as it does in the chair):

| Model | Right | Wrong moves | No answer | 90% within | US$ per 1,000 | Rule |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-120b-bedrock | 92% | 1 of 234 | 0% | 1.27 s | 0.18 | chosen |
| gpt-oss-20b-bedrock | 94% | 0 | 0% | 1.26 s | 0.11 | 3 answers empty or cut off (98.7% readable) |
| scx-gemma-4-31B-it | 89% | 0 | 9% | 2.27 s | 0.25 | passes |
| scx-gpt-oss-120b | 87% | 0 | 8% | 2.45 s | 0.22 | passes |
| exact phrases only | 29% | 0 | 0% | 0 s | 0 | |
| trained classifier (naive Bayes, as tech-5) | 53% | 2 of 117 | 0% | 0 s | 0 | fails |

On the 114 test requests, which no model had seen, gpt-oss-120b-bedrock was right on 92% with no
wrong moves (exact phrases only: 25%, the trained classifier: 51% with 2 wrong moves). All 9 misses
did nothing. Read out by the Mac's voice and recognised as above, it was right on 88% of them (exact
phrases alone: 32%), with 1.33 s (median) from letting go of **K** to the answer. 3 answers were
wrong because the words were misheard ("the imaging department" written down as "the emerging
department", then Emergency): all 3 were the model's guesses, so the chair would have asked before
moving.

## Exact commands without the model

`python webots/voice_eval.py run --tts` reads 85 phrases (the commands, every hospital place, 8
markers, 5 numbers and 15 things that are not commands) in espeak-ng's 6 English voices, with every
hospital place and marker listened for at once (177 phrases). Pink noise is mixed in at 20 dB (a
quiet ward), 10 dB (a busy corridor) and 5 dB:

| Noise | Understood | Wrong command | Stopped instead | Said again | Stops that stopped | Not commands, acted on |
| --- | --- | --- | --- | --- | --- | --- |
| quiet | 88% | 0 | 1% | 11% | 18 of 18 | 0 of 90 |
| 20 dB | 86% | 0 | 1% | 13% | 18 of 18 | 0 of 90 |
| 10 dB | 73% | 0 | 1% | 25% | 18 of 18 | 0 of 90 |
| 5 dB | 63% | 0 | 0% | 37% | 18 of 18 | 1 of 90 (taken as a stop, which is safe) |

None of the 1,680 commands sent the chair somewhere it was not asked to go.

Checking the exact pass against the words written down turns away about 8% of good commands with
these voices (96% understood in quiet with the exact pass alone), because the second pass often
mishears espeak. That is the price of never acting on "go to pharmacy" when the rider said "I don't
want to go to pharmacy". The phrases most often said again are Ward 4A's: an A at the end
("Ward 4A") is heard as "a ray" or "me". With the model on, these go to it instead (and the chair
asks), and without it "go to number" with the number on the list works.

Limits: computer voices are clearer than people, so these numbers are a best case. Real voices are
measured the same way after `python webots/voice_eval.py record <name>`, which records each phrase
into `data/voice/` (kept out of Git). English only, and own words need the internet. Holding **K**
needs a keyboard, where a real chair would have a button on its joystick.

## References

- **Vosk** offline speech recognition, with its small US English model (`vosk-model-small-en-us-0.15`), [alphacephei.com/vosk](https://alphacephei.com/vosk).
- **gpt-oss-120b**, the model chosen for own words, reached through the course gateway (`gpt-oss-120b-bedrock`). OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card.* arXiv:2508.10925, 2025.
- **espeak-ng**, the computer voices for the spoken commands, [github.com/espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng).
- **sounddevice**, the microphone, [python-sounddevice.readthedocs.io](https://python-sounddevice.readthedocs.io).
