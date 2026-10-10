# Evaluation

All test data is synthetic. No real names, emails or credentials are used.

## Highlights

- **Emails 18/18, credentials 16/16** (AWS, GitHub, Google and Stripe keys, private
  keys, passwords), **client terms 36/38** masked on the test screens.
- **Person names 97% masked** with GLiNER and session memory, up from 6% with rules
  alone. Leaks drop from 112 to 5.
- **Names in text: 92/92** on 339 labelled lines in English and Polish, with only
  9/219 harmless lines (UI text, public brands) masked by mistake.
- **While scrolling, names stay masked:** 112 of 118 names are masked in every
  scrolled variant, and missed name-frames drop from 8.9% to 1.9%.

## Screen eval

`python -m safe_share.eval_screens` runs each screenshot through OCR, detection and
masking, then runs OCR again on the masked image. A **leak** is a known sensitive string
that Vision can still read after masking. **Harmless lines masked** counts lines with
no sensitive text that were covered anyway.

Test screens: 8 scenes (Teams chat, Outlook in English and Polish, code editor, API
keys page, Jira board, CRM dashboard, credentials page), light and dark, 17 screenshots,
118 person names. The screenshots stay local; the ground truth is in
`fixtures/detect/screens.json` (credential values in a local, uncommitted file).

| Detection | Person names masked | Leaks | Harmless lines masked | Detection time per frame (median) |
|---|---|---|---|---|
| Rules only | 7/118 (6%) | 112 | 2/645 | 0 ms |
| + GLiNER | 108/118 (92%) | 11 | 15/645 | 87 ms |
| **+ GLiNER + session memory** | **114/118 (97%)** | **5** | 15/645 | 86 ms |

Emails 18/18, credentials 16/16 and client terms 36/38 in all three. The 5 remaining
leaks: two short first names the model does not recognize on these screens (2
occurrences each) and one client term written inside a code identifier.

## Stability while scrolling

When a pane scrolls, the text around a name changes, and the model can recognize a name
on one frame and miss it on the next. `eval_screens --context` simulates this: each
screenshot is run 11 times, once as is and 10 times with one pane shifted by up to 6
lines, and every person name is checked in every run (118 names × 11 runs = 1,298
name-frames).

| | Names masked in every run | Missed name-frames |
|---|---|---|
| GLiNER | 99/118 | 115/1,298 (8.9%) |
| **GLiNER + session memory** | **112/118** | **25/1,298 (1.9%)** |

Session memory remembers each name the model has found and masks it on every later
frame, so a single miss no longer shows it. The remaining misses come from three names
the model recognizes late or not at all on their screen; until the model finds a name
once, there is nothing to remember.

This test starts every screenshot with empty memory. In the app, memory carries over
the whole session, so a name seen on any earlier screen stays masked. In a live test
scrolling the CRM dashboard, no name became visible.

## Text eval

`python -m safe_share.eval_detect fixtures/detect/names.json --detector union`: 339
labelled lines in English and Polish, including UI text, public brands and name-like
words as negatives.

| | Person recall | Harmless lines masked |
|---|---|---|
| GLiNER | 91/92 | 7/219 |
| **GLiNER + session memory** | **92/92** | 9/219 |

## About these results

- The screenshots were also used during development, so the numbers describe these
  test screens, not every possible screen.
- Zero leaks on a test screen is a test result, not a guarantee.
- Session memory also remembers a false positive, so a word wrongly taken for a name
  stays masked for the session.
