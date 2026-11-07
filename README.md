# HazardLoop

See it, decide, act: hard-hat checks on worksite photos, with the evidence for
which approach to trust.

HazardLoop takes a worksite photo, works out who is and isn't wearing a hard
hat, and decides what happens next:

- **pass**: everyone visible is wearing one
- **flag**: send the photo to a person
- **escalate**: 3 or more people without hard hats, or at least half of those visible

It started as a quick OpenCV prototype for the OpenCV AI Competition 2026
(Agentic Vision track). This version asks the question a prototype has to
answer before anyone builds on it: **does it actually work on real photos, and
which approach should we use?**

## The short answer

On 300 held-out photos that were scored once, after all tuning was done:

| Method | Precision | Recall | F1 | Decision right | Head-count error |
|---|---|---|---|---|---|
| OpenCV (original prototype) | 0.40 | 0.59 | 0.48 | 219/300 | 1.46 per photo |
| Vision LLM (gpt-4.1-mini) | 0.55 | **1.00** | 0.71 | **246/300** | 0.70 per photo |

After checking every vision LLM false alarm by hand (14 of its 48 "false
alarms" were bare heads the labels skipped; see below):

| Method | Precision | Recall | F1 |
|---|---|---|---|
| OpenCV | 0.52 | 0.62 | 0.56 |
| Vision LLM | **0.68** | **1.00** | **0.81** |

The vision LLM caught all 73 photos with a violation. OpenCV missed 28 of them.

- The original OpenCV approach (find faces, look for hat-coloured pixels above
  them) is not usable. It misses most violations and raises false alarms at the
  same time. Changing its threshold doesn't fix that (see below).
- The vision LLM catches violations reliably. Its weakness is false alarms, and
  a hand check showed that many of those are bare heads the dataset never
  labeled.
- **Recommendation: let the vision LLM decide, and use OpenCV only as the
  fallback when the LLM call fails.** The cross-check I designed on the tuning
  set (send a photo to a person when OpenCV sees a violation the LLM doesn't)
  did not pay off on held-out photos: it sent 42 of 300 photos to review, 40 of
  them had no violation, and it caught nothing the LLM had missed. With a
  detector this weak, its disagreements are mostly its own mistakes.

## How it works

```
photo --> OpenCV 5 stage (app/detector.py)
            YuNet face detector + hard-hat colour check above each face
      --> vision LLM stage (app/vision_llm.py)
            gpt-4.1-mini, strict JSON schema:
            with_hardhat, without_hardhat, unclear_heads, needs_human_review
            reply validated (whole numbers 0-50, no extra fields),
            one retry with the errors listed, else no counts + human review
      --> agent decision (app/agent.py, cross_check)
            LLM counts drive pass / flag / escalate
            human review when: the LLM marks unclear heads, OpenCV sees a
            violation the LLM doesn't, or the LLM call fails or is refused
      --> optional AWS S3 log (app/aws.py)
      --> FastAPI: POST /inspect
```

**Why only one direction of disagreement goes to review.** When the LLM
reports a violation, the decision already flags or escalates it, so asking for
a second opinion there adds review work without catching anything new. On the
tuning set this change cut reviews from 36 to 16 of 100 photos with no
violations missed. (On held-out photos the remaining rule still didn't earn its
keep; see the recommendation above.)

**Failure handling.** One held-out photo was refused by Azure's content-safety
filter. The first run crashed on it. The API now treats a refused or failed
call like any other bad reading: no counts, the OpenCV result is used, and the
photo goes to a person. That was the 1 fallback in the held-out run, and there's
a test for it.

## How it was tested

**Data.** The test split of the Hard Hats dataset (Roboflow Universe, CC BY
4.0): 2,001 worksite photos at 640x640, every head boxed as `hardhat` or
`no-hardhat`. `bench/make_sets.py` draws two disjoint sets with a fixed seed:

| Set | Photos | With a violation | Used for |
|---|---|---|---|
| tuning | 100 | 18 | looked at while changing the prompt, threshold and review rule |
| held-out | 300 | 59 | scored once at the end |

**What counts as right.** The positive class is "this photo has at least one
person without a hard hat". Precision is how many alarms were real; recall is
how many real violations were caught. Decision accuracy compares
pass/flag/escalate with the same rules applied to the true counts.

**Tuning set results (100 photos):**

| Method | Precision | Recall | F1 | Decision right | Head-count error |
|---|---|---|---|---|---|
| OpenCV (original) | 0.27 | 0.39 | 0.32 | 70/100 | 1.19 per photo |
| Vision LLM | 0.56 | 1.00 | 0.72 | 86/100 | 0.64 per photo |

**OpenCV threshold sweep (tuning set).** The colour check calls a head
"hatted" when more than 15% of the region above it is hard-hat coloured.
Sweeping that from 2% to 30% never got F1 above 0.38, so the threshold stayed
at 15%. The real limit is the face detector: it found 179 of 266 heads on the
tuning set, and it can't see heads turned away or under a helmet brim.

## Checking the labels by hand

A false alarm only counts against the model if the label is right. Every
photo where the vision LLM disagreed with the labels was opened and checked by
eye. The rule: **a label is only overruled when a bare head (or a cap that is
clearly not a hard hat) is plainly visible. Anything unclear counts against the
model.** Verdicts and notes are in `bench/audit_tuning.json` and
`bench/audit_heldout.json`.

| Set | LLM false alarms checked | Labels that missed a bare head | Still counted as model errors |
|---|---|---|---|
| tuning | 14 | 5 | 9 |
| held-out | 48 | 14 | 34 |

Typical label misses: officials in suits walking a site with bare heads,
bystanders behind a rescue team, workers in baseball caps or a knit beanie.
Typical real model errors: crowded or distant scenes where it counts one
helmet-less head that isn't there, and collages of several photos.

Only the vision LLM's disagreements were checked, so OpenCV false alarms on
other photos keep their original labels. `bench/audited_scores.py` applies the
corrections and writes `bench/audited_<set>.json`.

**What this changes.** The labels only mark some heads in each photo, so any
detector scored against them looks worse than it is on false alarms. A
production version would need a relabeled test set before a precision target
could be trusted.

## Limits

- 640x640 still photos from one public dataset, mostly construction sites. No
  video yet, and no night-time or CCTV-angle test set.
- The labels are per head and incomplete (see above), so the "hand-checked"
  numbers rest on one reviewer's judgment, applied with a conservative rule.
- The vision LLM takes about 6 s per photo on Azure and needs a network
  connection; OpenCV runs locally in milliseconds. That trade-off matters for
  live video.
- The low Azure request quota meant the benchmark ran one photo at a time
  (median 6.0 s per photo).
- The OpenCV stage only adds value as a tripwire for review. A trained
  hard-hat detector would be the next thing to compare.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                      # 26 offline tests
uvicorn app.main:app --reload
curl -X POST -F "file=@photo.jpg" "http://127.0.0.1:8000/inspect"              # cross-check
curl -X POST -F "file=@photo.jpg" "http://127.0.0.1:8000/inspect?mode=opencv"  # OpenCV only
```

The vision LLM needs `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_DEPLOYMENT` (a vision-capable model) and optionally
`AZURE_OPENAI_API_VERSION` in `.env`. Without them `/inspect` falls back to
OpenCV and marks every photo for review.

Set `HAZARDLOOP_S3_BUCKET` (plus standard AWS credentials) to log inspections
to S3. Without it, logging is skipped.

**Reproduce the benchmark:**

```bash
mkdir -p data/raw && curl -L -o data/raw/test.zip \
  https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data/test.zip
unzip -q data/raw/test.zip -d data/raw/test
python bench/make_sets.py
python bench/run_eval.py --set tuning
python bench/run_eval.py --set heldout
python bench/contact_sheet.py --set heldout --method vision_llm   # sheets for the hand check
```

Vision LLM replies are cached in `bench/cache/`, so reruns cost nothing.

## Credit

Photos and labels: Hard Hats Dataset, Roboflow Universe Projects, Roboflow
Universe (2022), https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5,
licensed CC BY 4.0. Mirrored on Hugging Face as `keremberke/hard-hat-detection`.
Face detector: YuNet from the OpenCV Zoo.
