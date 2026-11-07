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

## Results at a glance

| Question | Answer | Evidence |
|---|---|---|
| OpenCV prototype or vision LLM? | Vision LLM | 300 held-out photos: recall 0.59 vs 1.00, F1 0.48 vs 0.71 (round 1) |
| Does OpenCV help as a second opinion? | No, only as a fallback | its review flags were 40 of 42 wrong, caught nothing new (round 1) |
| Can the false alarms be cut? | Yes, by 18% | prompt v2 on 300 fresh photos: false alarms 45 to 37, decisions 251 to 258 right (round 2) |
| Is the model close to a person? | About as close as the dataset labels | on 50 photos labeled by a person: model 42-43/50, dataset 43/50 (round 3) |
| Does it work on video? | Yes, with smoothing | false alert events on safe clips 7 to 2; the violation clip alerts from second 1 (round 4) |
| When can its flags be trusted? | When heads are big enough | smallest head 96 px or more: false-alarm rate 0.02, precision 0.97 on fresh photos (round 5) |

**Go/no-go and engineering handoff:** [HANDOFF.md](HANDOFF.md) (recommendation,
API contract, camera requirement, monitoring and post-launch validation plan).

## Round 1: OpenCV vs vision LLM

On 300 held-out photos that were scored once, after all tuning was done
(prompt v1):

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
  detector this weak, its disagreements are mostly its own mistakes. That
  review rule is now off by default (`opencv_tripwire=false`).

## Round 2: cutting false alarms, proven on fresh photos

Most real model errors in round 1 were crowded, distant, dark or collage photos
where it reported a bare head that wasn't there. Prompt v2 (`app/vision_llm.py`,
`PROMPTS["v2"]`) only counts a head as bare when the top of it is clearly
visible and sends small, distant, dark or cut-off heads to `unclear_heads`
instead. It was written using the 100 tuning photos only (false alarms 14 to 8
there).

Round 1's held-out photos had already been looked at during the hand check, so
they can't prove the fix. A third set (`heldout2`, 300 photos the model and I
had never seen, drawn with the same seed after the others) was scored once with
both prompts:

| Prompt | Precision | Recall | F1 | False alarms | Decision right |
|---|---|---|---|---|---|
| v1 | 0.55 | 0.97 | 0.70 | 45 | 251/300 |
| v2 | **0.59** | 0.93 | **0.73** | **37** | **258/300** |
| OpenCV, for reference | 0.27 | 0.40 | 0.32 | 62 | 198/300 |

Recall dropped from 0.97 to 0.93, so I checked all 4 photos v2 "missed". In 3
of them it reported heads it couldn't judge (a collage of masked faces, a crowd
of 15 masked commuters, a face covered by hands), and unclear heads send the
photo to a person. Counting what actually reaches **pass with no review**:

| Prompt | Violations that passed with no review | Photos sent to review only for unclear heads |
|---|---|---|
| v1 | 1 | 4 |
| v2 | 1 | 8 |

So v2 cut false alarms by 18% for the cost of 4 more quick reviews per 300
photos, with the same single silent miss (a bare-headed driver inside a truck
at night). v2 is the default.

These photos also showed that the dataset includes pictures that aren't
worksites at all (selfies with face masks, commuter crowds), which the labels
treat as "no hard hat". A real deployment would see camera footage of its own
site, which is a reason to build a site-specific test set before trusting any
of these numbers in production.

## Round 3: how close is it to a person?

Navyatha labeled 50 photos from `heldout2` herself (20 the dataset marks as
violations, 30 as compliant, drawn with a fixed seed) in a small labeling page
(`bench/human/index.html`), without seeing the dataset labels or the model's
answers. `bench/human_compare.py` compares the three:

| Pair | Agreed |
|---|---|
| Person vs dataset labels | 43/50 |
| Vision LLM v1 vs person | 43/50 |
| Vision LLM v2 vs person | 42/50 |

The model agrees with a person about as often as the dataset's own labels do.
Where the person and the dataset disagreed, 5 of 7 were the person seeing a
bare head the labels didn't mark, the same pattern the hand check found. 50
photos and one labeler is a small sample, so this is "about human-level on this
set", not a general claim.

## Round 4: video

Single photos can't show whether a violation lasts. `app/video.py` adds
`SmoothedAlerts`: sample a clip at 1 frame per second and raise an alert only
when a violation is reported in 2 of the last 3 frames.

Test footage: 10 free Pexels clips (5 to 29 s, 170 frames), labeled by eye per
clip in `bench/video_labels.json`: 5 where everyone wears a hard hat, 1 with a
worker in a baseball cap the whole time, and 4 "hard" clips (drone shots,
distant crowds, overhead views) where heads are too small to judge.

| Clips | Frames | Separate alerts, per frame | Separate alerts, smoothed |
|---|---|---|---|
| 5 compliant (every alert is false) | 66 | 7 | **2** |
| 1 violation | 17 | 2 (first at 0 s) | 1 continuous (from 1 s) |
| 4 hard | 87 | 8 | 4 |

- Smoothing removed 5 of the 7 false alert events on safe footage and turned
  the violation into one continuous alert after 1 second.
- Both remaining false alerts came from heads cut off by the frame edge in
  close-up footage, which the model counted as bare even though prompt v2 says
  to mark them unclear.
- On the "hard" clips the model still raised alerts (drone and overhead
  views). The system should not be pointed at footage like that; a fixed
  camera angle is a requirement, not a nice-to-have.
- One frame took a median 6.1 s on Azure, so this is a batch review of
  recorded clips at 1 frame per second, not live video.

Only one clip contains a violation, so the video numbers show the effect of
smoothing, not a recall figure.

## Round 5: where it fails (error analysis)

`bench/error_analysis.py` joins every result with the dataset's head boxes in
pandas and groups photos by crowd size and by the size of the smallest
labeled head:

| Smallest head | Photos (fresh set) | False-alarm rate, v2 | Precision, v2 | False-alarm rate, first held-out set (v1) |
|---|---|---|---|---|
| under 24 px | 34 | 0.23 | 0.30 | 0.41 |
| 24-48 px | 79 | 0.24 | 0.26 | 0.21 |
| 48-96 px | 99 | 0.14 | 0.54 | 0.20 |
| 96 px or more | 81 | **0.02** | **0.97** | 0.07 |

The same pattern shows up on both held-out sets: once heads are about 96 px or
larger in a 640 px photo, false alarms almost disappear. Crowd size matters
less than head size. That becomes a deployment rule in
[HANDOFF.md](HANDOFF.md): mount cameras so heads are at least that big, and
don't send drone or distant-crowd footage. Head size here comes from the
labels, so it describes the footage, not something the system measures at run
time.

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
            human review when: the LLM marks unclear heads, or the LLM call
            fails or is refused (then the OpenCV result is used)
            optional opencv_tripwire: also review when OpenCV sees a violation
            the LLM doesn't (off by default, see round 1)
      --> video mode (app/video.py)
            1 frame per second, alert when 2 of the last 3 frames show a violation
      --> optional AWS S3 log (app/aws.py)
      --> FastAPI: POST /inspect
```

**Why only one direction of disagreement goes to review.** When the LLM
reports a violation, the decision already flags or escalates it, so asking for
a second opinion there adds review work without catching anything new. On the
tuning set this change cut reviews from 36 to 16 of 100 photos with no
violations missed. (On held-out photos the remaining rule still didn't earn its
keep, so it is now off by default; see round 1.)

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

- 640x640 still photos from one public dataset. Most are construction sites,
  but some are not worksites at all. No night-time or CCTV-angle test set.
- Video: 10 short stock clips, only one with a violation, sampled at 1 frame
  per second. Not live.
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
python -m pytest -q                      # 32 offline tests
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
python bench/run_eval.py --set heldout2 --prompt v1
python bench/run_eval.py --set heldout2 --prompt v2
python bench/human_compare.py --prompt v2
python bench/error_analysis.py --set heldout2 --prompt v2
```

Video clips (Pexels, free license, 640x360 versions) go in `data/videos/`, named
by their Pexels id: 5434220, 10810473, 10810476, 11355903, 11798561, 12098511,
13921040, 14346212, 16393893, 19832500 (`https://www.pexels.com/video/<id>/`).
Run `python bench/sample_frames.py` (1 frame per second), then
run `python bench/video_eval.py --prompt v2`.

Vision LLM replies are cached in `bench/cache/`, so reruns cost nothing.

## Credit

Photos and labels: Hard Hats Dataset, Roboflow Universe Projects, Roboflow
Universe (2022), https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5,
licensed CC BY 4.0. Mirrored on Hugging Face as `keremberke/hard-hat-detection`.
Face detector: YuNet from the OpenCV Zoo. Video clips: Pexels contributors,
used under the Pexels license (ids listed above).
