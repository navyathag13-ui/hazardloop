# HazardLoop: recommendation and engineering handoff

What this document is: the go/no-go call on the proof of concept, and what an
engineering team would need to build it for real. Every number links back to a
file in `bench/`; the README explains how each one was measured.

## 1. Recommendation

**Go, as a review assistant, not as an automatic enforcer, and only on
footage that meets the camera requirement in section 4.**

| Option | Verdict | Why |
|---|---|---|
| OpenCV prototype (face detector + hat-colour check) | No-go | recall 0.40-0.59 and F1 0.32-0.48 on two held-out sets; threshold changes never beat F1 0.38 on tuning photos |
| Vision LLM (gpt-4.1-mini, prompt v2) deciding alone | Go, with human review of flags | 300 fresh photos: recall 0.93, precision 0.59, 258/300 decisions right; 1 violation passed without review |
| Vision LLM + OpenCV as a second opinion | No-go | 40 of its 42 review flags were wrong and it caught nothing new (round 1) |
| Vision LLM on video with 2-of-3-frame smoothing | Go for recorded clips | false alert events on safe clips 7 to 2; not fast enough for live use (about 6 s per frame) |

The model is about as consistent with a person as the dataset labels are
(42-43 of 50 vs 43 of 50), so its flags are worth a reviewer's time. Its
precision is not high enough to act on a flag without a person.

## 2. What to build

```
camera / clip upload
  -> frame sampler (1 fps)                         app/video.py, bench/sample_frames.py
  -> vision LLM, strict JSON schema                app/vision_llm.py (prompt v2)
       validation, 1 retry, else human review
       refused or failed call -> OpenCV result + human review
  -> decision rules                                app/agent.py
       pass / flag / escalate; unclear heads -> review
  -> smoothing for video (2 of last 3 frames)      app/video.py
  -> review queue + audit log                      (not built; see section 5)
```

**API contract (today):** `POST /inspect` with an image returns
`people` (OpenCV boxes), `vision_llm.counts` (`with_hardhat`, `without_hardhat`,
`unclear_heads`, `needs_human_review`), `vision_llm.meta` (validated, attempts,
fallback_used, errors) and `decision` (`action`, `reason`, `needs_human_review`,
`source`). `?mode=opencv` skips the LLM. `?opencv_tripwire=true` turns the
round 1 second-opinion rule back on.

**Decision rules (`app/agent.py`):** escalate at 3 or more people without hard
hats, or at least half of those visible; flag at any smaller number; pass when
none. Any unclear head, failed validation or refused call means a person
reviews the photo.

## 3. Cost and speed

- About 6 s per photo (median) on Azure OpenAI gpt-4.1-mini with the request
  quota used here; the benchmark had to run one photo at a time.
- 1 frame per second of video therefore needs about 6 parallel requests per
  camera to keep up. That's a quota and cost question to settle before any
  live use.
- OpenCV runs locally in milliseconds but isn't accurate enough to use alone.

## 4. Where it can be trusted (error analysis)

`bench/error_analysis.py` (pandas) splits results by the size of the smallest
labeled head in the photo:

| Smallest head | Fresh set (v2): false-alarm rate | First held-out set (v1): false-alarm rate |
|---|---|---|
| under 24 px | 0.23 | 0.41 |
| 24-48 px | 0.24 | 0.21 |
| 48-96 px | 0.14 | 0.20 |
| 96 px or more | **0.02** (precision 0.97) | **0.07** (precision 0.84) |

**Camera requirement:** mount cameras so a worker's head is at least about 96
px tall in a 640 px frame (roughly 15% of the frame height). With heads that
size the model's flags were right 97% of the time on fresh photos. Below about
48 px, roughly 1 in 4 safe photos raised a false alarm.

Drone, overhead and distant-crowd footage should not be sent at all: on those
clips the model still raised alerts (round 4).

## 5. Not built yet

- Review queue and audit log (who confirmed or dismissed each flag).
- Monitoring: log every decision with its counts, prompt version and latency;
  weekly, pull a random sample of reviewed flags and recompute precision and
  the false-alarm rate against section 4. Alert if the false-alarm rate on
  compliant footage rises above about 0.10 for a week.
- A site-specific labeled test set (the public dataset includes photos that
  aren't worksites, and its labels skip some heads).
- Live video. The current design reviews recorded clips.

## 6. Post-launch validation plan

1. Before launch: label 300 frames from the site's own cameras, score once,
   and compare with section 4. Go only if recall stays at 0.9 or above and the
   false-alarm rate on qualifying footage stays at or below 0.10.
2. First month: reviewers' confirm/dismiss decisions become new labels. Compare
   weekly precision with the POC numbers above.
3. Human parity check: have two reviewers label the same 50 frames and compare
   model-vs-reviewer agreement with reviewer-vs-reviewer agreement (the POC
   only had one reviewer: 42-43 of 50 vs the dataset's 43 of 50).

## 7. Responsible use

- The system counts hard hats. It does not identify people, and it should stay
  that way: no face recognition, no per-person tracking.
- Photos go to a third-party model API. Sites must agree to that, and footage
  should not be kept longer than the review needs. S3 logging is off unless a
  bucket is configured.
- A flag is a prompt for a person to look, never a disciplinary record on its
  own.
- One photo in testing was refused by the provider's content filter; refusals
  go to human review instead of failing (tested).

## 8. Open risks

- One labeler for the human comparison, one violation clip for video.
- Prompt v2 was tuned on 100 photos; a site with different gear (for example
  bump caps or hoods under helmets) needs its own check.
- Model or prompt changes must be re-scored on a held-out set before release;
  `bench/run_eval.py` caches replies per prompt version so this is cheap.
