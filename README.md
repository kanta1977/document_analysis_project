# What is a TL;DR, actually?

### A Reddit TL;DR is expected to be a summary. We measure how far it sits from an AI summary, and how that varies by community

_Group members: Jorge Lastra, Kanta Ito

---

## Introduction

When a Reddit post ends with **"TL;DR: …"** line, that line is usually
treated as a summary. The Webis-TLDR-17 corpus (Völske et al., 2017) was built
on this assumption and is widely used as human "ground-truth" summaries, as in a landmark RLHF paper from OpenAI in which TL;DR is used as human made summaries.

Working with the corpus we found that many TL;DRs are not exactly summaries: some are
jokes, questions, or replies. Our first instinct was to classify them, and we
built a small rule-based typology (summary, question, reaction, advice). It
worked as a rough description, but we could not validate the categories without
manual labels, and forcing every TL;DR into one box hid more than it showed.
So we changed the question. Instead of asking *what type is this TL;DR*, we ask
*how far is it from a plain summary*, and to make "plain summary" concrete we
generate one for every post with a language model. This led us to our stance:

> **We treat the TL;DR as a text of unknown type, and measure how far it sits
> from a plain summary, using an AI summary of the same post as a fixed
> reference point. Then we ask how that distance differs across communities.**

The AI summary is a **reference point**, not a gold standard. We describe
*distance* from it without claiming it to be correct or that a distant TL;DR is
"wrong".

We look at four things, on the human TL;DR and on the AI summary alike:

1. **Does the first person survive?** voice.
2. **Surface signals** how often a TL;DR contains a question mark or
   advice-type words. We report *rates only* and do not label a TL;DR a
   "question"; we say it *may* read as one.
3. **Semantic distance** cosine similarity between post and TL;DR.
4. **Keyword containment** do the post's key words appear in the TL;DR.

## Dataset

We use the **Webis-TLDR-17** corpus (Völske et al., 2017): 3,848,330 Reddit
posts from 29,651 subreddits, each with the author's own TL;DR
([dataset](https://huggingface.co/datasets/webis/tldr-17),
[paper](https://aclanthology.org/W17-4508.pdf)). We group nine subreddits into
three buckets and draw a stratified sample of **39,859 posts** (up to ~5,000 per
subreddit; 100–600 content words; TL;DR ≥ 10 words).

| Bucket | Subreddits |
|--------|------------|
| political | politics, PoliticalDiscussion, worldnews |
| mental_health | depression, offmychest, Anxiety |
| advice | legaladvice, personalfinance, relationships |

For the AI summaries we take a **stratified subsample of ~200 posts per
subreddit (~1,800 total)** and generate one summary each. All results are
aggregates; no usernames are shown; mental-health posts are never quoted.

## Methods

### Setup

Plain Python, CPU-only for items 1–4 (the AI summary is made with Gemma 3).

- Python 3.11; dependencies pinned in [`code/requirements.txt`](code/requirements.txt).
- Recreate and test:

```bash
conda create --name tldr python=3.11
conda activate tldr
pip install -r code/requirements.txt
pytest code/tests -q
```

Run order and file-by-file flow are in [`code/RUNNING.md`](code/RUNNING.md) and
[`code/PIPELINE.md`](code/PIPELINE.md); a full walkthrough is in
[`code/GUIDE.md`](code/GUIDE.md).

### Experiments

**Preprocessing.** `01_inventory.py` counts posts per subreddit; `02_sample.py`
filters and draws the stratified sample into `sample.jsonl`.

**Human features.** `03_features.py` computes one row per post
(`src/tldr_audit/features.py`): first-person density, the surface flags
(`has_question_mark`, `has_advice_marker`, `has_second_person`),
`summary_novelty`, `keyword_containment`, sentiment, and compression.

**AI summaries.** `04_ai_baseline.py` sends each subsampled post body to
**Gemma 3 27B**, served with vLLM on the University of Konstanz computational
cluster, at **temperature 0** (deterministic output) with one neutral prompt,
identical for every subreddit:

> *System:* "You are a neutral summarization baseline. Write a faithful, concise
> summary of a single Reddit post. Add no opinions, advice, questions, or jokes,
> and no information not in the post. Output only the summary text — no preamble,
> no quotation marks, no labels."
> *User:* "Summarize the following Reddit post in one or two sentences. Do not
> add anything that is not in the post."

We deliberately do **not** tell the model to keep the first person, so the
model's own default is what we measure (item 1). No external API is involved;
the model runs entirely on university infrastructure.

**Semantic distance & comparison.** `06_semantic_distance.py` computes
cosine(post, TL;DR) with Sentence-BERT, or TF-IDF cosine as a no-download
fallback so it runs anywhere. `07_human_vs_ai.py` measures items 1–4 on the
human TL;DR and the AI summary for each post, then aggregates **per subreddit**
and **overall** (the overall value is the reference line).

## Results and Discussion

_The community figures (compression, sentiment, summariness) are built from the
full human sample of 39,859 posts. The human-vs-AI comparison (items 1–4) uses
the 1,800-post subsample with AI summaries; the full table is in
`results/tables/human_vs_ai_by_subreddit.csv`._

Before the four items, one basic fact about the corpus: people keep only a
small share of their own words. The median TL;DR is 8–13% of the post's
length, and this is nearly identical across all three community types.
Whatever people do differently in a TL;DR, it is not about how much they cut.

![Compression by bucket](figures/01_compression_by_bucket.png)

### 1. Does the first person survive?

Summarizing generally thins out the first person: an AI summary tends to shift
"I can't pay my rent" toward "the poster has a payment issue". Yet in Reddit's
advice and mental-health communities the TL;DR keeps "I" at nearly the density
of the post. So the survival of the first person is not a property of
summarizing, it is a property of Reddit's self-narration culture, strongest
where people tell their own stories and near-absent in political writing.

![First-person density](figures/03_first_person_survival.png)

### 2. Surface signals (reported as rates, not labels)

We count how often a TL;DR contains a question mark or advice-type words
(should / need to / avoid …). We stop at the rate: a question mark **may**
indicate the TL;DR is really a question rather than a summary, but we do not
label it as such. These rates differ by community, political comment threads
carry far more "reaction"-like markers than advice self-posts, and reading the
rates, rather than forcing a category, keeps the interpretation open.

### 3. Semantic distance, and 4. keyword containment

How far is the TL;DR from the post in meaning (cosine, item 3), and does it
reuse the post's key words (containment, item 4)? The two together separate
cases that either one alone confuses:

- high containment + high cosine → an extractive summary
- **low containment + high cosine → a paraphrase** (different words, same meaning)
- **low containment + low cosine → genuinely diverged** (not really a summary)

Containment alone only raises a *possibility* ("few key words reused"); cosine
tells paraphrase apart from real divergence. We show both on one map. Across
communities, only ~6% of TL;DRs are clearly extractive; most rewrite the post,
and political TL;DRs sit farthest from the post.

![Containment vs cosine map](figures/05_containment_vs_cosine.png)

Each dot is one post. Most human TL;DRs cluster in the paraphrase region:
they reuse few of the post's key words but stay close in meaning. The
political posts (orange) spread furthest into the diverged corner, where both
the words and the meaning move away from the post.

![Summariness, split by post type](figures/04b_summariness_decomposed.png)

Splitting self-posts from comments matters: comment TL;DRs barely reuse the post
because they are replies, so political communities looked extreme mainly because
they are ~92% comments. The community gap shrinks once you separate the two,
but does not vanish.

### The AI reference point makes the numbers legible

On its own, "the median TL;DR keeps ~55% new words" invites the question *is
that a lot?* Placing the AI summary beside it answers this: the AI summary of
the same post stays lexically and semantically closer to it, so the human TL;DR
is measurably farther from a plain summary, and that distance is uneven across
communities. Because the AI almost never produces a question, joke, or advice
line, the human side's surface-signal rates make the human "non-summariness"
visible by contrast.

![Human TL;DR vs AI summary by community](figures/07_human_vs_ai_by_subreddit.png)

The five panels place the human TL;DR (dark) beside the AI summary (orange)
for each community. The gaps are large and consistent. First person: 5.5% of
human TL;DR words vs 0.1% in the AI summary, a difference of more than 50×.
Question marks: 22.6% of human TL;DRs contain one, the AI never does; in
r/legaladvice the human rate reaches 47%, so nearly half of those TL;DRs read
as questions rather than summaries. Meanwhile the AI summary is both
semantically closer to the post (cosine 0.80 vs 0.74) and reuses more of its
key words (33% vs 19%). The human TL;DR is not a worse summary; it is doing
a different job.

**Sentiment.** We compare the mood of the post body, the human TL;DR, and the
AI summary on the same axis. Human TL;DRs mostly drift toward neutral, with
r/depression staying negative and r/Anxiety moving calmer. The AI summary
behaves differently: it lands consistently on the negative side of both the
body and the human TL;DR, most visibly in the advice communities, where the
body reads positive but the AI summary reads negative. We report this as an
observation, not a finding. Two explanations are plausible and our data cannot
separate them: VADER may score the AI's clinical wording as negative even when
the content is neutral, or the AI may genuinely strip the positive framing
that authors give their own stories. We return to this in Further research.

![Sentiment: body, human TL;DR, AI summary](figures/08_sentiment_body_human_ai.png)

### Limitations

These are surface and abstractive proxies, **not** semantic verdicts. Keyword
containment and novelty are lexical; a faithful paraphrase scores as "novel".
Cosine is a **distance we describe, not a decision** that a TL;DR "is not a
summary", low cosine means far in meaning, which is evidence, not proof. The
surface flags are reported as rates, not categories. The AI summary is one
model's output at temperature 0, a single reference point, not ground truth,
and not something we audit. VADER is blunt on short text and the corpus is 2017
English Reddit, so magnitudes are directional.

## Further research

Three directions came out of this work that we consider worth pursuing.

**What TL;DRs actually are.** We deliberately stopped at rates and distances
instead of labelling each TL;DR. A validated classifier of TL;DR types
(summary, question, reaction, advice), built on manually annotated data,
would turn our descriptive gaps into a typology that dataset builders could
use to filter or stratify corpora like Webis-TLDR-17.

**Bias introduced by AI summarization.** The reference model rewrites every
post in the same register: third person, neutral, no questions. If such
summaries are consumed at scale, subtle shifts in tone and framing could
accumulate. Measuring what AI summarization systematically adds or removes,
across models and communities, follows naturally from our setup.

**The sentiment gap of AI summaries.** Our AI summaries score consistently
more negative than both the post and the human TL;DR. Whether this reflects a
limitation of lexicon-based sentiment tools on formal summary language, or a
real tendency of the model to drop the positive framing people give their own
stories, is an open question. Answering it would need a sentiment method
validated on summary-style text, and more than one summarization model.

## Conclusion

A TL;DR is a text of unknown type. Measured against an AI summary as a fixed
reference point, the human TL;DR sits a measurable distance from a plain summary:
it keeps the author's first person, sometimes reads as a question or a
reaction, and rewrites rather than reuses the post, and that distance varies
systematically across communities and between self-posts and comments. This
means Webis-TLDR-17 is not uniformly a set of ground-truth summaries: the label
"TL;DR" hides heterogeneity that a summarizer trained on it inherits.

## Contributions

| Team Member | Contributions |
|-------------|---------------|
| Jorge | Model serving (Gemma 3 27B via vLLM on the university cluster), AI summary generation, GPU notebooks on the H100 (embeddings, semantic distance, human-vs-AI comparison), sentiment comparison |
| Kanta | Corpus inventory and stratified sampling, linguistic feature pipeline, community-level analyses (compression, first person, summariness), report and website |

Research design, interpretation of results, and the narrative were developed
jointly.

## References

- Völske, M., Potthast, M., Syed, S., & Stein, B. (2017). TL;DR: Mining Reddit
  to Learn Automatic Summarization. *Proc. Workshop on New Frontiers in
  Summarization*, 59–63.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for
  Sentiment Analysis of Social Media Text. *Proc. ICWSM*, 8(1), 216–225.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks. *Proc. EMNLP-IJCNLP*, 3982–3992.
- Grusky, M., Naaman, M., & Artzi, Y. (2018). Newsroom: A Dataset of 1.3 Million
  Summaries with Diverse Extractive Strategies. *Proc. NAACL-HLT*, 708–719.
- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., … Christiano, P. (2020). Learning to summarize from human feedback. NeurIPS, 33, 3008–3021.
< pages rebuild after source reset -->
