# ML Curriculum

A teaching document, not a runbook. The goal is that you understand *why* each piece exists,
not just which function to call.

Stages 0–2 are written now because they are machine-independent theory and are what the
from-scratch track needs first. Stages 3–7 get written when we reach them, once there is a
measured benchmark to size them against (`docs/BRIEF.md` §6.6).

---

## What you will and will not get

Set this straight before anything else.

**The deliverable** is a QLoRA fine-tune of an Apache-2.0 7–8B base. It inherits its base's
competence and gains your voice, your preferences, and ARC's tool-call format. It does **not**
become smarter than its base — fine-tuning specialises a model, it doesn't add capability.
People routinely expect fine-tuning to teach a model new facts; it mostly teaches it new
*behaviour*. Knowledge belongs in Phase 3's memory system, not in the weights.

**The learning exercise** is a 5–10M parameter transformer you write yourself, trained on
TinyStories in a few hours. It will produce output like *"Once upon a time there was a little
girl named Lily. She had a red ball."* and nothing more sophisticated. That is a complete
success. A model that small cannot hold world knowledge; what it can do is demonstrate that
your attention implementation, your tokenizer, and your training loop all work.

Very few people finish either of these. Both are worth doing.

---

## Stage 0 — Foundations

Don't skip this. Everything downstream is meaningless without it.

### Tensors are not matrices

A tensor is an n-dimensional array plus the bookkeeping needed to differentiate through it. The
shape discipline matters more than the math: most bugs in model code are shape bugs, and they
usually surface as a wrong answer rather than an error.

Convention used throughout this codebase: `(B, T, C)` — batch, time (sequence position),
channels (embedding dimension).

### Autograd, by hand

A neural network is a composition of functions. Training means computing the derivative of a
scalar loss with respect to every parameter, then stepping downhill.

The chain rule does the work. For `y = f(g(x))`:

```
dy/dx = (dy/dg) · (dg/dx)
```

Autograd builds a graph of operations during the forward pass, then walks it backwards
multiplying local derivatives. Each node needs to know only its own derivative — that locality
is what makes it composable.

**Do this by hand once**, on a two-layer network with three neurons, on paper. Then implement a
`Value` class holding a scalar, its gradient, and a `_backward` closure. When your hand-computed
gradients match your implementation's, you understand backprop. Nothing else in this document is
as important.

The one place intuition usually breaks: **gradients accumulate**. A parameter used twice in the
forward pass receives two contributions. This is why `optimizer.zero_grad()` exists, and why
forgetting it produces a model that trains bizarrely rather than crashing.

### The bigram model

Before attention: predict the next token from the current one alone. A lookup table of counts,
normalised to probabilities.

It's a terrible language model and an excellent teaching one. It establishes what a language
model *is* — a probability distribution over the next token given context — and it gives you a
loss number to beat. It also demonstrates why context matters, by showing exactly how badly you
do without it.

Cross-entropy loss is the negative log-likelihood of the correct token. A useful sanity check
you'll use constantly: an untrained model over a vocabulary of size V should have loss ≈ ln(V).
For V = 50,000 that's about 10.8. **If your initial loss isn't near ln(V), something is wrong
before you've trained a single step** — usually initialisation or a shape bug.

### One attention head

The single most important idea. Every token emits three vectors:

- **Query** — what am I looking for?
- **Key** — what do I contain?
- **Value** — what do I contribute if attended to?

Attention scores are query·key dot products. Softmax them into weights, take a weighted sum of
values.

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

The `√d_k` divisor is not decoration. Dot products of d-dimensional vectors have variance
proportional to d, so without scaling the softmax saturates for large d, gradients vanish, and
the model stops learning. Try it without the divisor once and watch it fail — that failure is
more instructive than the explanation.

**Causal masking**: set scores to `-inf` for future positions before the softmax, so position t
attends only to positions ≤ t. Without it the model trivially cheats by reading the answer, gets
a beautiful training loss, and generates nonsense.

---

## Stage 1 — Tokenizer

### Why not characters or words

Characters give a tiny vocabulary but long sequences, and attention costs O(T²). Words give
short sequences but an unbounded vocabulary and no way to handle anything unseen.

BPE splits the difference. Start with bytes; repeatedly merge the most frequent adjacent pair;
stop at your target vocabulary size. Common words become single tokens, rare words decompose
into pieces, and nothing is ever out-of-vocabulary.

### The trade you are actually making

**Vocabulary size trades against sequence length.** A larger vocabulary means shorter sequences
(cheaper attention, more text per context) but a larger embedding matrix and softmax, and fewer
examples per token during training — rare tokens get undertrained.

At the ~10M scale of the from-scratch model, a large vocabulary is actively harmful: the
embedding table would dominate the parameter count, and you'd be training a lookup table rather
than a transformer. Something in the low thousands is appropriate. This is a real design
decision, not a default to copy.

### Why tokenization causes weird failures

A model never sees characters. It sees token IDs. So:

- It can't reliably count letters in a word, because the word may be one opaque token.
- Arithmetic is erratic partly because numbers tokenize inconsistently — "123" might be one
  token while "124" is two.
- Trailing whitespace changes tokenization and therefore changes output.

These get blamed on reasoning. They're tokenization. Knowing this will save you debugging time
on a real model later.

---

## Stage 2 — The transformer

From scratch in PyTorch, no `transformers` library. Modern architecture, not the 2017 original —
the differences are all things the field learned the hard way.

### Multi-head attention

Run several attention heads in parallel with lower dimension each, then concatenate. One head
computes one weighted average and can only represent one relationship at a time; several let the
model attend to syntax and semantics simultaneously. Total compute is roughly unchanged, since
each head is `d_model / n_heads` wide.

### RoPE instead of learned position embeddings

Attention is permutation-invariant — without position information, "dog bites man" and "man
bites dog" are identical inputs.

Rotary Position Embedding rotates the query and key vectors by an angle proportional to
position. The dot product between two rotated vectors then depends on their *relative* distance,
which is what actually matters linguistically. It also extrapolates to longer sequences better
than learned absolute embeddings, which simply have no entry for position 5000 if you trained to
2048.

### RMSNorm instead of LayerNorm

LayerNorm centres and scales: subtract the mean, divide by the standard deviation. RMSNorm skips
the centring:

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · g
```

It works about as well, costs less, and the mean-subtraction turns out not to be doing much.
A small, real win — this is what most modern models use.

### SwiGLU feed-forward

Instead of `Linear → ReLU → Linear`, use a gated unit: one projection produces values, another
produces a gate, and they're multiplied elementwise. The gate lets the network modulate
information flow per-dimension rather than applying a fixed nonlinearity.

Because SwiGLU needs three weight matrices instead of two, the hidden dimension is usually set
to about `8/3 · d_model` rather than `4 · d_model`, keeping the parameter count comparable.

### The residual stream

Each block *adds* to its input rather than replacing it:

```
x = x + attention(norm(x))
x = x + feedforward(norm(x))
```

This is what makes deep networks trainable. Gradients flow through the addition unimpeded, so
the signal reaching early layers doesn't vanish. A useful mental model: the residual stream is a
shared channel that each block reads from and writes to, rather than a pipeline that transforms
its input.

Note the norm goes *inside* the residual branch (pre-norm), not after the addition. Post-norm —
the original 2017 design — needs careful warmup to train at all.

### KV cache

At generation time, each new token re-attends to everything before it. Recomputing keys and
values for the whole prefix every step is O(T²) work for O(T) new information.

Cache them instead. Memory grows linearly with sequence length, which is exactly why context
length is expensive at inference and why headroom beyond the weights matters
(`arc/hardware.py` warns about this).

---

## Stage 3 onward

Written when we get there. Stage 3 (data), 4 (training the small model), 5 (fine-tuning —
the deliverable), 6 (evaluation, including an honest comparison against the stock base model),
7 (integration via `arc/model/custom.py`).

The first task before any of them is a throughput benchmark on the target GPU. Sizing anything
before measuring would be guesswork, and this document would rather say "not yet known" than
print a confident number that turns out to be off by 2.5×.
