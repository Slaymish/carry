# Approach

## The split that makes this coherent

An earlier design collapsed because "emergent memory" had nowhere to live. If it is
written down it is specified, and once specified it is a style guide again. The
resolution is a substrate split rather than a single store:

| layer | holds | property |
|---|---|---|
| parameters | the manner | trained, not stated, cannot be diffed |
| files | the ledger | stated, checkable, wrong is visibly wrong |
| a judge | the truth | external, adversarial, can go down |

The ledger is commitments, corrections, and things told once. It already exists in
the agent and it works. This project is the two layers around it.

## Substrate

A small local model that can actually be updated. This is a hard requirement, not a
preference: you cannot adapt weights you do not hold, and adaptation on a hosted
frontier model means either prompt design (which is the file layer again) or a vendor
feature with no evaluation surface.

The constraint is deliberate. A smaller model that can be measured beats a larger one
that can only be described.

## Deletion by rebuild

Subtracting a concept from a checkpoint is an open problem and a bad one to depend on.
So the design keeps the training data, and deletion means: drop the segment, rebuild
the adapter from the remainder.

This costs compute and is honest. It also means the retained corpus is the real
sensitive artefact, which is why the repository contains none of it, and why the
retention policy is a first-class design decision rather than a footnote.

## What is measured, and what is not

Measured: held-out score, retention across time, sabotage sensitivity, deletion probes.

Not measured, and treated as design axioms instead: whether the agent "feels"
continuous, whether the user stops re-briefing it. Those are real success criteria and
they are not measurable from inside, so we do not pretend a number covers them. They
are reported separately, as qualitative notes, clearly labelled.

## Position on recursive training

Not on generated summaries of the stream. Recursive training on model output is a
documented collapse mechanism, and training on our own reading of a conversation
would be that mechanism with extra steps. The stream is the raw interaction.

## Rejected directions, and why

- **Retrieval-only continuity.** Already the default. Cannot hold a manner, and cost
  grows with the relationship.
- **A single unified memory store.** Every unified store we looked at becomes either a
  ledger or a manner and then does the other one badly.
- **Relevance computed at speak time.** It is the proactive channel, and the failure
  mode of the proactive channel is confident and wrong, at the moment of least
  scrutiny. Relevance is better spent choosing what to inject than choosing what to say.
- **Containment by subject matter.** Separate conversation "rooms" per topic. Rejected:
  relevance is a property of the moment, not of a filing cabinet, and containment
  would have hidden the third-party data problem rather than solved it.
