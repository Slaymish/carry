# Open questions

Honest list. None of these are resolved, and several could end the project.

1. **How many corrections are there in a year, really?** If the answer is a few dozen,
   then preference pairs are enough for an evaluation and nowhere near enough to train
   on. That would make the method an evaluation of prompt-and-file changes rather than
   an argument for weights.
2. **Does a manner transfer across sessions at all, or is it context-bound?** It is
   possible that what reads as "someone who knows me" is entirely reconstructable from
   a small ledger plus a good system prompt, and the parameters add nothing measurable.
   This is the most likely way the project fails, and it should be tested cheaply and
   early.
3. **What is the right unit of update?** Per conversation, per day, per month. Too
   frequent and the adapter chases a mood; too rare and it never lands.
4. **Can forgetting be made monotone enough to trust?** Retention is measured, not
   guaranteed. A rebuild-from-scratch policy gives determinism at real compute cost,
   and the crossover point is unknown.
5. **Is the anchor set actually stationary?** If the person changes, the anchor measures
   an outdated standard. There is no good answer to a person changing, only a decision
   about how fast the standard is allowed to move.
6. **What is the retention policy for the corpus?** Rebuild-based deletion requires
   keeping the data. Keeping it is the privacy risk. This circle has not been squared.
7. **Does any of this make the agent cheaper?** The motivating claim is that a reply
   should cost about the same in year two as in week one. Shorter injected context helps
   and adaptation may not. Unmeasured.

## Where we would stop

If a frozen base model with a small hand-reviewed ledger and no parameter updates
matches an adapted model on held-out items, the project has its answer and it is not
the answer we wanted. That outcome is worth publishing and we intend to publish it.
