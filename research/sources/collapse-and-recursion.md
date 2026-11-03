# Training on your own output

Checked 2026-11-03.

## Shumailov et al., 2024

**AI models collapse when trained on recursively generated data.** Ilia Shumailov,
Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, Ross Anderson. *Nature* 631.
https://www.nature.com/articles/s41586-024-07566-y

Recursive training on model-generated data degrades the tails of the distribution first,
then the whole model. The damage is not recoverable by adding clean data later.

Why it matters here: it rules out the shortcut. The tempting design is to let the agent
summarise its own conversations and train on the summaries, which is compact and cheap
and produces a model of the model. The training stream has to be raw interaction, held
and rebuilt deliberately, which is why corpus retention is a design decision rather
than an operational detail.
