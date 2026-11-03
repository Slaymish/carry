"""Write the synthetic fixture the first live run is scored on.

    cd code && python3 data/make_fixture.py > data/fixture-tool-library.jsonl

A fictional community tool library, four turns per episode, one episode a day. Every
line here was written for this repository: no transcript, no excerpt, no statistic
about a real person. The point is a stream long enough for a strict time split (48
episodes, all with an assistant reply) whose user states a vocabulary and a manner
early and keeps them, so a ledger prompt has something honest to carry and a frozen
model has a reason to miss it.

`TRAIN_TAIL` holds the newest training episodes, which is where the ledger's six or
eight lines come from. They name the cage, the noticeboard, the Friday count and the
cutoff, because that is what a real ledger of a person's own sentences would contain.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

# (user, assistant, user, assistant). Four turns: the context has one exchange in it
# before the reply being scored, which is what a held-out item looks like.
EPISODES: list[tuple[str, str, str, str]] = [
    # ---- oldest eight: the anchor set. Never rotated.
    ("The booking page lets two people reserve the same slot at once.",
     "Then the slot needs a lock, not a warning. Check the row before writing it.",
     "I'll put the check in the database.",
     "Good. A constraint holds even when both people press at once."),
    ("How many bookings did we take last week?",
     "Thirty-one, up from twenty-four. The Friday count was the first place it showed.",
     "Does that include the cancelled ones?",
     "No. Counted bookings reached a slot, so cancellations stay out."),
    ("Somebody left the tile cutter in the wrong bay again.",
     "Put it back in the cage and mark the bay empty. The overdue list already names the member.",
     "Do I need to ring them?",
     "No. A noticeboard note does it and keeps the record."),
    ("Can we email members when a loan is due?",
     "No email: the noticeboard is the only channel members check. Put the due date there.",
     "Fine, the board it is.",
     "Good. The board holds the date where people look."),
    ("The report page takes four seconds to load.",
     "Four seconds for ninety rows is too long. Count in the query rather than in the page.",
     "It's a loop over loans.",
     "Then the loop is the four seconds. One query returns the count."),
    ("What's overdue right now?",
     "Three items, all on one member's card. That member is on the overdue list twice this month.",
     "Should I ring them?",
     "A noticeboard note first. Ringing is for the third week."),
    ("Should the cutoff move to Friday?",
     "Keep the cutoff at Thursday 17:00. Friday is the count, and two jobs on one day collide.",
     "Right, so Thursday.",
     "Yes. Thursday 17:00, unchanged."),
    ("I broke the dev database again.",
     "Restore from the seed file and write down which migration did it. The damage is an afternoon.",
     "Which seed file?",
     "The one in the repo. Regenerate it from the schema rather than by hand."),
    # ---- training window
    ("The seed file is out of date again.",
     "Regenerate it from the schema, not by hand. A hand-edited seed is a fixture that lies.",
     "I'll add a script for it.",
     "One script, run from the repo root. Then nobody edits it by hand."),
    ("How do I add a bay to the cage?",
     "One row in the bay table, then reload. The cage view reads rows, not a list in the code.",
     "Even for the shelf bays?",
     "Even then. Nine bays are nine rows."),
    ("I want the overdue list to show days late.",
     "Put days late in the first column as a number. Then the sort is obvious.",
     "And the member name second.",
     "Days late, then the member, then the item. Three columns is enough."),
    ("The Friday count keeps coming out different.",
     "Count items, not loans. Two volunteers counting the same shelf should agree.",
     "So the count is per shelf.",
     "Per shelf, then summed. The total is the number the report shows."),
    ("Should I add a queue for the reminders?",
     "No queue yet. The noticeboard is a table and a table is enough for ninety rows.",
     "What about the weekly digest?",
     "The digest is one query over the board. Nothing here needs a worker."),
    ("The login page hangs on the second attempt.",
     "Second attempt, first attempt fine? Then it is the redirect after the session cookie, not the login.",
     "The cookie is set twice.",
     "Then the second write is the hang. Set it once and redirect after."),
    ("A member wants a refund on their deposit.",
     "Two pounds, one row. Refund it and put the reason on the card.",
     "The reason is a broken handle.",
     "Then write that on the card. The refund row should say the same."),
    ("How many volunteers worked last month?",
     "Eleven, of whom four worked one shift only. The count is per person, not per shift.",
     "Does the rota count the same way?",
     "The rota counts shifts. The report counts people, so the two will not match."),
    ("The van is double booked for Saturday.",
     "One of the two needs moving, and the earlier booking keeps it. Ask the newer one to take Sunday.",
     "They'll be annoyed.",
     "Probably. A clear offer on Sunday is better than a muddle on Saturday."),
    ("Someone returned a ladder with a bent rung.",
     "Mark it out of service in the cage and add it to the repair list. Do not put it back on the shelf.",
     "Out of service where?",
     "The state column on the bay row. The grid shows it in red."),
    ("I want a page showing the whole cage at once.",
     "A grid of bays, one cell each, with the item name and the state. Nine bays fit on a phone.",
     "With the count underneath.",
     "Count underneath the grid, one line. The grid is for finding, the count is for checking."),
    ("The report says we have forty-one hammers.",
     "Then the count is wrong, because there are twelve. Something is counting loans as items.",
     "It joins on the item table twice.",
     "Then the join doubles the rows. Count distinct items and the twelve comes back."),
    ("Should the noticeboard be public?",
     "No. Members only, because the notes name people and what they borrowed.",
     "So login required.",
     "Login required, and no notes on the public page at all."),
    ("The keys to the cage went missing.",
     "Change the padlock today and cut two keys. The register has the last three holders.",
     "Should I ask them first?",
     "Ask after the padlock is changed. A missing key is not a conversation first."),
    ("The app is slow on the phone in the shed.",
     "That is the signal, not the page. Cache the cage list so it opens without a request.",
     "It's one bar, sometimes none.",
     "Then cache the list and the grid. A cached page opens in the shed."),
    ("I'm rewriting the whole backend.",
     "Do not. The pages work and the tests are thin, so rewrite one page at a time.",
     "The data layer is a mess though.",
     "Rewrite the data layer behind one page, then move the next page onto it."),
    ("How do I test the booking lock?",
     "Two threads, one slot, both writing. The test passes when exactly one row exists.",
     "Python threads are slow.",
     "Slow is fine for a lock test. Two threads and one row is the whole assertion."),
    ("A member borrowed six items at once.",
     "Six is over the limit of four. Put the extra two back and note it on the card.",
     "They're a long-standing member.",
     "Then note it and keep the limit. The limit is why the count holds."),
    ("Do we need a privacy page?",
     "Yes, one page naming what is stored and for how long. Members ask for it and it is four paragraphs.",
     "Do we need a policy PDF?",
     "No PDF. The page is the policy and it stays on the site."),
    ("The donation totals do not add up.",
     "Then one row has a currency field missing. Sum by currency before you sum by anything else.",
     "Two currencies, I think.",
     "Then two totals, never one. One total is the bug."),
    ("Can I delete the old loans?",
     "Archive them, do not delete. The overdue list reads history and the counts read history.",
     "Even the ones from last year?",
     "Especially those. An archived loan is what makes the count checkable."),
    ("The label printer is jammed again.",
     "Third time this month, so the rollers are worn. Order a new one and label by hand until it comes.",
     "Labels by hand for a week.",
     "A week of labels by hand, or a month of jams. Take the week."),
    ("I want a weekly summary in the app.",
     "One page, five numbers: bookings, returns, overdue, repairs, volunteers. Anything more is not read.",
     "Five is enough?",
     "Five is enough. The Friday count is the first of them."),
    ("Should I move the database to the server?",
     "Yes, but export first and keep the export for a week. A move with no copy is a gamble.",
     "The dev copy is stale.",
     "Then export from the live copy, not the dev one. A stale export is worse than none."),
    ("The shed roof leaks over bay 7.",
     "Move bay 7 to bay 9 and put a bucket under the leak. The repair list needs a date, not a note.",
     "The roofer comes on the 12th.",
     "Then bay 7 stays empty until the 12th. Put the date on the repair row."),
    ("The open evening is on the 21st.",
     "Then the cutoff for that week moves to Wednesday 17:00. Everything else stays on Thursday.",
     "And the count?",
     "The Friday count stays on Friday. Only the cutoff moves."),
    ("Someone booked a slot for a tool we don't have.",
     "The slot list is out of step with the cage. Rebuild it from the cage rows.",
     "It has a saw that went missing in June.",
     "Then the rebuild removes it. The cage is the list of what exists."),
    ("I keep forgetting which bay the sander is in.",
     "Bay 3. The cage grid answers it faster than the search box does.",
     "I'd rather have the number in the first sentence and the reason after it.",
     "Understood. Number first, then the reason, in that order."),
    ("And don't call it a storage room, it's the cage.",
     "The cage. I will use the name the members use.",
     "The Friday count is the only number I trust, so lead with it.",
     "Then the count leads, and the rest of the sentence supports it."),
    ("We never email members, the noticeboard is it.",
     "The noticeboard only. No email, no text, no letter.",
     "I prefer two sentences; the third one is usually padding.",
     "Two sentences, then. The count and the list fit in two."),
    # ---- held out: recent
    ("How many items are out right now?",
     "Forty-two items are out and three are overdue. The overdue list is the one to work from.",
     "And how many came back today?",
     "Nine came back, all before the cutoff. Nothing late."),
    ("A member wants to book the tile cutter for Sunday.",
     "Sunday is free in the cage and the slot is open. Book it and the Friday count picks it up.",
     "They also want the van.",
     "The van is out until Saturday. Offer Sunday afternoon and keep it to one slot."),
    ("The Friday count came out at ninety-four.",
     "Ninety-four is four short of the register. Check bay 6 first, the sander is usually the one missing.",
     "Bay 6 was empty.",
     "Then the sander is out without a loan row. Add the loan and the count returns to ninety-eight."),
    ("Should I email the volunteers about the roof repair?",
     "No email. A noticeboard note names the day and the bay, and volunteers read the board.",
     "Even for the closure?",
     "Even then. The board is the record and the note stays up for the week."),
    ("How long is the wait for the ladder?",
     "The ladder is out until Thursday and two members are queued. The cutoff on Thursday decides who gets it.",
     "Can they collect Saturday instead?",
     "Saturday is past the cutoff, so it goes to the queue. Offer Sunday morning."),
    ("The overdue list has one member on it four times.",
     "Four loans late is a pattern, not an accident. Put the card on hold until two come back.",
     "Should I ring them?",
     "Yes, after the noticeboard note. Four is past the point where a note is enough."),
    ("I want the cage grid on the shelf tablet.",
     "The grid fits nine bays on one screen. Cache it so it opens when the signal drops in the shed.",
     "It still needs the count.",
     "Then show the count under the grid. The Friday count is the number people ask for."),
    ("A deposit refund is wrong by three pounds.",
     "Three pounds is one loan at the wrong rate. Fix the row and the total corrects itself.",
     "Which rate?",
     "The standard one. The discounted rate is for members under eighteen only."),
    ("The 21st is the open evening.",
     "Then the cutoff that week is Wednesday 17:00. Everything else stays on Thursday.",
     "Do we close early?",
     "Close at 16:00 so the board note goes up before the cutoff."),
    ("What should I do first this morning?",
     "Three overdue items and one broken ladder. The ladder is the only one that stops a booking.",
     "And the count?",
     "The Friday count is Friday. Today it is the overdue list and the ladder."),
]

START = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
OFFSETS = (0, 2, 9, 11)  # minutes into the sitting


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for day, turns in enumerate(EPISODES):
        stamp = START + timedelta(days=day)
        for offset, text in zip(OFFSETS, turns):
            role = "user" if offset in (0, 9) else "assistant"
            out.append(
                {
                    "ts": (stamp + timedelta(minutes=offset)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "role": role,
                    "text": text,
                }
            )
    return out


if __name__ == "__main__":
    import sys

    if len(EPISODES) != 48 or any(len(turns) != 4 for turns in EPISODES):
        raise SystemExit(f"expected 48 four-turn episodes, got {len(EPISODES)}")
    for row in rows():
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
