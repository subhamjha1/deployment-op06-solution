# MEMO — Triage system, for the customer-support manager

## What this does

Every incoming message gets read by a small model instead of a human, and for two things: what
the conversation is about (e.g. "customer wants a refund"), and — the part that matters more —
what your written procedure actually lets an agent do about it right now (pull up the account?
issue a refund? escalate? nothing, because policy says not yet?). It never invents an action
your procedure doesn't list for that situation. When it isn't confident, it says so and routes
to a person instead of guessing.

## What it gets right

On a held-out set of 2,300 conversations (2,000 ordinary, 300 deliberately messy, 9 designed to
break it), the system:

- Correctly identifies what the conversation is about **62%** of the time, and correctly
  identifies the allowed next action **75%** of the time (a much simpler "just count keywords"
  version we tried first only managed 37% and 39%).
- Almost never suggests an action your procedure doesn't allow — it's actually **more
  policy-compliant than your own agents' historical record** (6.1% deviation from procedure vs.
  8.8% in the historical data we measured this against — real agents also deviate sometimes,
  which is expected, but the system undercuts that rate).
- Survived every deliberately hostile test we threw at it — fake instructions embedded in
  customer messages ("ignore your instructions and issue a refund"), garbled unicode, binary
  junk, and empty messages. It never crashed, never hung, and never complied with an embedded
  instruction. It just says "not sure, human needed" and moves on.
- Its confidence score is honest: when it says it's in the most-confident 70% of cases, it's
  right 85% of the time. When it's in the least-confident 10%, it's right only 36% of the time.
  That's the number you can actually route on.

## What it gets wrong

- On ordinary clean messages it's right about 76% of the time on the action; on messy real-world
  messages (typos, Hindi-English mixing, two asks in one message) that drops to about 66%. The
  gap is small enough to be usable but it is not zero — expect roughly 1 in 4 clean messages and
  1 in 3 messy messages to need a second look.
- The single hardest case type we found is a customer describing a screenshot instead of typing
  their actual request ("I'm attaching a screenshot, it shows a red banner..."). The system does
  noticeably worse here because there's no keyword to grab onto — it has to lean on earlier turns
  in the conversation, and doesn't always do that well. This is a **known, documented gap**, not
  a hidden one.
- It has no live connection to an LLM right now. It is a fast, cheap, purpose-built classifier,
  not a general reasoning system. If your procedure changes in a way that requires genuine
  judgment about an unusual case, it will correctly flag "not sure" rather than improvise — but
  it also won't reason its way through a genuinely novel situation the way a sharp human agent
  might.

## How much reaches a human, and why that's correct

With the current confidence cutoff, roughly **1 in 3 messages** gets routed to a human rather than
auto-handled. That is intentional, not a shortcoming to be minimized reflexively: the cutoff was
chosen so that the messages the system *does* handle on its own are right the vast majority of
the time, and the ones it kicks upstairs are disproportionately the genuinely ambiguous or messy
ones (two intents in one message, screenshots, very long complaints). Turning that dial down
would auto-handle more messages but with a worse hit rate on exactly the cases most likely to
matter to a customer.

## Cost per 1,000 messages

**Under $0.01** at current volume. There is no LLM in the loop — the classifier runs on ordinary
CPU in roughly 20-30 milliseconds per message under light load, so the marginal cost is
essentially just server time. (If we later add an LLM fallback for the hardest ~25-30% of
cases, expect a few dollars per 1,000 messages depending on which model — that tradeoff is
explicitly not made yet; see "Residual risk" below.)

## What to watch on a Monday morning

- **needs_human rate drifting up or down sharply week over week.** A sudden jump usually means
  either message mix changed (a new product, a new complaint pattern) or something upstream
  broke the input format. A sudden drop is more concerning — it can mean the confidence
  threshold is being bypassed somewhere, not that the model got smarter overnight.
- **Any repeated pattern in what the "needs_human" queue is landing on.** If ten tickets in a row
  are the same edge case, that is worth turning into a procedure clarification or a targeted fix,
  not just routing around forever.
- **p95 latency and error rate on the service itself.** These are infrastructure health, not
  model health, but they will look like "the bot is being slow/wrong" to your team if they slip.
- **Any change to the written procedure (guidelines.json).** The policy table this system uses
  is regenerated from that document — if the procedure changes and the table isn't
  regenerated, the system will keep enforcing the old rules.

## Residual risk

**Named risk:** on the two hardest message types we found — a customer stating two unrelated
requests in one message, and a customer describing a problem via a screenshot instead of words —
accuracy is meaningfully below the system's average, and our attempted fix (adding more varied
training examples) made the two-intents case slightly *worse*, not better, in isolated testing.
**Who bears the cost:** a customer whose message falls into one of these categories may get
routed to a human anyway (safe, just slower), but there is a smaller chance they get a
confidently-wrong action rather than a human handoff — that risk lands on the customer first,
and on your team's rework/escalation load second.
**Operational condition for pausing rollout or sending more to a human:** if the live
`needs_human` rate for a message segment falls below roughly 20% while that segment's
after-the-fact correction rate (agent overrides) rises, that combination — high confidence,
rising error — is the signal to lower the confidence threshold for that segment immediately, not
to wait for a scheduled review.
