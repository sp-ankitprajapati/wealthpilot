"""
Week 3 — Policy RAG.

These are ILLUSTRATIVE policy documents I wrote for this build — not a real
institution's actual underwriting manual. In a real deployment, swap these
for your own bank/NBFC's real policy text, or a public source like SBA SOP
50 10 or an RBI MSME lending circular. What matters for this week is the
pipeline (chunk -> embed -> retrieve -> cite), which works identically
regardless of whose policy text goes in.

Deliberately written to reference the EXACT thresholds Week 2's
detect_red_flags already hardcodes (DSCR 1.0, D/E 2.5, current ratio 1.0,
Z' zones). That's not a coincidence — it closes a loop: those thresholds
stopped being "numbers I picked" and became "numbers the policy specifies,"
which is a real, citable answer to "why 1.0 and not 0.9?" in an interview.
"""

POLICY_DOCS = [
    {
        "source": "SME-Underwriting-Policy-v1",
        "text": """
## 3.1 Debt Service Coverage Requirement
Applicants must demonstrate a Debt Service Coverage Ratio (DSCR) of at
least 1.0 based on trailing twelve-month EBIT against total annual debt
service. A DSCR below 1.0 indicates operating income alone cannot cover
scheduled debt payments and is grounds for decline absent compensating
factors documented under Section 5.1.

## 3.2 Leverage Limits
Total liabilities must not exceed 2.5 times book shareholder equity
(debt-to-equity ratio of 2.5 or lower). Applicants exceeding this
threshold require secondary review under Section 5.1 even if all other
ratios are satisfactory, since high leverage reduces the equity cushion
available to absorb a downturn.

## 3.3 Liquidity Requirement
The current ratio (current assets divided by current liabilities) should
be at least 1.0. A ratio below 1.0 does not by itself justify decline —
it is common in businesses with strong long-term fundamentals but seasonal
or working-capital timing pressure — but it requires Section 5.1 review
rather than standard-track approval.

## 3.4 Distress Zone Handling (Altman Z'-score)
Use the private-firm Z'-score, not the original public-company formula, for
any applicant without a market-traded equity value. A Z'-score above 2.90
is in the safe zone and requires no additional scrutiny on this basis
alone. A score between 1.23 and 2.90 is the grey zone and should be
weighed alongside other ratios, not treated as disqualifying by itself. A
score below 1.23 is the distress zone and requires Section 5.1 review at
minimum, decline in the absence of strong compensating factors.

## 3.5 Revenue Trend Considerations
A year-over-year revenue decline, by itself, is not disqualifying — many
healthy businesses have a single soft year. Sustained decline combined
with a grey-or-worse Z'-score, however, should be treated as a trend
concern rather than a one-off, and routed to Section 5.1.
""",
    },
    {
        "source": "SME-Exception-and-Review-Process-v1",
        "text": """
## 5.1 Review Track Criteria
An application is routed to manual review, rather than standard-track
approval or automatic decline, when exactly one of the following holds
while the remaining fundamentals are sound: liquidity below the Section
3.3 threshold, leverage above the Section 3.2 threshold, or a grey-zone
Z'-score under Section 3.4. Multiple simultaneous breaches should be
treated as grounds for decline, not review, unless a senior underwriter
documents specific compensating factors.

## 5.2 Required Secondary Documentation
Applications routed to review require: twelve months of bank statements
to verify actual cash position beyond the balance-sheet snapshot, an
accounts-receivable aging schedule if the liquidity concern involves
receivables, and a signed statement from the applicant addressing the
specific ratio that triggered review.

## 5.3 Escalation Path
A reviewer who cannot resolve a review-track case within the criteria
above must escalate to a senior underwriter rather than defaulting to
either approval or decline. Silent default in either direction is a
process violation, not a judgment call.
""",
    },
]
