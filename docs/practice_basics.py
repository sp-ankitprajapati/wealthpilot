"""Hinglish learning exercises. Original projects ka code nahi.

Run: python 05_practice_basics.py
Sirf Python standard library use hoti hai. Koi API call ya dependency install nahi.
Pehle function ka expected output predict karo, phir run karo.
"""

import json
import math


def nonnegative_finite(value):
    """Number convert karo, negative/NaN/infinity reject karo."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Finite, non-negative number chahiye")
    return value


def calculate_emissions_demo(quantity, factor):
    """Assumption: quantity aur factor ki units match karti hain."""
    kg = nonnegative_finite(quantity) * nonnegative_finite(factor)
    if not math.isfinite(kg):
        raise ValueError("Multiplication ka result finite hona chahiye")
    return {"kg": round(kg, 3), "tonnes": round(kg / 1000, 5)}


def normalize_electricity_kwh(quantity, unit):
    """Learning conversion: electricity ke liye sirf kWh/MWh support."""
    quantity = nonnegative_finite(quantity)
    conversion = {"kwh": 1, "mwh": 1000}
    normalized_unit = unit.strip().lower()
    if normalized_unit not in conversion:
        raise ValueError("Electricity ke liye kWh ya MWh chahiye")
    result = quantity * conversion[normalized_unit]
    if not math.isfinite(result):
        raise ValueError("Converted quantity finite honi chahiye")
    return result


def totals_by_scope(records):
    """Input example: [{"scope": 1, "tonnes": 0.268}]."""
    totals = {1: 0.0, 2: 0.0, 3: 0.0}
    for record in records:
        scope = record["scope"]
        if scope not in totals:
            raise ValueError("Scope 1, 2 ya 3 hona chahiye")
        totals[scope] += nonnegative_finite(record["tonnes"])
    return {scope: round(total, 5) for scope, total in totals.items()}


def needs_review(confidence, threshold=0.55):
    """Threshold score ka routing rule hai, accuracy proof nahi."""
    confidence = float(confidence)
    threshold = float(threshold)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence 0 se 1 ke beech finite hona chahiye")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold 0 se 1 ke beech finite hona chahiye")
    return confidence < threshold


def choose_label(labels, scores):
    """Sirf argmax idea. Yeh trained classifier ya calibration nahi."""
    if not labels or len(labels) != len(scores):
        raise ValueError("Labels aur scores non-empty aur same length chahiye")
    scores = [float(score) for score in scores]
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Scores finite aur 0 se 1 ke beech hone chahiye")
    index = max(range(len(scores)), key=lambda i: scores[i])
    return labels[index], scores[index]


def validate_activity_key(key, candidate_keys):
    if key == "UNMATCHED" or key in candidate_keys:
        return key
    raise ValueError("Returned key retrieved candidates mein nahi hai")


def ratio_demo(ebit, debt_service):
    """Learning DSCR. Zero pe yahan None choose kiya, repo inf deta hai."""
    ebit = float(ebit)
    if not math.isfinite(ebit):
        raise ValueError("EBIT finite hona chahiye")
    debt_service = nonnegative_finite(debt_service)
    if debt_service == 0:
        return None
    return ebit / debt_service


def flag_demo(dscr, current_ratio):
    """Two-rule learning subset. Original tools.py mein five checks hain."""
    flags = []
    if dscr is None:
        flags.append("DSCR undefined: denominator check karo")
    elif dscr < 1:
        flags.append("DSCR below 1")
    if current_ratio < 1:
        flags.append("Current ratio below 1")
    return flags


def next_step_demo(state):
    """Supervisor idea only. Yeh actual LangGraph workflow nahi."""
    if state.get("final_status"):
        return "END"
    steps = [
        ("financial_statement", "document_parser"),
        ("ratios", "risk_analyst"),
        ("policy_findings", "credit_researcher"),
        ("recommended_verdict", "final_review"),
    ]
    for field, node in steps:
        if field not in state:
            return node
    return "human_approval"


def parse_recommendation_demo(text):
    """Proposed structured-output idea. Original repo substring scan karti hai."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON object chahiye")
    verdict = data.get("recommended_verdict")
    if not isinstance(verdict, str) or verdict not in {"approve", "review", "decline"}:
        raise ValueError("Allowed verdict approve, review ya decline hai")
    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Non-empty rationale chahiye")
    return {"recommended_verdict": verdict, "rationale": rationale}


def group_parity_demo(outcomes_by_group):
    """Metric illustration. Yeh fairness certification nahi."""
    rates = {}
    for group, outcomes in outcomes_by_group.items():
        if any(not isinstance(outcome, bool) for outcome in outcomes):
            raise ValueError("Outcomes True/False hone chahiye")
        rates[group] = sum(outcomes) / len(outcomes) if outcomes else None
    valid = [rate for rate in rates.values() if rate is not None]
    ratio = min(valid) / max(valid) if valid and max(valid) > 0 else None
    return {"approval_rates": rates, "ratio": ratio}


if __name__ == "__main__":
    print("1. Diesel:", calculate_emissions_demo(100, 2.68))
    print("2. 2 MWh in kWh:", normalize_electricity_kwh(2, "MWh"))
    print("3. Scopes:", totals_by_scope([
        {"scope": 1, "tonnes": 0.268},
        {"scope": 2, "tonnes": 0.716},
    ]))
    print("4. Review at 0.54 / 0.55:", needs_review(0.54), needs_review(0.55))
    print("5. Best label:", choose_label(["diesel", "electricity"], [0.8, 0.2]))
    print("6. Candidate:", validate_activity_key("diesel_litre", {"diesel_litre"}))
    print("7. DSCR and flags:", ratio_demo(15, 20), flag_demo(0.75, 1.5))
    state = {"financial_statement": {}, "ratios": {}, "policy_findings": []}
    print("8. Empty findings next step:", next_step_demo(state))
    structured = '{"recommended_verdict":"decline","rationale":"Do not approve"}'
    print("9. Structured verdict:", parse_recommendation_demo(structured))
    print("10. Group metric:", group_parity_demo({"A": [True, False], "B": [True, True]}))
