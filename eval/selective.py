"""Selective-prediction scoring: risk-coverage curves, AURC, E-AURC.

Runner-agnostic. Consumes a stream of (prediction, confidence, correct)
triples and produces the standard selective-prediction metrics. Nothing in
this module knows about VLMs, scene graphs, or the reasoner -- that is the
point: the VLM baseline and the rules system are scored by the SAME code so
their curves are comparable.

--------------------------------------------------------------------------
DEFINITIONS (standard; do not silently redefine)
--------------------------------------------------------------------------
A selective predictor is a pair (f, g): a predictor f and a selection
function g: X -> {0, 1}. Here g is induced by thresholding a confidence
score at tau, so the whole family of selective predictors is swept by
sweeping tau.

With n items, 0/1 loss l_i = 1 - correct_i, and items sorted by confidence
in DESCENDING order:

    coverage(k) = k / n                         for k = 1..n
    risk(k)     = (1/k) * sum_{j<=k} l_(j)      (selective risk = error rate
                                                 among the k covered items)

  * risk-coverage curve : the set {(coverage(k), risk(k))}. Precision at a
    coverage level is 1 - risk.
        El-Yaniv & Wiener (2010), "On the foundations of noise-free
        selective classification", JMLR 11:1605-1641.
        Geifman & El-Yaniv (2017), "Selective classification for deep
        neural networks", NeurIPS 2017.

  * AURC : area under the risk-coverage curve, computed as the mean
    selective risk over all n coverage levels:
        AURC = (1/n) * sum_{k=1}^{n} risk(k)
    Lower is better. AURC conflates two things: how accurate the predictor
    is, and how well the confidence score RANKS its own errors.

  * AURC* (optimal AURC) : the AURC achievable by an oracle confidence that
    ranks every correct prediction above every incorrect one. With e errors
    out of n:
        risk(k) = 0                for k <= n - e
        risk(k) = (k - (n - e))/k  for k >  n - e
        AURC*   = (1/n) * sum_{k=n-e+1}^{n} (k - (n - e)) / k
    This is the exact discrete oracle. Geifman et al. also give the
    continuous-limit closed form  r + (1-r)*ln(1-r)  with r = e/n; it is
    reported alongside as `aurc_optimal_closed_form` for cross-checking but
    is NOT used for E-AURC (it disagrees with the discrete value at small n,
    which is exactly the regime this repo evaluates in: n = 56).

  * E-AURC (excess AURC) : AURC - AURC*. Zero iff the confidence score
    ranks errors perfectly. This is the number that isolates CALIBRATION
    QUALITY from raw accuracy, so it is the one to quote when claiming "the
    system knows which questions it can answer".
        Geifman, Uziel & El-Yaniv (2019), "Bias-reduced uncertainty
        estimation for deep neural classifiers", ICLR 2019.

  * coverage@risk<=t : the largest coverage whose selective risk is <= t.
    Selective risk is NOT monotone in k, so this is a max over all k, not
    the first crossing. 0.0 when no operating point meets the budget.

--------------------------------------------------------------------------
TIE POLICY (matters -- read this)
--------------------------------------------------------------------------
A confidence score that emits the same value for many items does not
actually rank them. Naive implementations sort and inherit the input order,
which silently credits the predictor for an ordering it never produced.

  pessimistic (DEFAULT) : within a tie group, incorrect items are ranked
      first. Worst-case, and the honest number to report.
  optimistic            : correct items first. Best case. Useful only as an
      upper bound / to reproduce naive implementations.
  given                 : keep input order within a tie group.

A fully-tied confidence (e.g. every item at 0.0) makes pessimistic and
optimistic AURC differ maximally; that gap IS the diagnostic that the
confidence signal is uninformative.

--------------------------------------------------------------------------
ABSTENTION
--------------------------------------------------------------------------
An item marked `abstained=True` is a refusal to predict, not a low-
confidence prediction. Abstained items are forced to the bottom of the
ranking (below every non-abstained item, regardless of confidence) so they
only enter coverage at the far right of the curve. Their `correct` value
still counts as a loss there -- an abstention is not a free pass, it just
cannot be selected before anything the system was willing to answer.

CLI:
    python3 -m eval.selective --triples items.json --out report.json

    items.json: {"items": [{"id": "...", "confidence": 0.8,
                            "correct": true, "abstained": false}, ...]}
    (a bare JSON list of the same objects is also accepted)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


TiePolicy = Literal["pessimistic", "optimistic", "given"]

DEFAULT_RISK_TARGETS: tuple[float, ...] = (0.05, 0.10)

# Confidence assigned to abstained items so they sort below every real
# prediction. Not a probability -- purely an ordering sentinel.
_ABSTAIN_RANK = float("-inf")


@dataclass(frozen=True)
class SelectiveItem:
    """One (prediction, confidence, correct) triple.

    prediction is carried through for traceability only; no metric reads it.
    """
    item_id: str
    confidence: float
    correct: bool
    prediction: Any = None
    abstained: bool = False


@dataclass(frozen=True)
class CurvePoint:
    k: int                 # number of items covered
    coverage: float        # k / n
    risk: float            # error rate among the covered k
    precision: float       # 1 - risk
    tau: float             # confidence of the k-th item ("select conf >= tau")
    n_errors: int          # errors among the covered k


@dataclass
class SelectiveReport:
    n: int
    n_correct: int
    n_errors: int
    n_abstained: int
    base_risk: float               # risk at coverage 1.0
    aurc: float
    aurc_optimal: float            # exact discrete oracle
    aurc_optimal_closed_form: float
    e_aurc: float
    normalized_e_aurc: float       # E-AURC / AURC*, when AURC* > 0
    tie_policy: TiePolicy
    n_distinct_confidences: int
    max_tie_fraction: float        # size of the largest tie group / n
    aurc_optimistic: float         # same data, best-case tie-break
    tie_policy_spread: float       # aurc(pessimistic) - aurc(optimistic)
    coverage_at_risk: dict[str, dict[str, float]] = field(default_factory=dict)
    points: list[CurvePoint] = field(default_factory=list)
    threshold_points: list[CurvePoint] = field(default_factory=list)


def _coerce(items: Iterable[Any]) -> list[SelectiveItem]:
    """Accept SelectiveItem, dicts, or bare (prediction, confidence, correct)
    tuples. Tuples are the documented minimal contract."""
    out: list[SelectiveItem] = []
    for i, raw in enumerate(items):
        if isinstance(raw, SelectiveItem):
            out.append(raw)
            continue
        if isinstance(raw, dict):
            conf = raw.get("confidence")
            if conf is None:
                raise ValueError(f"item {i}: missing 'confidence'")
            if "correct" not in raw:
                raise ValueError(f"item {i}: missing 'correct'")
            out.append(SelectiveItem(
                item_id=str(raw.get("id", raw.get("item_id", i))),
                confidence=float(conf),
                correct=bool(raw["correct"]),
                prediction=raw.get("prediction"),
                abstained=bool(raw.get("abstained", False)),
            ))
            continue
        if isinstance(raw, (tuple, list)):
            if len(raw) < 3:
                raise ValueError(
                    f"item {i}: expected (prediction, confidence, correct), got {raw!r}"
                )
            pred, conf, corr = raw[0], raw[1], raw[2]
            out.append(SelectiveItem(
                item_id=str(i), confidence=float(conf),
                correct=bool(corr), prediction=pred,
            ))
            continue
        raise TypeError(f"item {i}: unsupported type {type(raw).__name__}")
    return out


def _order(items: Sequence[SelectiveItem], tie_policy: TiePolicy) -> list[SelectiveItem]:
    """Rank by confidence descending, applying the tie policy within groups
    of equal confidence. Abstained items sort below everything."""
    if tie_policy == "pessimistic":
        # Within a tie, errors first -> selected earliest -> worst risk.
        secondary = lambda it: 0 if not it.correct else 1  # noqa: E731
    elif tie_policy == "optimistic":
        secondary = lambda it: 0 if it.correct else 1      # noqa: E731
    elif tie_policy == "given":
        secondary = lambda it: 0                            # noqa: E731
    else:
        raise ValueError(f"unknown tie_policy {tie_policy!r}")

    decorated = [
        (_ABSTAIN_RANK if it.abstained else it.confidence, secondary(it), idx, it)
        for idx, it in enumerate(items)
    ]
    # -conf ascending == conf descending; idx keeps it stable within a group.
    decorated.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [t[3] for t in decorated]


def _curve(ordered: Sequence[SelectiveItem]) -> list[CurvePoint]:
    n = len(ordered)
    pts: list[CurvePoint] = []
    errors = 0
    for k in range(1, n + 1):
        if not ordered[k - 1].correct:
            errors += 1
        risk = errors / k
        it = ordered[k - 1]
        tau = _ABSTAIN_RANK if it.abstained else it.confidence
        pts.append(CurvePoint(
            k=k, coverage=k / n, risk=risk, precision=1.0 - risk,
            tau=tau, n_errors=errors,
        ))
    return pts


def aurc_from_points(points: Sequence[CurvePoint]) -> float:
    """AURC = mean selective risk over all n coverage levels."""
    if not points:
        return 0.0
    return sum(p.risk for p in points) / len(points)


def aurc_optimal_discrete(n: int, n_errors: int) -> float:
    """Exact AURC of the oracle ranking (all correct above all incorrect)."""
    if n <= 0:
        return 0.0
    e = n_errors
    if e <= 0:
        return 0.0
    base = n - e
    total = 0.0
    for k in range(base + 1, n + 1):
        total += (k - base) / k
    return total / n


def aurc_optimal_closed_form(n: int, n_errors: int) -> float:
    """Continuous-limit oracle AURC:  r + (1 - r) * ln(1 - r),  r = e/n.
    Reported for cross-checking only; see module docstring."""
    if n <= 0:
        return 0.0
    r = n_errors / n
    if r <= 0.0:
        return 0.0
    if r >= 1.0:
        return 1.0
    return r + (1.0 - r) * math.log(1.0 - r)


def coverage_at_risk(points: Sequence[CurvePoint], target: float) -> dict[str, float]:
    """Largest coverage whose selective risk is <= target.

    Selective risk is not monotone in coverage, so this maximises over every
    operating point rather than stopping at the first crossing.
    """
    best: CurvePoint | None = None
    for p in points:
        if p.risk <= target + 1e-12 and (best is None or p.coverage > best.coverage):
            best = p
    if best is None:
        return {"coverage": 0.0, "risk": float("nan"), "tau": float("nan"), "k": 0}
    return {
        "coverage": best.coverage,
        "risk": best.risk,
        "tau": best.tau,
        "k": float(best.k),
    }


def _threshold_curve(ordered: Sequence[SelectiveItem]) -> list[CurvePoint]:
    """Curve restricted to DEPLOYABLE operating points: one point per
    distinct confidence value, selecting every item with confidence >= tau.

    The per-k curve above can name an operating point that no threshold can
    actually realise when confidences tie. This is the curve you can ship.
    """
    n = len(ordered)
    pts: list[CurvePoint] = []
    errors = 0
    for k in range(1, n + 1):
        if not ordered[k - 1].correct:
            errors += 1
        cur = _ABSTAIN_RANK if ordered[k - 1].abstained else ordered[k - 1].confidence
        nxt = None
        if k < n:
            it = ordered[k]
            nxt = _ABSTAIN_RANK if it.abstained else it.confidence
        if nxt is not None and nxt == cur:
            continue  # tie group not finished; this k is not thresholdable
        risk = errors / k
        pts.append(CurvePoint(
            k=k, coverage=k / n, risk=risk, precision=1.0 - risk,
            tau=cur, n_errors=errors,
        ))
    return pts


def evaluate(
    items: Iterable[Any],
    *,
    tie_policy: TiePolicy = "pessimistic",
    risk_targets: Sequence[float] = DEFAULT_RISK_TARGETS,
) -> SelectiveReport:
    """Score a stream of (prediction, confidence, correct) triples.

    Raises ValueError on an empty stream -- an empty risk-coverage curve is
    not a result, it is a missing measurement.
    """
    parsed = _coerce(items)
    n = len(parsed)
    if n == 0:
        raise ValueError("no items: cannot compute a risk-coverage curve")

    ordered = _order(parsed, tie_policy)
    points = _curve(ordered)
    thresh = _threshold_curve(ordered)

    n_errors = sum(1 for it in parsed if not it.correct)
    aurc = aurc_from_points(points)
    opt = aurc_optimal_discrete(n, n_errors)
    e_aurc = aurc - opt

    ranks = [(_ABSTAIN_RANK if it.abstained else it.confidence) for it in parsed]
    distinct = len(set(ranks))
    group_sizes: dict[float, int] = {}
    for v in ranks:
        group_sizes[v] = group_sizes.get(v, 0) + 1
    max_tie_fraction = max(group_sizes.values()) / n

    # How much of the reported AURC is an artefact of the tie-break rather
    # than of the confidence signal. A large spread means the score is not
    # really ranking, even when n_distinct_confidences > 1.
    if tie_policy == "optimistic":
        aurc_opt_ties = aurc
    else:
        aurc_opt_ties = aurc_from_points(_curve(_order(parsed, "optimistic")))
    aurc_pes_ties = (
        aurc if tie_policy == "pessimistic"
        else aurc_from_points(_curve(_order(parsed, "pessimistic")))
    )

    return SelectiveReport(
        n=n,
        n_correct=n - n_errors,
        n_errors=n_errors,
        n_abstained=sum(1 for it in parsed if it.abstained),
        base_risk=n_errors / n,
        aurc=aurc,
        aurc_optimal=opt,
        aurc_optimal_closed_form=aurc_optimal_closed_form(n, n_errors),
        e_aurc=e_aurc,
        normalized_e_aurc=(e_aurc / opt) if opt > 0 else float("nan"),
        tie_policy=tie_policy,
        n_distinct_confidences=distinct,
        max_tie_fraction=max_tie_fraction,
        aurc_optimistic=aurc_opt_ties,
        tie_policy_spread=aurc_pes_ties - aurc_opt_ties,
        coverage_at_risk={
            f"{t:.2f}": coverage_at_risk(points, t) for t in risk_targets
        },
        points=points,
        threshold_points=thresh,
    )


def _point_to_dict(p: CurvePoint) -> dict[str, Any]:
    return {
        "k": p.k,
        "coverage": p.coverage,
        "risk": p.risk,
        "precision": p.precision,
        "tau": p.tau if math.isfinite(p.tau) else None,
        "n_errors": p.n_errors,
    }


def report_to_dict(r: SelectiveReport) -> dict[str, Any]:
    def _f(x: float) -> float | None:
        return x if math.isfinite(x) else None

    return {
        "schema": "selective_prediction_report",
        "schema_version": 1,
        "n": r.n,
        "n_correct": r.n_correct,
        "n_errors": r.n_errors,
        "n_abstained": r.n_abstained,
        "base_risk": r.base_risk,
        "aurc": r.aurc,
        "aurc_optimal": r.aurc_optimal,
        "aurc_optimal_closed_form": r.aurc_optimal_closed_form,
        "e_aurc": r.e_aurc,
        "normalized_e_aurc": _f(r.normalized_e_aurc),
        "tie_policy": r.tie_policy,
        "n_distinct_confidences": r.n_distinct_confidences,
        "confidence_is_degenerate": r.n_distinct_confidences <= 1,
        "max_tie_fraction": r.max_tie_fraction,
        "aurc_optimistic": r.aurc_optimistic,
        "tie_policy_spread": r.tie_policy_spread,
        # AURC is only trustworthy when the score actually ranks. Fires when
        # a single tie group holds >= half the items.
        "confidence_mostly_tied": r.max_tie_fraction >= 0.5,
        "coverage_at_risk": {
            k: {kk: _f(vv) for kk, vv in v.items()}
            for k, v in r.coverage_at_risk.items()
        },
        "points": [_point_to_dict(p) for p in r.points],
        "threshold_points": [_point_to_dict(p) for p in r.threshold_points],
    }


def format_report(r: SelectiveReport, *, title: str = "selective") -> str:
    lines = [
        f"=== {title} ===",
        f"  n={r.n}  correct={r.n_correct}  errors={r.n_errors}  "
        f"abstained={r.n_abstained}  base_risk={r.base_risk:.4f}",
        f"  AURC={r.aurc:.6f}  AURC*={r.aurc_optimal:.6f}  "
        f"E-AURC={r.e_aurc:.6f}",
        f"  tie_policy={r.tie_policy}  distinct_confidences={r.n_distinct_confidences}"
        f"  max_tie_fraction={r.max_tie_fraction:.4f}"
        + ("   <-- DEGENERATE: confidence carries no ranking information"
           if r.n_distinct_confidences <= 1
           else ("   <-- MOSTLY TIED: AURC is dominated by the tie-break"
                 if r.max_tie_fraction >= 0.5 else "")),
        f"  AURC(optimistic ties)={r.aurc_optimistic:.6f}  "
        f"tie_policy_spread={r.tie_policy_spread:.6f}",
    ]
    for t, v in r.coverage_at_risk.items():
        if v["k"] == 0:
            lines.append(f"  coverage@risk<={t}: UNREACHABLE (no operating point)")
        else:
            lines.append(
                f"  coverage@risk<={t}: {v['coverage']:.4f} "
                f"(risk={v['risk']:.4f}, tau={v['tau']:.4f}, k={int(v['k'])})"
            )
    lines.append("  risk-coverage curve (deployable thresholds):")
    for p in r.threshold_points:
        tau = f"{p.tau:.4f}" if math.isfinite(p.tau) else "  abstain"
        lines.append(
            f"    tau>={tau}  coverage={p.coverage:.4f}  "
            f"risk={p.risk:.4f}  precision={p.precision:.4f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--triples", required=True, type=Path,
                    help="JSON file: {'items': [{id, confidence, correct, abstained?}]}")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    ap.add_argument("--tie-policy", default="pessimistic",
                    choices=["pessimistic", "optimistic", "given"])
    ap.add_argument("--risk-targets", default="0.05,0.10")
    ap.add_argument("--title", default="selective")
    args = ap.parse_args(argv)

    raw = json.loads(args.triples.read_text(encoding="utf-8"))
    items = raw["items"] if isinstance(raw, dict) else raw
    targets = tuple(float(x) for x in str(args.risk_targets).split(",") if x.strip())

    report = evaluate(items, tie_policy=args.tie_policy, risk_targets=targets)
    print(format_report(report, title=args.title))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8"
        )
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
