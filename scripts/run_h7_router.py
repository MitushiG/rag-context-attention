"""H7(b): choose the router threshold on one half of the questions, evaluate on the other.

Uses H2 outputs only (no GPU needed).
"""
import json

import numpy as np

from hla.data import read_jsonl
from hla.router import choose_threshold, route, split_halves, sweep
from hla.runtime import RESULTS


def main():
    single = {r["qid"]: r for r in read_jsonl(RESULTS / "h2" / "single_shot.jsonl")}
    multi = {r["qid"]: r for r in read_jsonl(RESULTS / "h2" / "ircot.jsonl")}
    qids = sorted(set(single) & set(multi))
    conf = np.array([single[q]["confidence"] for q in qids])
    s_ok = np.array([single[q]["em"] for q in qids])
    m_ok = np.array([multi[q]["em"] for q in qids])
    m_calls = np.array([multi[q]["llm_calls"] for q in qids])

    cal, test = split_halves(qids)
    t = choose_threshold(conf[cal], s_ok[cal], m_ok[cal], m_calls[cal])
    res = route(conf[test], s_ok[test], m_ok[test], m_calls[test], t)

    out = {
        "threshold": t,
        "test_n": int(len(test)),
        "router": {k: v for k, v in res.items() if k not in ("correct", "calls")},
        "always_single": {"em": float(s_ok[test].mean()), "mean_calls": 1.0},
        "always_ircot": {"em": float(m_ok[test].mean()), "mean_calls": float(m_calls[test].mean())},
        "curve_test": sweep(conf[test], s_ok[test], m_ok[test], m_calls[test]),
    }
    (RESULTS / "h7").mkdir(parents=True, exist_ok=True)
    (RESULTS / "h7" / "router.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "h7" / "threshold.json").write_text(json.dumps({"threshold": t}))
    # per-question test-half outcomes, for paired bootstrap in analyze.py
    np.savez(RESULTS / "h7" / "test_outcomes.npz", router=res["correct"], router_calls=res["calls"],
             single=s_ok[test], ircot=m_ok[test], ircot_calls=m_calls[test])
    print(json.dumps({k: out[k] for k in ("threshold", "router", "always_single", "always_ircot")}, indent=2))


if __name__ == "__main__":
    main()
