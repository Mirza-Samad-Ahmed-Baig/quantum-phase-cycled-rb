"""Generate a hardware report from every saved acquisition, never job intent."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def main():
    runs = []
    for path in sorted(Path("data").glob("phase_cycle_*_results.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        pending = path.with_name(path.name.replace("_results", "_pending"))
        d = json.loads(pending.read_text(encoding="utf-8"))["design"]
        raw = path.with_name(path.name.replace("_results", "_counts"))
        if hashlib.sha256(raw.read_bytes()).hexdigest() != r["raw_counts_sha256"]:
            raise ValueError(f"Raw-count hash mismatch: {path}")
        runs.append((r, d, path))
    failures = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(Path("data").glob("phase_cycle_*_failure.json"))]
    charged = sum(f.get("metrics", {}).get("usage", {}).get("qpu_charge_time_seconds", 0)
                  for f in failures)
    out = Path("paper/phase_cycle_hardware.tex")
    if not runs:
        out.write_text("The new phase-cycle protocol has no completed hardware dataset "
                       f"in the local archive. {len(failures)} jobs returned no usable counts, "
                       f"with {charged} seconds of QPU charges. A charge is not a measurement.\n",
                       encoding="utf-8")
        return
    if len({r["job_id"] for r, _, _ in runs}) != len(runs):
        raise ValueError("Duplicate job cannot be counted as a replication")
    for r, d, _ in runs:
        for key in ("backend", "qubit", "blocks", "shots", "angle", "beta", "bootstrap"):
            if r[key] != runs[0][0][key]:
                raise ValueError("Acquisitions differ; adapt the comparison explicitly")
        if d["delta_ns"] != runs[0][1]["delta_ns"]:
            raise ValueError("Acquisitions have different idle durations")
    fmt = lambda x: f"{x:.4f}"
    math = lambda x: "$" + fmt(x) + "$"
    interval = lambda ci: "$[" + fmt(ci[0]) + ", " + fmt(ci[1]) + "]$"
    lines = ["% Generated from all saved outcomes by scripts/report_phase_cycle.py.",
             r"\subsection{Injected phase-cycle results}",
             f"The archive contains {len(runs)} completed acquisitions."]
    for i, (r, d, _) in enumerate(runs, 1):
        lines.append(
            f"Acquisition {i} uses {r['blocks']} programmed blocks per arm and {r['shots']} "
            f"shots per setting ({d['circuits']:,} circuit evaluations, {d['circuit_shots']:,} "
            f"circuit shots), with random seed {d['seed']} and readout gain {r['readout_gain']:.4f}.")
    if len(runs) > 1:
        lines.append(
            "The second acquisition was specified after inspecting the first result, with a fresh "
            "seed and the same block count, shots, phase amplitude, observable and controls. "
            "Both acquisitions used the same qubit on the same day; they test within-session "
            "repeatability, not robustness across devices or calibration cycles.")
    lines.extend([
        f"Intervals use {runs[0][0]['bootstrap']:,} paired block bootstrap replicates and "
        "resample readout calibration counts. They are empirical intervals conditional on "
        "the stability assumptions in Sec.~\\ref{sec:statistics}. The conservative coverage "
        "guarantee for bounded uncorrected observations is not transferred to these "
        "calibration-corrected hardware intervals.", ""])
    for i, (r, _, _) in enumerate(runs, 1):
        p = r["primary"]
        lines.append(f"The primary correlated-minus-reset contrast in acquisition {i} is " +
                     math(p["estimate"]) + ", with nominal 95\\% interval " + interval(p["ci"]) + ". " +
                     ("It excludes zero in the predicted direction." if p["ci"][0] > 0 else
                      "It does not establish the predicted positive separation."))
    nulls_cover = all(row["ci"][0] <= 0 <= row["ci"][1]
                     for r, _, _ in runs for row in r["arms"] if row["arm"] != "correlated")
    lines.append("All individual negative-control intervals contain zero." if nulls_cover else
                 "At least one negative-control interval excludes zero; all outcomes are retained.")
    for i, (r, _, _) in enumerate(runs, 1):
        row = next(x for x in r["arms"] if x["arm"] == "correlated")
        if not row["ci"][0] <= row["ideal_ensemble_truth"] <= row["ci"][1]:
            lines.append(f"In acquisition {i}, the shared-sign interval does not contain the ideal "
                         "population prediction " + math(row["ideal_ensemble_truth"]) + ". "
                         "The positive separation therefore does not imply exact quantitative "
                         "agreement with an ideal device.")
    lines.extend([
        "All arms appear in Table~\\ref{tab:cyclehardware} and Fig.~\\ref{fig:cyclehardware}.", "",
        r"\begin{table}[t]",
        r"\caption{\label{tab:cyclehardware}Measured injected phase-cycle contrasts and empirical 95\% intervals. "
        "The ideal population contrast is " + math(runs[0][0]["arms"][0]["ideal_ensemble_truth"]) +
        " for shared signs and zero for the other arms.}",
        r"\begin{ruledtabular}", r"\begin{tabular}{lcc}",
        r"Arm & Estimate & 95\% interval \\", r"\colrule"])
    names = {"correlated": "Shared sign", "reset": "Independent reset",
             "static": "Fixed detuning", "none": "No injection"}
    for i, (r, _, _) in enumerate(runs, 1):
        lines.append(r"\multicolumn{3}{c}{Acquisition " + str(i) + r"} \\")
        for row in r["arms"]:
            lines.append(names[row["arm"]] + " & " + fmt(row["connected"]) + " & " +
                         interval(row["ci"]) + r" \\")
        if i < len(runs):
            lines.append(r"\colrule")
    lines.extend([
        r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}", "",
        r"\begin{figure}[t]",
        r"\includegraphics[width=\columnwidth]{../figures/fig_phase_cycle_hardware.pdf}",
        r"\caption{\label{fig:cyclehardware}Hardware response to engineered phase correlations. Points and error bars are connected estimates and paired block bootstrap intervals for each acquisition; crosses are ideal population predictions.}",
        r"\end{figure}", "",
        f"The archive also retains {len(failures)} unsuccessful jobs with no usable counts "
        f"and {charged} seconds of recorded charges. They are excluded from the estimator "
        "and not counted as replications. Reductions in sample count and Runtime packaging "
        "preceded the first successful measurement. The completed results are engineered-noise "
        "proofs of concept; they establish neither native memory nor a physical correlation-time scan."])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    offsets = np.linspace(-.12, .12, len(runs)) if len(runs) > 1 else [0.]
    for j, ((r, _, _), offset) in enumerate(zip(runs, offsets, strict=True)):
        for i, row in enumerate(r["arms"]):
            val = row["connected"]
            lo, hi = row["ci"]
            ax.errorbar(i + offset, val, yerr=[[val - lo], [hi - val]],
                        fmt=("o", "s", "^")[j % 3], capsize=3, color=f"C{j}",
                        label=f"Acquisition {j + 1}" if i == 0 else None)
    for i, row in enumerate(runs[0][0]["arms"]):
        ax.plot(i, row["ideal_ensemble_truth"], "x", color="black", ms=7,
                label="Ideal" if i == 0 else None)
    ax.axhline(0, color="gray", lw=.7)
    ax.set(xticks=np.arange(4), xticklabels=["shared\nsign", "reset", "fixed", "none"],
           ylabel="connected contrast")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_phase_cycle_hardware.{ext}", dpi=240)
    plt.close(fig)
    summary = dict(kind="complete_local_phase_cycle_acquisition_record",
                   successes=[dict(file=str(p), seed=d["seed"], **r) for r, d, p in runs],
                   failures=failures, unsuccessful_qpu_charge_seconds=charged,
                   completed_qpu_charge_seconds=sum(r["metrics"]["usage"]["qpu_charge_time_seconds"]
                                                    for r, _, _ in runs))
    Path("data/phase_cycle_campaign.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated section, campaign record and figure from all {len(runs)} successful acquisitions")


if __name__ == "__main__":
    main()
