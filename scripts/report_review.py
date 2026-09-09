"""Generate manuscript review additions directly from the saved checks."""
import json
from pathlib import Path


def main():
    r = json.loads(Path("data/phase_cycle_review.json").read_text(encoding="utf-8"))
    m, c = r["mathematics"], r["comparator"]
    lines = [r"\subsection{Additional assumption checks}",
             f"An additional {m['unequal_modulated_schedules']} randomly generated schedules use "
             "unequal slots with piecewise real modulation of either sign. Their continuous/reset "
             f"mean discrepancy is below $10^{{-14}}$. Testing {m['arbitrary_joint_laws']} "
             "discrete, nonsymmetric joint phase laws and varying the cycle angle verifies "
             "Eq.~\\eqref{eq:covariance} below the same tolerance. A deliberately asymmetric "
             f"telegraph example gives a reset discrepancy {m['asymmetric_counterexample_difference']:.6f}; "
             "the symmetry assumption cannot simply be dropped.", "",
             r"\subsection{An ideal resource-matched Ramsey comparator}",
             r"\label{sec:ramsey}",
             "A coherent Ramsey/echo pair measures the mean cosines of the sum and difference "
             "of two phases. Two single-slot $Y$-quadrature settings measure their sine means. "
             "Thus a four-setting ideal construction gives",
             r"\begin{align}",
             r" C_R={}&\tfrac12\mathbb{E}[\cos(\phi_1-\phi_2)-\cos(\phi_1+\phi_2)]\nonumber\\",
             r" &-\mathbb{E}[\sin\phi_1]\mathbb{E}[\sin\phi_2]\nonumber\\",
             r" ={}&\Cov(\sin\phi_1,\sin\phi_2).",
             r"\end{align}",
             "Independent state-amplitude propagation verifies all four ideal responses. "
             "This comparator is an ideal Ramsey/echo construction motivated by established "
             "correlation spectroscopy~\\cite{Yan2012,Zohar2026}; it is not a reproduction "
             "of the RESOLUTE experiment or its full control sequence.", "",
             f"We simulate {c['trials_per_model']} trials per class, with {c['blocks']} independent "
             f"phase-pair blocks. RB uses eight settings and {c['rb_shots_per_setting']} shots "
             f"per setting; Ramsey uses four settings and {c['ramsey_shots_per_setting']} shots. "
             f"Both therefore use {c['circuit_shots_per_method']:,} circuit shots and "
             f"{c['sensing_windows_per_method']:,} sensing-window exposures per trial. "
             "They share the same sampled phase pairs and use the distinct-block connected "
             "estimator. RB estimates are multiplied by $9/4$ so both target $C_R$. "
             "Noise parameters match the preceding validation; shots and blocks differ. "
             "This compares shot and sensing-window resources, not gate duration or wall time.", "",
             r"\begin{table}[t]",
             r"\caption{\label{tab:ramsey}Root mean squared error for the same sine covariance in ideal simulation. Neither method has native drift or imperfect gates in this comparison.}",
             r"\begin{ruledtabular}", r"\begin{tabular}{lrrr}",
             r"Model & RB & Ramsey/echo & Ratio \\", r"\colrule"]
    names = {"rtn": "Continuous RTN", "reset": "Independent reset",
             "static": "Fixed detuning", "white": "Independent Gaussian"}
    for row in c["rows"]:
        lines.append(f"{names[row['model']]} & {row['rmse'][0]:.4f} & {row['rmse'][1]:.4f} & "
                     f"{row['rb_to_ramsey_rmse_ratio']:.2f} " + r"\\")
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}", "",
                  "The Ramsey/echo estimate is more precise in all tested classes "
                  "(Table~\\ref{tab:ramsey}). The supplementary data retain all trial estimates "
                  "and paired Monte Carlo uncertainty for each ratio. These results support "
                  "the RB embedding as an interpretive construction, not an efficiency claim."])
    # Keep the equation literal easy to inspect independently of numeric formatting.
    text = "\n".join(lines)
    Path("paper/phase_cycle_review.tex").write_text(text+"\n", encoding="utf-8")
    rows = r["hardware_readout_sensitivity"]
    lines = [r"\subsection{Readout calibration sensitivity}",
             "A zero-error calibration count is not evidence of a zero population error rate. "
             "To assess this boundary, we construct two exact binomial intervals at 97.5\\% "
             "per acquisition and combine them into a simultaneous 95\\% interval for the "
             "readout gain by a union bound. With experimental survivals held fixed, we "
             "recompute the primary contrast over a dense gain grid."]
    for i, row in enumerate(rows, 1):
        lo, hi = row["primary_range_over_gain_grid"]
        lines.append(f"For acquisition {i}, this calibration-only range is "
                     "$["+f"{lo:.4f}, {hi:.4f}"+"]$.")
    lines.append("These ranges do not include experimental sampling error and are not "
                 "confidence intervals for the primary contrast. They indicate that the "
                 "observed positive point estimates do not depend on setting unobserved "
                 "calibration errors exactly to zero. They do not test drift or setting-dependent SPAM.")
    Path("paper/readout_sensitivity.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Generated review comparison and readout sensitivity sections")


if __name__ == "__main__":
    main()
