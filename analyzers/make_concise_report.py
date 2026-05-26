"""Build a concise per-L report.

Pulls the summary table out of the full loss report and embeds, for every
method that has them, the flow-sample and flow-correlation images -- all in
one markdown file, so a single L can be reviewed at a glance.

Usage:  python analyzers/make_concise_report.py -L 8  -t 2.269
        python analyzers/make_concise_report.py -L 16 -t 2.269
Output: analyzers/concise_report_L{L}_T{T}.md
"""
import argparse
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYZERS = os.path.join(REPO, "analyzers")


def extract_summary(report_path):
    """Return the '## Summary ...' section of a full report, else None."""
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        text = f.read()
    idx = text.find("## Summary")
    return text[idx:].rstrip() if idx >= 0 else None


def find_method_pngs(L, t_str, png_name):
    """{method: path-relative-to-analyzers} for data/{L}Ising_T{t}_*/{png}."""
    out = {}
    for fp in sorted(glob.glob(
            os.path.join(REPO, "data", f"{L}Ising_T{t_str}_*", png_name))):
        m = re.search(rf"{L}Ising_T{re.escape(t_str)}_(\w+)", fp)
        if not m:
            continue
        method = m.group(1)
        if "broken" in method.lower():
            continue  # skip explicitly-broken runs
        out[method] = os.path.relpath(fp, ANALYZERS)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-L", "--lattice", type=int, required=True)
    p.add_argument("-t", "--temp", type=float, default=2.269)
    args = p.parse_args()
    L = args.lattice
    t_str = f"{args.temp:g}"

    report = os.path.join(ANALYZERS, f"loss_report_L{L}_T{t_str}.md")
    summary = extract_summary(report)
    samples = find_method_pngs(L, t_str, "flow_samples.png")
    corr = find_method_pngs(L, t_str, "flow_correlations.png")
    methods = sorted(set(samples) | set(corr))

    lines = [f"# Ising L={L} — Concise Report (T={t_str})", ""]

    if summary:
        lines += [summary, ""]
    else:
        lines += [f"_Summary table not found in `{os.path.basename(report)}` "
                  f"— run `loss_analyzer_fixT.py -L {L} -t {t_str}` first._", ""]

    lines += [
        "## Flow samples — flow output (x ~ q) vs HS target data (x ~ p)",
        "",
        "_Each panel pair: left = configurations the trained flow generates,_",
        "_right = HS samples from the true target. Same sigmoid(2x) rendering._",
        "",
    ]
    if methods:
        for mth in methods:
            if mth in samples:
                lines += [f"### {mth}", "", f"![{mth} flow samples]({samples[mth]})", ""]
    else:
        lines += ["_No flow_samples.png found. Run `shell/visualize_flows.sh`._", ""]

    lines += [
        "## Flow correlations — magnetisation P(M) and two-point G(r)",
        "",
        "_Left: per-config magnetisation distribution. Right: normalised_",
        "_axial two-point correlation G(r)/G(0). Flow (q) vs HS data (p)._",
        "",
    ]
    for mth in methods:
        if mth in corr:
            lines += [f"### {mth}", "", f"![{mth} flow correlations]({corr[mth]})", ""]
    if not any(m in corr for m in methods):
        lines += ["_No flow_correlations.png found. Run `shell/visualize_flows.sh`._", ""]

    out_path = os.path.join(ANALYZERS, f"concise_report_L{L}_T{t_str}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Concise report written: {out_path}")
    print(f"  methods with images: {', '.join(methods) if methods else '(none)'}")


if __name__ == "__main__":
    main()
