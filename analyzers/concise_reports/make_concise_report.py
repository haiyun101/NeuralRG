"""Build a concise per-L report.

Pulls the summary table out of the full loss report and embeds, for every
method that has them, the flow-sample and flow-correlation images -- all in
one markdown file, so a single L can be reviewed at a glance.

Usage:  python analyzers/make_concise_report.py -L 8  -t 2.269
        python analyzers/make_concise_report.py -L 16 -t 2.269
Output: analyzers/concise_report_L{L}_T{T}.md

Freshness check (default ON): before writing the report this script verifies
every method folder under data/{L}Ising_T{t}_*/savings/ has a
flow_diagnostic.json + flow_samples.png + flow_correlations.png whose mtime
is at least as recent as the latest checkpoint in that folder. If any are
missing or stale, the script exits with a non-zero status and tells you to
submit shell/analyze_L<N>.sh. Override with --skip-freshness (e.g. for an
intentionally partial report during work-in-progress).
"""
import argparse
import glob
import os
import re
import sys

# This script now lives in analyzers/concise_reports/ (post-reorg), so we
# need three dirname() pops to reach the repo root. Old version used two
# which silently pointed REPO at analyzers/ and made every glob miss.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYZERS = os.path.join(REPO, "analyzers")
CONCISE_REPORTS = os.path.join(ANALYZERS, "concise_reports")

# Files every diagnosed run folder must have, alongside savings/*.saving.
DIAG_ARTIFACTS = ("flow_diagnostic.json", "flow_samples.png", "flow_correlations.png")


def extract_summary(report_path):
    """Return the '## Summary ...' section of a full report, else None."""
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        text = f.read()
    idx = text.find("## Summary")
    return text[idx:].rstrip() if idx >= 0 else None


def _latest_saving_mtime(folder):
    """Return mtime of the most recent savings/*.saving in `folder`, or None.

    Used to anchor the freshness check: each diagnostic artifact must be at
    least as new as the latest checkpoint, otherwise the report would embed
    figures generated from a stale model.
    """
    savings = glob.glob(os.path.join(folder, "savings", "*.saving"))
    if not savings:
        return None
    return max(os.path.getmtime(p) for p in savings)


def check_diagnostic_freshness(L, t_str):
    """Verify every method folder has fresh flow-diagnostic artifacts.

    Returns a dict {folder_basename: [reason, ...]} mapping each stale or
    incomplete folder to the list of problems. Empty dict => everything is
    up to date.
    """
    issues = {}
    folders = sorted(glob.glob(os.path.join(REPO, "data", f"{L}Ising_T{t_str}_*")))
    for folder in folders:
        ckpt_mtime = _latest_saving_mtime(folder)
        if ckpt_mtime is None:
            # Folder exists but no checkpoint yet -- not a method folder.
            continue
        reasons = []
        for artifact in DIAG_ARTIFACTS:
            path = os.path.join(folder, artifact)
            if not os.path.exists(path):
                reasons.append(f"missing {artifact}")
                continue
            if os.path.getmtime(path) < ckpt_mtime:
                reasons.append(f"{artifact} older than latest checkpoint")
        if reasons:
            issues[os.path.basename(folder)] = reasons
    return issues


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
    p.add_argument("--skip-freshness", action="store_true",
                   help="Don't refuse to build when flow_diagnostic.json / "
                        "flow_*.png are missing or older than the latest "
                        "checkpoint. Off by default — see feedback memory "
                        "`concise-report-diagnostic-first`.")
    args = p.parse_args()
    L = args.lattice
    t_str = f"{args.temp:g}"

    if not args.skip_freshness:
        issues = check_diagnostic_freshness(L, t_str)
        if issues:
            print(f"REPORT INCOMPLETE: {len(issues)} folder(s) need a fresh "
                  f"flow-sample diagnostic before this report can be built:",
                  file=sys.stderr)
            for folder, reasons in issues.items():
                print(f"  {folder}:", file=sys.stderr)
                for r in reasons:
                    print(f"    - {r}", file=sys.stderr)
            print(f"\nSubmit  shell/analyze_L{L}.sh  (which runs "
                  f"analyzers/flow_sample_diagnostic.py on every method "
                  f"folder) and rerun me when it completes.\n"
                  f"Or use --skip-freshness to force a partial build.",
                  file=sys.stderr)
            sys.exit(1)

    # After the reorg, the thermodynamic reports live under analyzers/loss/.
    report = os.path.join(ANALYZERS, "loss", f"loss_report_L{L}_T{t_str}.md")
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

    out_path = os.path.join(CONCISE_REPORTS, f"concise_report_L{L}_T{t_str}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Concise report written: {out_path}")
    print(f"  methods with images: {', '.join(methods) if methods else '(none)'}")


if __name__ == "__main__":
    main()
