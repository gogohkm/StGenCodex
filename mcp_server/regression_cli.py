# mcp_server/regression_cli.py
import argparse
import json
import sys

from mcp_server.server import structai_regression_run_suite_v2, structai_regression_report_generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ratio_tol", type=float, default=1e-3)
    ap.add_argument("--isolated_db", action="store_true", default=True)
    ap.add_argument("--no_isolated_db", action="store_true", default=False)
    ap.add_argument("--report", action="store_true", default=True)
    args = ap.parse_args()

    isolated = args.isolated_db and (not args.no_isolated_db)

    r = structai_regression_run_suite_v2(suite_name=args.suite, isolated_db=isolated, ratio_tol=args.ratio_tol)
    run_id = r["run_id"]

    if args.report:
        structai_regression_report_generate(run_id=run_id, formats=["md"])

    print(json.dumps(r, ensure_ascii=False, indent=2))

    if r.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
