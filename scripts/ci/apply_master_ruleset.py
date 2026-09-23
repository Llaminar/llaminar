#!/usr/bin/env python3
"""Install release-branch protection after PR checks have run once.

The master rule requires four GitHub Actions checks. The develop rule blocks
deletion/force-push but permits the post-squash ancestry join, which is a
two-parent fast-forward commit.
Without that guard, the repository's auto-delete-on-merge setting could remove
develop before its post-merge benchmark evidence is committed. Both proposals
derive from the installed rulesets; run without ``--apply`` to inspect them.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from wait_for_develop_image_gate import github_json

MASTER_RULESET_ID = 15240606
DEVELOP_RULESET_ID = 17989170
GITHUB_ACTIONS_APP_ID = 15368
REQUIRED_CHECKS = (
    "Only develop may merge to master",
    "Admit exact develop image source",
    "Published E2E (AVX512 and AVX2)",
    "Published benchmarks (AVX512 and AVX2)",
)


def proposed_ruleset(current: dict) -> dict:
    """Activate the existing exact-master rule, replacing its stale check."""
    if (current.get("target") != "branch" or current.get("name") != "master"
            or current.get("conditions", {}).get("ref_name", {}).get("include")
            != ["~DEFAULT_BRANCH"]):
        raise ValueError("repository master ruleset identity changed; review manually")
    rules = []
    for rule in current["rules"]:
        if rule["type"] == "required_status_checks":
            rules.append({"type": "required_status_checks", "parameters": {
                "do_not_enforce_on_create": False,
                "strict_required_status_checks_policy": True,
                "required_status_checks": [
                    {"context": context, "integration_id": GITHUB_ACTIONS_APP_ID}
                    for context in REQUIRED_CHECKS],
            }})
        else:
            rules.append(rule)
    if sum(rule["type"] == "required_status_checks" for rule in rules) != 1:
        raise ValueError("master ruleset lacks exactly one status-check rule")
    if not any(rule["type"] == "pull_request" for rule in rules):
        raise ValueError("master ruleset must continue to require a PR")
    return {"name": current["name"], "target": current["target"],
            "enforcement": "active", "bypass_actors": current.get("bypass_actors", []),
            "conditions": current["conditions"], "rules": rules}


def proposed_develop_ruleset(current: dict) -> dict:
    """Protect develop while admitting the certified master ancestry join."""
    if (current.get("target") != "branch" or current.get("name") != "develop"
            or current.get("conditions", {}).get("ref_name", {}).get("include")
            != ["refs/heads/develop"]):
        raise ValueError("repository develop ruleset identity changed; review manually")
    types = {rule["type"] for rule in current["rules"]}
    if types not in ({"deletion", "non_fast_forward", "required_linear_history"},
                     {"deletion", "non_fast_forward"}):
        raise ValueError("develop guard must not acquire an unrelated PR or status-check rule")
    # A squash commit on master is not an ancestor of develop. The evidence
    # publisher must be able to join that exact master commit as parent two;
    # a linear-history requirement would reject this safe fast-forward.
    rules = [rule for rule in current["rules"] if rule["type"] != "required_linear_history"]
    return {"name": current["name"], "target": current["target"],
            "enforcement": "active", "bypass_actors": current.get("bypass_actors", []),
            "conditions": current["conditions"], "rules": rules}


def main(argv: list[str] | None = None) -> int:
    """Show or apply one audited repository-admin ruleset update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="Llaminar/llaminar")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository requires OWNER/REPO")
    master_endpoint = f"repos/{args.repository}/rulesets/{MASTER_RULESET_ID}"
    develop_endpoint = f"repos/{args.repository}/rulesets/{DEVELOP_RULESET_ID}"
    master = proposed_ruleset(github_json(master_endpoint))
    develop = proposed_develop_ruleset(github_json(develop_endpoint))
    print(json.dumps({"master": master, "develop": develop}, indent=2, sort_keys=True))
    if args.apply:
        for name, endpoint, planned, projector in (
                ("develop", develop_endpoint, develop, proposed_develop_ruleset),
                ("master", master_endpoint, master, proposed_ruleset)):
            subprocess.run(["gh", "api", "--method", "PUT", endpoint, "--input", "-"],
                           input=json.dumps(planned), text=True, check=True,
                           stdout=subprocess.DEVNULL)
            installed = github_json(endpoint)
            if projector(installed) != planned or installed["enforcement"] != "active":
                raise ValueError(f"{name} ruleset verification did not match installed policy")
            print(f"[release-ruleset] active {name} policy verified", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[master-ruleset] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
