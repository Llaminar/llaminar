#!/usr/bin/env python3
"""Install release-branch protection after PR checks have run once.

The master rule requires four GitHub Actions checks. The develop rule requires
the image-bound PR prerequisite check and blocks direct human pushes. Its only
bypass is the repository's dedicated release-evidence deploy key, used to
publish the certified post-squash two-parent ancestry join. Never grant the
general GitHub Actions app bypass: it also arms ordinary PR auto-merge.
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
DEVELOP_REQUIRED_CHECK = "Unit + ProductionTestPreflight (AVX512)"
RELEASE_DEPLOY_KEY_TITLE = "llaminar-release-evidence"
RELEASE_DEPLOY_KEY_BYPASS = [{"actor_id": None, "actor_type": "DeployKey",
                              "bypass_mode": "always"}]


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
    """Require the image-bound PR gate, except for the dedicated release key."""
    if (current.get("target") != "branch" or current.get("name") != "develop"
            or current.get("conditions", {}).get("ref_name", {}).get("include")
            != ["refs/heads/develop"]):
        raise ValueError("repository develop ruleset identity changed; review manually")
    types = {rule["type"] for rule in current["rules"]}
    allowed = {"deletion", "non_fast_forward", "required_linear_history",
               "pull_request", "required_status_checks"}
    if not {"deletion", "non_fast_forward"}.issubset(types) or not types.issubset(allowed):
        raise ValueError("develop ruleset contains an unknown or missing guard")
    if current.get("bypass_actors", []) not in ([], RELEASE_DEPLOY_KEY_BYPASS):
        raise ValueError("develop ruleset has an unrelated bypass actor")
    # A squash commit on master is not an ancestor of develop. The evidence
    # publisher must be able to join that exact master commit as parent two;
    # a linear-history requirement would reject this safe fast-forward.
    rules = [rule for rule in current["rules"] if rule["type"] in
             {"deletion", "non_fast_forward"}]
    rules.extend([
        {"type": "pull_request", "parameters": {
            "required_approving_review_count": 0,
            "dismiss_stale_reviews_on_push": False,
            "required_reviewers": [],
            "require_code_owner_review": False,
            "dismissal_restriction": {"enabled": False, "allowed_actors": []},
            "require_last_push_approval": False,
            "required_review_thread_resolution": False,
            "require_extra_approval_for_unattributed_changes": True,
            "allowed_merge_methods": ["squash"],
        }},
        {"type": "required_status_checks", "parameters": {
            "strict_required_status_checks_policy": True,
            "do_not_enforce_on_create": False,
            "required_status_checks": [{"context": DEVELOP_REQUIRED_CHECK,
                                        "integration_id": GITHUB_ACTIONS_APP_ID}],
        }},
    ])
    return {"name": current["name"], "target": current["target"],
            "enforcement": "active", "bypass_actors": RELEASE_DEPLOY_KEY_BYPASS,
            "conditions": current["conditions"], "rules": rules}


def require_dedicated_release_key(repository: str) -> None:
    """Reject broad deploy-key bypass if another writable key is installed."""
    keys = github_json(f"repos/{repository}/keys")
    writable = [key for key in keys if not key["read_only"]]
    if (len(writable) != 1 or writable[0]["title"] != RELEASE_DEPLOY_KEY_TITLE
            or not writable[0].get("enabled", False)):
        raise ValueError("develop PR bypass requires exactly one writable deploy key: "
                         + RELEASE_DEPLOY_KEY_TITLE)


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
        require_dedicated_release_key(args.repository)
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
