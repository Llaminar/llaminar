#!/usr/bin/env python3
## @file apply_master_ruleset.py
# @brief Install exact release-branch protection from the source-owned PR gates.
"""Install exact release-branch protection from the source-owned PR gates.

The master rule requires seven GitHub Actions checks, including RC lineage. The develop rule requires
both image-bound ISA checks, including AVX2 hardware preflight, and blocks direct human pushes.
Repository administrators may explicitly bypass develop checks through a PR;
the dedicated release-evidence deploy key also publishes the certified
post-squash two-parent ancestry join. Never grant the general GitHub Actions
app bypass: it also arms ordinary PR auto-merge.
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
OWNER_RULESET_ID = 26798917
RC_RULESET_NAME = 'release candidates'
DEVELOP_OWNER_RULESET_NAME = 'Develop owner approval'
DEVELOP_REF = 'refs/heads/develop'
GITHUB_ACTIONS_APP_ID = 15368
REQUIRED_CHECKS = (
    "Release branch lineage",
    "Admit frozen RC image source",
    "Full production preflight (AVX512 and AVX2)",
    "PyTorch accuracy panels (AVX512 and AVX2)",
    "Published benchmarks (AVX512 and AVX2)",
    "OpenCode stress (six five-phase sessions)",
    "RC release eligibility",
)
DEVELOP_REQUIRED_CHECKS = (
    "Develop PR / Unit + hardware preflight (AVX2)",
    "Develop PR / Unit + host (AVX512)",
    "ARM64 build and Unit + host",
)
RELEASE_DEPLOY_KEY_TITLE = "llaminar-release-evidence"
RELEASE_DEPLOY_KEY_BYPASS = [{"actor_id": None, "actor_type": "DeployKey",
                              "bypass_mode": "always"}]
# GitHub's built-in repository administrator role is 5. PR-only bypass keeps
# direct human pushes, deletion and force pushes protected.
DEVELOP_BYPASS_ACTORS = [*RELEASE_DEPLOY_KEY_BYPASS,
                         {"actor_id": 5, "actor_type": "RepositoryRole",
                          "bypass_mode": "pull_request"}]


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
    """Require all develop PR gates with explicit admin PR and release-key exceptions."""
    if (current.get("target") != "branch" or current.get("name") != "develop"
            or current.get("conditions", {}).get("ref_name", {}).get("include")
            != ["refs/heads/develop"]):
        raise ValueError("repository develop ruleset identity changed; review manually")
    types = {rule["type"] for rule in current["rules"]}
    allowed = {"deletion", "non_fast_forward", "required_linear_history",
               "pull_request", "required_status_checks"}
    if not {"deletion", "non_fast_forward"}.issubset(types) or not types.issubset(allowed):
        raise ValueError("develop ruleset contains an unknown or missing guard")
    if any(actor not in DEVELOP_BYPASS_ACTORS
           for actor in current.get("bypass_actors", [])):
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
            "required_status_checks": [{"context": context,
                                        "integration_id": GITHUB_ACTIONS_APP_ID}
                                       for context in DEVELOP_REQUIRED_CHECKS],
        }},
    ])
    return {"name": current["name"], "target": current["target"],
            "enforcement": "active", "bypass_actors": DEVELOP_BYPASS_ACTORS,
            "conditions": current["conditions"], "rules": rules}


def proposed_rc_ruleset() -> dict:
    """Protect RC updates while allowing an initial develop snapshot branch cut."""
    return dict(name=RC_RULESET_NAME, target='branch', enforcement='active', bypass_actors=[],
                conditions={'ref_name': {'include': ['refs/heads/rc/**'], 'exclude': []}}, rules=[
                    {'type': 'non_fast_forward'},
                    {'type': 'pull_request', 'parameters': {
                        'required_approving_review_count': 0, 'dismiss_stale_reviews_on_push': True,
                        'required_reviewers': [],
                        'dismissal_restriction': {'enabled': False, 'allowed_actors': []},
                        'require_extra_approval_for_unattributed_changes': True,
                        'require_code_owner_review': False, 'require_last_push_approval': False,
                        'required_review_thread_resolution': True, 'allowed_merge_methods': ['squash']}},
                    {'type': 'required_status_checks', 'parameters': {
                        'do_not_enforce_on_create': True, 'strict_required_status_checks_policy': False,
                        'required_status_checks': [{'context': 'Release branch lineage',
                                                    'integration_id': GITHUB_ACTIONS_APP_ID}]}}])


def proposed_owner_ruleset(current: dict, *, separate_develop: bool = False) -> dict:
    """Extend the existing sole-owner approval policy to release candidates."""
    if current.get('name') != 'Owner approval' or current.get('target') != 'branch':
        raise ValueError('owner approval ruleset identity changed')
    proposed = {key: current[key] for key in ('name', 'target', 'enforcement', 'bypass_actors', 'conditions', 'rules')}
    proposed = json.loads(json.dumps(proposed))
    include = proposed['conditions']['ref_name']['include']
    if 'refs/heads/master' not in include or not (separate_develop or DEVELOP_REF in include):
        raise ValueError('owner approval must still protect master and develop')
    if 'refs/heads/rc/**' not in include:
        include.append('refs/heads/rc/**')
    return proposed


def develop_owner_policy(owner: dict, app_id: int) -> dict:
    """Clone only owner review for develop. The app receives no CI exception."""
    if (owner['enforcement'] != 'active' or [r['type'] for r in owner['rules']] != ['pull_request']
            or owner['rules'][0]['parameters']['required_approving_review_count'] < 1
            or not owner['rules'][0]['parameters'].get('required_reviewers')
            or app_id <= 0 or app_id == GITHUB_ACTIONS_APP_ID
            or any(a['actor_type'] == 'Integration' for a in owner['bypass_actors'])):
        raise ValueError('expected an active human review rule and a dedicated app')
    return {**owner, 'name': DEVELOP_OWNER_RULESET_NAME,
            'conditions': {'ref_name': {'include': [DEVELOP_REF], 'exclude': []}},
            'bypass_actors': [*owner['bypass_actors'],
                              {'actor_type': 'Integration', 'actor_id': app_id, 'bypass_mode': 'exempt'}]}


def verify_policy(installed: dict, planned: dict) -> None:
    """Verify rule content including exception membership, independent of actor order."""
    keys = planned.keys() - {'bypass_actors'}
    if (any(installed.get(k) != planned[k] for k in keys)
            or {json.dumps(a, sort_keys=True) for a in installed['bypass_actors']} !=
               {json.dumps(a, sort_keys=True) for a in planned['bypass_actors']}):
        raise ValueError(f"installed policy differs from {planned['name']}")


def install_policy(endpoint: str, planned: dict) -> None:
    """Write a ruleset and compare the installed policy verbatim with the proposal."""
    subprocess.run(['gh', 'api', '--method', 'PUT', endpoint, '--input', '-'],
                   input=json.dumps(planned), text=True, check=True,
                   stdout=subprocess.DEVNULL)
    verify_policy(github_json(endpoint), planned)


def owner_protection(repository: str, *, split: bool, apply: bool) -> dict:
    """Install develop review first; only remove overlap after reading it back."""
    endpoint = f'repos/{repository}/rulesets'
    existing = github_json(f'{endpoint}/{OWNER_RULESET_ID}')
    rulesets = [r for r in github_json(endpoint) if r['name'] == DEVELOP_OWNER_RULESET_NAME]
    if len(rulesets) > 1:
        raise ValueError('duplicate develop owner approval rulesets')
    owner = proposed_owner_ruleset(existing, separate_develop=bool(rulesets))
    if not (rulesets or split):
        return owner
    installations = github_json(f"orgs/{repository.split('/')[0]}/installations", 'per_page=100')
    apps = [app for app in installations['installations'] if app['app_slug'] == 'llaminar-bot']
    if (len(apps) != 1 or apps[0].get('suspended_at') or apps[0]['repository_selection'] not in ('selected', 'all')
            or apps[0]['permissions'] != {'contents': 'write', 'pull_requests': 'write', 'metadata': 'read'}):
        raise ValueError('install the private llaminar-bot with the documented repository permissions')
    app = github_json('apps/llaminar-bot')
    if (app['id'] != apps[0]['app_id'] or app['owner']['type'] != 'Organization'
            or app['owner']['login'].lower() != repository.split('/')[0].lower()):
        raise ValueError('llaminar-bot must be owned by the repository organization')
    planned = develop_owner_policy(owner, app['id'])
    if rulesets:
        verify_policy(github_json(f"{endpoint}/{rulesets[0]['id']}"), planned)
    elif split and apply:
        created = json.loads(subprocess.check_output(['gh', 'api', '--method', 'POST', endpoint,
                             '--input', '-'], input=json.dumps(planned), text=True))
        verify_policy(github_json(f"{endpoint}/{created['id']}"), planned)
    if split:
        includes = owner['conditions']['ref_name']['include']
        if DEVELOP_REF in includes:
            includes.remove(DEVELOP_REF)
        print(json.dumps({'develop_review': planned, 'release_review': owner}, indent=2))
        if apply and any(existing.get(k) != v for k, v in owner.items()):
            target = f'{endpoint}/{OWNER_RULESET_ID}'
            subprocess.run(['gh', 'api', '--method', 'PUT', target, '--input', '-'],
                           input=json.dumps(owner), text=True, check=True, stdout=subprocess.DEVNULL)
            verify_policy(github_json(target), owner)
    return owner


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
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--develop-only", action="store_true",
                       help="Update the develop CI ruleset only")
    scope.add_argument("--owner-auto-merge-only", action="store_true",
                       help="Split the develop review rule for the registered llaminar-bot")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository requires OWNER/REPO")
    develop_endpoint = f"repos/{args.repository}/rulesets/{DEVELOP_RULESET_ID}"
    if args.develop_only:
        develop = proposed_develop_ruleset(github_json(develop_endpoint))
        print(json.dumps({"develop": develop}, indent=2, sort_keys=True))
        if args.apply:
            require_dedicated_release_key(args.repository)
            install_policy(develop_endpoint, develop)
            print("[release-ruleset] active develop policy verified", flush=True)
        return 0
    owner = owner_protection(args.repository, split=args.owner_auto_merge_only, apply=args.apply)
    if args.owner_auto_merge_only:
        return 0
    master_endpoint = f"repos/{args.repository}/rulesets/{MASTER_RULESET_ID}"
    master = proposed_ruleset(github_json(master_endpoint))
    develop = proposed_develop_ruleset(github_json(develop_endpoint))
    owner_endpoint = f'repos/{args.repository}/rulesets/{OWNER_RULESET_ID}'
    rc = proposed_rc_ruleset()
    print(json.dumps({"master": master, "develop": develop, 'owner': owner, 'rc': rc}, indent=2, sort_keys=True))
    if args.apply:
        require_dedicated_release_key(args.repository)
        install_policy(develop_endpoint, develop)
        print("[release-ruleset] active develop policy verified", flush=True)
        subprocess.run(["gh", "api", "--method", "PUT", master_endpoint, "--input", "-"],
                       input=json.dumps(master), text=True, check=True,
                       stdout=subprocess.DEVNULL)
        installed = github_json(master_endpoint)
        if proposed_ruleset(installed) != master or installed["enforcement"] != "active":
            raise ValueError("master ruleset verification did not match installed policy")
        print("[release-ruleset] active master policy verified", flush=True)
        subprocess.run(['gh', 'api', '--method', 'PUT', owner_endpoint, '--input', '-'],
                       input=json.dumps(owner), text=True, check=True, stdout=subprocess.DEVNULL)
        verify_policy(github_json(owner_endpoint), owner)
        endpoint = f'repos/{args.repository}/rulesets'
        candidates = [r for r in github_json(endpoint) if r['name'] == RC_RULESET_NAME]
        if len(candidates) > 1:
            raise ValueError('duplicate RC rulesets require manual review')
        method = 'PUT' if candidates else 'POST'
        target = endpoint + '/' + str(candidates[0]['id']) if candidates else endpoint
        installed = json.loads(subprocess.check_output(['gh', 'api', '--method', method, target,
            '--input', '-'], input=json.dumps(rc), text=True))
        if any(installed.get(key) != value for key, value in rc.items()):
            raise ValueError('RC ruleset verification failed')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[master-ruleset] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
