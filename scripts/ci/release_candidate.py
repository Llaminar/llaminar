#!/usr/bin/env python3
## @file release_candidate.py
# @brief Admit the develop → rc/* → master lineage without mutable image tags.
"""Admit the develop → rc/* → master lineage without mutable image tags.

An RC begins at a develop commit. Subsequent changes arrive only through a
same-repository develop PR whose merge tree equals its develop source tree.
The receipt distinguishes the RC head from the image source revision, so
advancing develop cannot change an in-flight candidate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from production_artifacts import write_json
from wait_for_develop_image_gate import github_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'llaminar.release-candidate-source.v1'


def is_rc(branch: str) -> bool:
    """Recognize a nonempty, single-level release candidate branch name."""
    return bool(re.fullmatch(r'rc/[A-Za-z0-9][A-Za-z0-9._-]*', branch)) and not branch.endswith('.')


def check_route(repository: str, head_repository: str, head: str, base: str) -> None:
    """Reject forks, develop-to-master shortcuts and feature-to-RC PRs."""
    if head_repository.lower() != repository.lower():
        raise ValueError('release PRs must originate in the same repository')
    if base == 'master':
        if not is_rc(head):
            raise ValueError('master accepts only rc/* PRs; use develop → rc/* → master')
    elif is_rc(base):
        if head != 'develop':
            raise ValueError('rc/* branches accept PRs only from develop')
    else:
        raise ValueError('release branch policy applies only to master and rc/*')


def git(*arguments: str) -> str:
    """Read original Git objects without altering the checkout or index."""
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def ancestor(before: str, after: str) -> bool:
    """Distinguish an absent ancestry edge from an invalid Git object."""
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', before, after], cwd=ROOT)
    if result.returncode not in (0, 1):
        raise ValueError('cannot authenticate release source ancestry')
    return result.returncode == 0


def admit(repository: str, branch: str, head: str, develop: str) -> dict:
    """Authenticate an initial RC snapshot or one exact develop-to-RC PR merge.

    Both forms bind a complete develop tree and its immutable publication
    identity. Conflict resolutions or RC-only edits are deliberately rejected.
    """
    if not is_rc(branch) or not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('release source requires a full rc/* head identity')
    source, number = head, None
    if not ancestor(head, develop):
        candidates = github_json(f'repos/{repository}/commits/{head}/pulls')
        candidates = [pr for pr in candidates if pr.get('merged_at')
                      and pr.get('merge_commit_sha') == head
                      and pr.get('base', {}).get('ref') == branch
                      and pr.get('head', {}).get('ref') == 'develop'
                      and pr.get('head', {}).get('repo', {}).get('full_name', '').lower() == repository.lower()]
        if not candidates:
            # GitHub's commit index can lag the merged PR record. Both views
            # require the same exact merge SHA, source, target and repository;
            # neither branch names nor a commit title can authorize a source.
            for page in range(1, 21):
                closed = github_json(f'repos/{repository}/pulls', 'state=closed', f'base={branch}',
                                     'sort=updated', 'direction=desc', 'per_page=100', f'page={page}')
                candidates.extend(pr for pr in closed if pr.get('merged_at') and pr.get('merge_commit_sha') == head
                    and pr.get('base', {}).get('ref') == branch and pr.get('head', {}).get('ref') == 'develop'
                    and pr.get('head', {}).get('repo', {}).get('full_name', '').lower() == repository.lower())
                if candidates or len(closed) < 100:
                    break
        if len(candidates) != 1:
            raise ValueError('RC head is neither a develop snapshot nor one exact merged develop PR')
        source, number = candidates[0]['head']['sha'], candidates[0]['number']
    if not re.fullmatch('[0-9a-f]{40}', source) or not ancestor(source, develop):
        raise ValueError('RC image source is outside protected develop history')
    tree = git('rev-parse', head + '^{tree}')
    if git('rev-parse', source + '^{tree}') != tree:
        raise ValueError('RC tree contains changes outside the admitted develop snapshot')
    return dict(schema=SCHEMA, complete=True, repository=repository, rc_branch=branch,
                rc_revision=head, source_revision=source, source_tree=tree,
                develop_revision=git('rev-parse', develop), develop_pr=number)


def validate(receipt: dict, repository: str, branch: str, head: str, tree: str) -> str:
    """Replay the source proof from original Git/PR metadata before publishing."""
    if not ancestor(receipt['develop_revision'], 'origin/develop'):
        raise ValueError('recorded RC develop history is outside the protected branch')
    expected = admit(repository, branch, head, receipt['develop_revision'])
    if receipt != expected or receipt['source_tree'] != tree:
        raise ValueError('release source receipt differs from the merged RC tree or lineage')
    return receipt['source_revision']


def main(argv: list[str] | None = None) -> int:
    """Validate a PR route, optionally retaining its frozen RC source receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', type=Path, default=Path(os.environ.get('GITHUB_EVENT_PATH', 'event.json')))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args(argv)
    event = json.loads(args.event.read_text())
    pr = event['pull_request']
    repository = event['repository']['full_name']
    check_route(repository, pr['head']['repo']['full_name'], pr['head']['ref'], pr['base']['ref'])
    if args.output:
        if pr['base']['ref'] != 'master':
            parser.error('source admission applies only to an rc/* → master PR')
        receipt = admit(repository, pr['head']['ref'], pr['head']['sha'], 'origin/develop')
        write_json(args.output, receipt)
        if args.github_output:
            with args.github_output.open('a') as output:
                output.write('source_revision=' + receipt['source_revision'] + '\n')
    print('[release-candidate] admitted ' + pr['head']['ref'] + ' → ' + pr['base']['ref'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
