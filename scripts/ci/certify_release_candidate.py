#!/usr/bin/env python3
## @file certify_release_candidate.py
# @brief Revalidate every RC phase before the master PR may become merge eligible.
"""Revalidate every RC phase before the master PR may become merge eligible.

This metadata-only join is also replayed after merge by the release publisher.
Manual diagnostic runs and historical image evidence cannot grant eligibility.
"""
import argparse
from pathlib import Path

from production_artifacts import write_json
import publish_master_release as release


def main(argv=None) -> int:
    """Read one PR run's complete proof and retain an immutable certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--rc-branch', required=True)
    parser.add_argument('--rc-sha', required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    tree = release.git('rev-parse', args.rc_sha + '^{tree}')
    evidence = release.validate_proof(args.proof / 'preflight', args.proof / 'benchmarks',
        args.repository, args.rc_sha, tree, coding_directory=args.proof / 'opencode',
        accuracy_directory=args.proof / 'accuracy', source_directory=args.proof / 'rc-source',
        rc_branch=args.rc_branch)
    write_json(args.output, release.candidate_certificate(evidence))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
