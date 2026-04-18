#!/usr/bin/env bash
# Apply repository rulesets to llaminar/llaminar.
#
# Rulesets are GitHub's successor to classic branch protection and are
# available on private repos in free orgs (unlike classic branch protection).
#
# Creates two rulesets:
#
#   master — strict:
#     - Disallow force pushes and deletions.
#     - Require PR to merge (1 approval, dismiss stale, require conversation
#       resolution).
#     - Required status checks: "CI complete" + "Only develop may merge to
#       master".
#     - Linear history required.
#     - Bypass actors: NONE. (No admin override — enforce_admins equivalent.)
#
#   develop:
#     - Disallow force pushes and deletions.
#     - Require PR to merge (1 approval).
#     - Required status check: "CI complete".
#
# Idempotent: if a ruleset with the same name already exists, it is replaced.

set -euo pipefail

REPO="${1:-Llaminar/llaminar}"

apply_ruleset() {
    local name="$1"
    local payload_file="$2"

    # Find existing ruleset by name to update in place.
    local existing_id
    existing_id=$(gh api "repos/${REPO}/rulesets" --jq ".[] | select(.name==\"${name}\") | .id" | head -1 || true)

    if [[ -n "${existing_id}" ]]; then
        echo "  → updating existing ruleset '${name}' (id=${existing_id})"
        gh api \
            --method PUT \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "repos/${REPO}/rulesets/${existing_id}" \
            --input "${payload_file}" \
            >/dev/null
    else
        echo "  → creating ruleset '${name}'"
        gh api \
            --method POST \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "repos/${REPO}/rulesets" \
            --input "${payload_file}" \
            >/dev/null
    fi
}

echo "Applying rulesets to ${REPO} ..."

# -----------------------------------------------------------------------------
# master — strict, no admin bypass, PRs only from develop (enforced via the
# required 'Only develop may merge to master' status check).
# -----------------------------------------------------------------------------
cat > /tmp/ruleset-master.json <<'JSON'
{
  "name": "master",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {
    "ref_name": {
      "include": ["refs/heads/master"],
      "exclude": []
    }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_linear_history" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          { "context": "CI complete", "integration_id": 15368 },
          { "context": "Only develop may merge to master", "integration_id": 15368 }
        ]
      }
    }
  ]
}
JSON
apply_ruleset "master" /tmp/ruleset-master.json

# -----------------------------------------------------------------------------
# develop — PR-required with CI gating, no force push, no deletion.
# -----------------------------------------------------------------------------
cat > /tmp/ruleset-develop.json <<'JSON'
{
  "name": "develop",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {
    "ref_name": {
      "include": ["refs/heads/develop"],
      "exclude": []
    }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          { "context": "CI complete", "integration_id": 15368 }
        ]
      }
    }
  ]
}
JSON
apply_ruleset "develop" /tmp/ruleset-develop.json

rm -f /tmp/ruleset-master.json /tmp/ruleset-develop.json

echo ""
echo "Done. Rulesets for ${REPO}:"
gh api "repos/${REPO}/rulesets" --jq '.[] | {id, name, target, enforcement}'
