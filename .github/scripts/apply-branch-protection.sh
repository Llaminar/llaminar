#!/usr/bin/env bash
# Apply branch protections to llaminar/llaminar.
#
# Requires gh CLI authenticated with repo admin access.
#
# What it sets:
#
#   master:
#     - Required PR review (1 approval)
#     - Required status checks: "CI complete" + "Only develop may merge to master"
#     - Strict (branches must be up to date before merge)
#     - Enforce on admins (no admin override for push/force-push)
#     - Force pushes: DISABLED
#     - Deletions:    DISABLED
#     - Linear history: required
#     - Conversation resolution: required
#
#   develop:
#     - Required PR review (1 approval)
#     - Required status check: "CI complete"
#     - Force pushes: DISABLED
#     - Deletions:    DISABLED
#
# The branch-policy.yml workflow takes care of the "master only from develop"
# rule; we make its status check required here.

set -euo pipefail

REPO="${1:-Llaminar/llaminar}"

echo "Applying branch protection to ${REPO} ..."

# -----------------------------------------------------------------------------
# master
# -----------------------------------------------------------------------------
cat > /tmp/protect-master.json <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      { "context": "CI complete",                            "app_id": -1 },
      { "context": "Only develop may merge to master",       "app_id": -1 }
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON

gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "repos/${REPO}/branches/master/protection" \
    --input /tmp/protect-master.json \
    >/dev/null
echo "  ✓ master protected"

# -----------------------------------------------------------------------------
# develop
# -----------------------------------------------------------------------------
cat > /tmp/protect-develop.json <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      { "context": "CI complete", "app_id": -1 }
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON

gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "repos/${REPO}/branches/develop/protection" \
    --input /tmp/protect-develop.json \
    >/dev/null
echo "  ✓ develop protected"

rm -f /tmp/protect-master.json /tmp/protect-develop.json

echo ""
echo "Done. Summary:"
gh api "repos/${REPO}/branches/master/protection"  --jq '{branch: "master",  required_reviews: .required_pull_request_reviews.required_approving_review_count, required_status_checks: [.required_status_checks.checks[].context], enforce_admins: .enforce_admins.enabled, allow_force_pushes: .allow_force_pushes.enabled, allow_deletions: .allow_deletions.enabled, required_linear_history: .required_linear_history.enabled }'
gh api "repos/${REPO}/branches/develop/protection" --jq '{branch: "develop", required_reviews: .required_pull_request_reviews.required_approving_review_count, required_status_checks: [.required_status_checks.checks[].context], enforce_admins: .enforce_admins.enabled, allow_force_pushes: .allow_force_pushes.enabled, allow_deletions: .allow_deletions.enabled, required_linear_history: .required_linear_history.enabled }'
