#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that every non-merge commit in BASE..HEAD carries a well-formed
# "Signed-off-by: Name <email>" trailer (Developer Certificate of Origin).
#
# Usage: check-dco.sh <base-sha> <head-sha>
#
# Run from inside a git checkout that contains both commits. Exits 0 if all
# commits are signed off, 1 if any are not, 2 on usage error.

set -u

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <base-sha> <head-sha>" >&2
    exit 2
fi

base_sha=$1
head_sha=$2

# Name, then an email with exactly one '@' and no spaces or angle brackets.
signoff_re='^Signed-off-by: .+ <[^<>@[:space:]]+@[^<>@[:space:]]+>$'

failed=0
for sha in $(git log --no-merges --format=%H "${base_sha}..${head_sha}"); do
    # interpret-trailers restricts the match to the real trailer block, so a
    # "Signed-off-by:" mentioned in the middle of a message does not count.
    if ! git log -1 --format=%B "$sha" | git interpret-trailers --parse \
         | grep -Eq "$signoff_re"; then
        echo "::error::Commit $sha is missing a Signed-off-by line"
        git log -1 --format='  %h %an <%ae>: %s' "$sha"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    cat <<'EOF'

All commits must be signed off (see CONTRIBUTING.md).
To sign off every commit in the PR:
  git rebase --signoff <base-branch>
To sign off only the most recent commit:
  git commit --amend -s --no-edit
Either way the commits are rewritten, so update the PR with:
  git push --force-with-lease
EOF
    exit 1
fi

echo "All commits carry a Signed-off-by line."
