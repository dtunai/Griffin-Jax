#!/bin/bash

# Check if the current directory is a Git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "Error: Not in a Git repository."
  exit 1
fi

current_branch=$(git symbolic-ref --short HEAD)

# Check if there are any changes to commit
if [[ -n $(git status -s) ]]; then
  git fetch origin "$current_branch"
  git pull origin "$current_branch"

  commit_template="./gitops/commit-template.txt"

  if [ -f "$commit_template" ]; then
    git config commit.template "$commit_template"
    git commit
    git config --unset commit.template
    git push origin "$current_branch"

    echo "Committed and pushed changes successfully."
  else
    echo "Commit template not found: $commit_template"
    exit 1
  fi
else
  echo "No changes to commit."
fi
