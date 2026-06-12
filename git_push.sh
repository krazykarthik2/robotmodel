#!/bin/bash

# 1. Stage all changes
git add .

# 2. Prompt for a commit message if one wasn't passed as an argument
if [ -z "$1" ]; then
    read -p "Enter commit message: " msg
else
    msg="$1"
fi

# 3. Commit changes
git commit -m "$msg"

# 4. Push to the current tracking branch on GitHub
git push

