#!/usr/bin/env bash
# ReviewSense AI - One-Click GitHub Push Script

set -e

echo "=================================================="
echo " 🎬 ReviewSense AI — Push to GitHub"
echo "=================================================="

# Check if URL was passed as argument or prompt for it
REPO_URL="$1"

if [ -z "$REPO_URL" ]; then
    echo ""
    echo "Enter your GitHub Repository URL:"
    echo "Example: https://github.com/your-username/ReviewSenseAI.git"
    read -p "Repository URL: " REPO_URL
fi

if [ -z "$REPO_URL" ]; then
    echo "Error: No GitHub repository URL provided. Aborting."
    exit 1
fi

echo ""
echo "Configuring remote origin..."
if git remote | grep -q "^origin$"; then
    git remote set-url origin "$REPO_URL"
    echo "Updated existing origin remote to: $REPO_URL"
else
    git remote add origin "$REPO_URL"
    echo "Added origin remote: $REPO_URL"
fi

echo ""
echo "Ensuring all project files are committed..."
git add .
if git status --porcelain | grep -q .; then
    git commit -m "Update ReviewSense AI project files"
    echo "Committed latest updates."
else
    echo "Working tree is clean. Ready to push."
fi

echo ""
echo "Pushing branch 'main' to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "=================================================="
echo " 🎉 Successfully uploaded ReviewSense AI to GitHub!"
echo " Repository URL: $REPO_URL"
echo "=================================================="

