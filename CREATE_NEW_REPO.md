# Creating a Separate GitHub Repository from the `minimal` Branch

This guide will help you create a new standalone GitHub repository from the current `minimal` branch.

## Method 1: Create New Repository and Push (Recommended)

### Step 1: Create a new repository on GitHub

1. Go to https://github.com/new
2. Choose a repository name (e.g., `autolbl-minimal` or `autolbl-package`)
3. Set visibility (Public or Private)
4. **DO NOT** initialize with README, .gitignore, or license (we have these already)
5. Click "Create repository"

### Step 2: Add the new remote and push

```bash
# Add the new repository as a remote (replace with your new repo URL)
git remote add new-origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git

# Verify remotes
git remote -v

# Push the minimal branch to the new repository as main
git push new-origin minimal:main

# Optional: Push all branches
git push new-origin --all

# Optional: Push tags
git push new-origin --tags
```

### Step 3: Set the new repository as default

If you want to work only with the new repository:

```bash
# Remove old origin
git remote remove origin

# Rename new-origin to origin
git remote rename new-origin origin

# Set upstream branch
git branch --set-upstream-to=origin/main minimal

# Or rename your local branch to main
git branch -m minimal main
git branch --set-upstream-to=origin/main main
```

## Method 2: Clone and Create Fresh Repository

### Step 1: Create a fresh clone of just the minimal branch

```bash
# Navigate to parent directory
cd ..

# Clone only the minimal branch
git clone --single-branch --branch minimal https://github.com/s184361/autolbl.git autolbl-package

# Enter the new directory
cd autolbl-package
```

### Step 2: Remove old remote and history (optional)

```bash
# Remove git history if you want a fresh start
rm -rf .git
git init
git add .
git commit -m "Initial commit: AutoLbl package with prompt optimization"
```

### Step 3: Create new GitHub repository and push

```bash
# Create new repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Method 3: GitHub's "Use this template" Alternative

If you want to preserve history but keep it separate:

### Step 1: Push to new repository

```bash
# In your current autolbl directory
git remote add new-origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git
git push new-origin minimal:main
```

### Step 2: Update repository settings on GitHub

1. Go to your new repository on GitHub
2. Settings → General
3. Update description: "AutoLbl: Automatic labeling framework with prompt optimization"
4. Update topics: `computer-vision`, `object-detection`, `prompt-optimization`, `vlm`
5. Settings → Branches → Change default branch to `main`

## Recommended Repository Name

Consider one of these names for clarity:
- `autolbl` (if starting fresh)
- `autolbl-package` (emphasizes package structure)
- `autolbl-minimal` (references the branch)
- `autolbl-vlm` (emphasizes VLM focus)

## After Creating the New Repository

### Update README badges (if desired)

Add these badges to the top of README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
```

### Update package metadata

Update `pyproject.toml` with the new repository URL:

```toml
[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/NEW_REPO_NAME"
Repository = "https://github.com/YOUR_USERNAME/NEW_REPO_NAME"
Issues = "https://github.com/YOUR_USERNAME/NEW_REPO_NAME/issues"
```

### Create a release

Once you're happy with the repository:

```bash
git tag -a v0.1.0 -m "Initial release: AutoLbl package with prompt optimization"
git push origin v0.1.0
```

Then create a GitHub Release:
1. Go to Releases → "Create a new release"
2. Choose tag v0.1.0
3. Title: "v0.1.0 - Initial Package Release"
4. Description: Highlight key features (prompt optimization, multiple models, CLI tools)

## Verification Checklist

After creating the new repository, verify:

- [ ] All files are present
- [ ] README.md displays correctly
- [ ] Package can be installed: `pip install -e .`
- [ ] CLI tools work: `autolbl-prepare --help`, `autolbl-infer --help`
- [ ] Tests pass: `python test_package.py`
- [ ] Optimization scripts work: `python experiments/prompt_optimization/scipy_opt.py --help`
- [ ] Repository description and topics are set
- [ ] License is visible (if applicable)

## Current Branch State

Your `minimal` branch is now clean and ready to push:
- ✅ All obsolete files removed
- ✅ Package structure complete
- ✅ Documentation comprehensive
- ✅ All imports fixed
- ✅ Prompt optimization emphasized
- ✅ Tests passing
- ✅ Latest commit: "Complete package restructuring with prompt optimization"

## Questions?

If you encounter issues:
1. Check that you're on the `minimal` branch: `git branch`
2. Verify your changes are committed: `git status`
3. Ensure you have push access to the new repository
4. Check remote URLs: `git remote -v`
