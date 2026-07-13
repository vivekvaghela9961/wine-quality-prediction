# Contributing Guidelines

Thank you for contributing to the Wine Quality Prediction project! This guide contains instructions for setting up your local environment and running tasks.

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd "Wine Predictions Project"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize environment variables**:
   Copy `.env.example` to `.env` and fill in local secret variables:
   ```bash
   cp .env.example .env
   ```

5. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

---

## Code Quality Standards

We use `black` for formatting and `flake8` for linting. Ensure all changes are properly formatted before committing:
```bash
black .
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Running Tests

Run the test suite using `pytest` to generate a coverage report:
```powershell
# Windows PowerShell helper
./run_tests.ps1
```
Or run directly:
```bash
pytest --cov=api --cov=src --cov-report=term-missing tests/
```

## Git Branching Strategy

We use a feature-branch workflow. Work should be done on feature branches (e.g., `feat/my-feature`) and merged into `main` using non-fast-forward merges to preserve commit history:
```bash
git checkout -b feat/my-feature
# Implement changes...
git add .
git commit -m "feat(scope): describe changes"
git checkout main
git merge --no-ff feat/my-feature -m "merge: Phase X complete"
```
