# Run pytest with code coverage and missing lines
.venv\Scripts\python.exe -m pytest --cov=api --cov=src --cov-report=term-missing tests/
