"""Backend package marker for deployment.

Ensures `uvicorn backend.main:app` can import the FastAPI app when
running from the repository root or a different working directory.
"""