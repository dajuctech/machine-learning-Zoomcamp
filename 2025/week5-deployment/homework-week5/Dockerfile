FROM agrigorev/zoomcamp-model:2025

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy the FastAPI app
COPY app.py ./

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]