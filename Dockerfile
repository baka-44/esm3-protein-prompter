FROM python:3.11-slim

WORKDIR /app

# System deps needed by biopython + esm C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch FIRST to avoid pulling in the 2.5 GB CUDA wheel
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Cloud Run terminates TLS; tell google-auth-oauthlib the redirect URI is HTTPS
ENV OAUTHLIB_RELAX_TOKEN_SCOPE=1
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
