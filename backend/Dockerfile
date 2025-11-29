FROM python:3.13

# Install system dependencies required by some Python packages (OpenCV, Pillow, ffmpeg)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   ffmpeg \
	   libsm6 \
	   libxext6 \
	   libxrender1 \
	   libgl1 \
	   git \
	&& rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

# Set default environment variables for service
ENV SERVICE_HOST=0.0.0.0
ENV SERVICE_PORT=7860

# Use shell form to allow environment variable expansion
CMD sh -c "uvicorn main:app --host ${SERVICE_HOST} --port ${SERVICE_PORT}"
