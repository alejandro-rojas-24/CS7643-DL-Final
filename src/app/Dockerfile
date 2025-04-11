FROM python:3.12-slim

# Copy files first to leverage Docker cache
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Set the working directory
WORKDIR /src/app

COPY . .

EXPOSE 8080

CMD ["fastapi", "run", "main.py", "--port", "8080"]
