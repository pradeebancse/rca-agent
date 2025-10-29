# RCA Agent - Root Cause Analysis Agent

A FastAPI-based log analysis agent that performs automated root cause analysis on Kubernetes logs using AI and observability tools.

## Overview

The RCA Agent is a microservice designed to fetch, analyze, and provide insights on Kubernetes logs stored in Elasticsearch. It uses AI models to perform intelligent log analysis and provides observability through OpenTelemetry and Langfuse tracking.

## Features

- **Automated Log Fetching**: Retrieves latest logs from Elasticsearch
- **AI-Powered Analysis**: Uses OpenAI/compatible models for intelligent log analysis
- **Split Processing**: Handles large log volumes by splitting analysis into manageable chunks
- **Observability**: Integrated with OpenTelemetry for distributed tracing
- **Monitoring**: Langfuse integration for LLM observability and prompt management
- **Containerized**: Docker support with docker-compose deployment
- **RESTful API**: FastAPI-based endpoints for easy integration

## Architecture

The agent implements a graph-based workflow using LangGraph:
1. **Fetch Logs**: Retrieves latest logs from Elasticsearch
2. **Analyze Logs**: Performs AI-powered analysis with split processing for large datasets
3. **Generate Summary**: Combines analysis results into actionable insights

## Requirements

- Python 3.12+
- Docker and Docker Compose
- Elasticsearch instance
- OpenAI API or compatible AI gateway
- Langfuse account (optional, for observability)

## Environment Variables

Create a `.env` file in the project root with the following variables:

### Core Configuration

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `ELASTICSEARCH_URL` | Elasticsearch cluster URL | ✅ | `http://elasticsearch:9200` |
| `OPENAI_API_KEY` | OpenAI API key for direct access | ⚠️ | `sk-proj-...` |

### AI Gateway Configuration (Alternative to OpenAI)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `MAJORDOMO_AI_GATEWAY_URL` | AI gateway base URL | ⚠️ | `URL` |
| `MAJORDOMO_AI_API_KEY` | AI gateway API key | ⚠️ | `key` |
| `MAJORDOMO_AI_MODEL` | Model to use through gateway | ⚠️ | `gpt-3.5-turbo` |

### Observability (Optional)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key for LLM observability | ❌ | `pk-lf-` |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | ❌ | `sk-lf-8` |
| `LANGFUSE_HOST` | Langfuse instance URL | ❌ | `https://cloud.langfuse.com` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector endpoint | ❌ | `` |

### Environment Variables Notes

- **⚠️ AI Model Access**: You need either `OPENAI_API_KEY` OR the Majordomo AI Gateway configuration (`MAJORDOMO_AI_*` variables)
- **✅ Required**: `ELASTICSEARCH_URL` is always required
- **❌ Optional**: Observability variables are optional but recommended for production monitoring

## Installation & Setup

### 1. Clone and Setup

```bash
cd rca-agent
cp .env.example .env  # Create your environment file
# Edit .env with your configuration
```

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn log_analysis_agent:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d --build

# View logs
docker-compose logs -f log-analyzer-agent
```

## API Endpoints

### Health Check
```bash
GET /
```

### Analyze Logs
```bash
POST /analyze-logs
```

Triggers the complete log analysis workflow:
1. Fetches latest logs from Elasticsearch
2. Performs AI analysis with split processing
3. Returns comprehensive analysis summary

## Configuration

### Elasticsearch Index

The agent expects logs to be stored in the `k8s-logs` index with the following structure:
```json
{
  "timestamp": "2024-01-01T10:00:00Z",
  "level": "ERROR",
  "message": "Application error message",
  "pod_name": "app-pod-123",
  "namespace": "default"
}
```

### Langfuse Prompts

The agent uses Langfuse for prompt management. Ensure you have a prompt named `log_analysis_prompt` with label `production` configured in your Langfuse project.

## Monitoring & Observability

### OpenTelemetry

The agent automatically exports traces to the configured OTLP endpoint. Traces include:
- HTTP requests
- Elasticsearch queries
- AI model calls
- Processing steps

### Langfuse Integration

All AI model interactions are tracked in Langfuse with:
- Token usage
- Response times
- Input/output content
- Error tracking

## Docker Network

The application expects to be connected to the `poc-agents_elastic` network for Elasticsearch connectivity. Ensure this network exists:

```bash
docker network create poc-agents_elastic
```

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**
   - Verify `ELASTICSEARCH_URL` is correct
   - Ensure Elasticsearch is running and accessible
   - Check network connectivity

2. **AI Model API Errors**
   - Verify API keys are valid
   - Check model availability
   - Monitor rate limits

3. **Missing Logs**
   - Verify `k8s-logs` index exists in Elasticsearch
   - Check log ingestion pipeline
   - Verify index structure matches expected format

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Development

### Project Structure

```
rca-agent/
├── log_analysis_agent.py    # Main FastAPI application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Docker Compose setup
├── test.py                 # Test utilities
├── .env                    # Environment variables
└── README.md               # This file
```

### Testing

```bash
# Run the test script
python test.py
```

## License


This project is part of the POC Agents suite for log analysis and monitoring.
