import os
import time
import uuid
from click import prompt
from fastapi import FastAPI, HTTPException, Request   # ✅ FIXED
from fastapi.responses import Response
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse import Langfuse
from dotenv import load_dotenv
from openai import OpenAI
import urllib3

# ------------------ Observability Imports ------------------
from opentelemetry import trace, propagate
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry import propagate
from typing import Dict
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

# ------------------ Load environment variables ------------------
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------ FastAPI App ------------------
app = FastAPI(title="RCA Agent", version="1.0.0")

# Initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("MAJORDOMO_AI_GATEWAY_URL"),
    api_key=os.getenv("MAJORDOMO_AI_API_KEY"),
)

# ------------------ OpenTelemetry Setup ------------------
resource = Resource.create({
    "service.name": "RCA-Agent",
    "service.namespace": "log-analysis",
    "service.version": "1.0.0",
})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"), insecure=True)
)
print("🔌 Connecting to OpenTelemetry Collector at", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("log-analyzer-agent")


# ------------------ Initialize Clients ------------------
es = Elasticsearch(
    os.getenv("ELASTICSEARCH_URL"), 
    verify_certs=False,
    headers={"Accept": "application/vnd.elasticsearch+json;compatible-with=8"}
)
print("🔌 Connecting to Elasticsearch at", os.getenv("ELASTICSEARCH_URL"))
es._otel.enabled = False  # Disable OpenTelemetry instrumentation for Elasticsearch client

# Retry connection with backoff
max_retries = 5
for attempt in range(max_retries):
    try:
        if es.ping():
            print("✅ Successfully connected to Elasticsearch")
            break
    except Exception as e:
        print(f"❌ Attempt {attempt + 1}/{max_retries} failed to connect to Elasticsearch: {e}")
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))  # Exponential backoff
        else:
            print("🚨 Failed to connect to Elasticsearch after all retries")
            raise Exception("Cannot connect to Elasticsearch")

# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# ------------------ Instrument FastAPI (Disabled - too verbose) ------------------
# try:
#     from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
#     FastAPIInstrumentor.instrument_app(app)
#     print("✅ FastAPI instrumentation enabled")
# except ImportError as e:
#     print(f"⚠️ FastAPI instrumentation not available: {e}")

# ------------------ Agent State ------------------
class State(dict):
    logs: str
    summary: str
    part1_summary: str = None
    part2_summary: str = None
    main_trace_id: str = None

workflow_trace = None

def inject_otlp_context() -> dict:
    carrier = {}
    propagate.inject(carrier)
    return carrier


def activate_langfuse_span(obs):
    """Activate a Langfuse observation as an OpenTelemetry span context."""
    trace_id = int(obs.trace_id, 16)
    span_id = int(obs.id[:16], 16)
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED)
    )
    ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
    return ctx

def get_combined_headers(obs):
    """Inject both OTel and Langfuse context into outgoing headers."""
    headers = {}
    ctx = activate_langfuse_span(obs)
    propagate.inject(headers, context=ctx)
    headers["X-Langfuse-Trace-ID"] = obs.trace_id
    headers["X-Langfuse-Span-ID"] = obs.id
    return headers


# ------------------ Agent Steps ------------------
def fetch_logs(state: State) -> State:
    global workflow_trace
    print("🔍 Fetching latest logs from Elasticsearch...")
    start_time = time.time()
    fetch_obs = workflow_trace.start_observation(
        name="fetch_logs", as_type="span",
        input={"elasticsearch_index": "k8s-logs", "query_size": 10},
        metadata={"function": "fetch_logs", "start_time": start_time}
    )
    try:
        response = es.search(
            index="k8s-logs",
            body={"size": 10, "sort": [{"timestamp": {"order": "desc"}}], "query": {"match_all": {}}}
        )
        logs_str = "Kubernetes Logs:\n\n"
        for i, hit in enumerate(response['hits']['hits'], 1):
            src = hit['_source']
            logs_str += f"Log {i}:\nTimestamp: {src.get('timestamp')}\nLevel: {src.get('level')}\nMessage: {src.get('message')}\n{'-'*40}\n"
        
        logs_count = len(response['hits']['hits'])
        fetch_obs.update(output={
            "logs_retrieved": logs_count,
            "logs_preview": logs_str[:500] + "..." if len(logs_str) > 500 else logs_str
        })
        fetch_obs.end()
        state["logs"] = logs_str
    except Exception as e:
        fetch_obs.update(output={"error": str(e)}, metadata={"status": "error"})
        fetch_obs.end()
        raise e
    return state


def analyze_logs(state: dict) -> dict:
    global workflow_trace
    print("🧠 Analyzing logs with split processing...")
    start_time = time.time()

    # ---- Parent Observation ----
    analyze_obs = workflow_trace.start_observation(
        name="analyze_logs_openai", as_type="span",
        input={"logs_length": len(state['logs']), "task": "split_and_analyze_logs"},
        metadata={"function": "analyze_logs", "start_time": start_time, "processing_method": "split_analysis"}
    )

    logs = state.get('logs', '')
    MAJORDOMO_AI_MODEL = os.getenv("MAJORDOMO_AI_MODEL")

    # ---- Split logs ----
    log_lines = logs.split('\n')
    mid_point = len(log_lines) // 2
    logs_part1 = '\n'.join(log_lines[:mid_point])
    logs_part2 = '\n'.join(log_lines[mid_point:])
    print(f"📊 Split logs into 2 parts: Part 1 ({len(logs_part1)} chars), Part 2 ({len(logs_part2)} chars)")

    prompt_template = langfuse.get_prompt("log_analysis_prompt", label="production")

    # === Part 1 ===
    print("🔍 Analyzing logs part 1...")
    messages_part1 = prompt_template.compile(logs=logs_part1)

    openai_obs_1 = analyze_obs.start_observation(
        name="openai_analysis_part1", as_type="generation",
        model=MAJORDOMO_AI_MODEL,
        input={
            "messages": messages_part1,
            "part": "1_of_2",
            "logs_length": len(logs_part1)
        },
        metadata={"log_part": "first_half", "analysis_step": "1"}
    )

    headers_part1 = get_combined_headers(openai_obs_1)
    print("📤 Injected headers (Part 1):", headers_part1)

    response_1 = client.chat.completions.create(
        model=MAJORDOMO_AI_MODEL,
        messages=messages_part1,
        temperature=1.0,
        max_tokens=512,
        extra_headers=headers_part1
    )

    summary_part1 = response_1.choices[0].message.content
    openai_obs_1.update(output={
        "summary": summary_part1,
        "tokens_used": response_1.usage.total_tokens if response_1.usage else None
    })
    openai_obs_1.end()
    print(f"✅ Part 1 analysis complete ({len(summary_part1)} chars)")

    # === Part 2 ===
    print("🔍 Analyzing logs part 2...")
    messages_part2 = prompt_template.compile(logs=logs_part2)

    openai_obs_2 = analyze_obs.start_observation(
        name="openai_analysis_part2", as_type="generation",
        model=MAJORDOMO_AI_MODEL,
        input={
            "messages": messages_part2,
            "part": "2_of_2",
            "logs_length": len(logs_part2)
        },
        metadata={"log_part": "second_half", "analysis_step": "2"}
    )

    headers_part2 = get_combined_headers(openai_obs_2)
    print("📤 Injected headers (Part 2):", headers_part2)

    response_2 = client.chat.completions.create(
        model=MAJORDOMO_AI_MODEL,
        messages=messages_part2,
        temperature=1.0,
        max_tokens=512,
        extra_headers=headers_part2
    )

    summary_part2 = response_2.choices[0].message.content
    openai_obs_2.update(output={
        "summary": summary_part2,
        "tokens_used": response_2.usage.total_tokens if response_2.usage else None
    })
    openai_obs_2.end()
    print(f"✅ Part 2 analysis complete ({len(summary_part2)} chars)")

    # === Combine Summaries ===
    print("🔗 Combining summaries...")
    combine_messages_prompt = langfuse.get_prompt("combine_log_analysis_summaries", label="production")
    combine_messages = combine_messages_prompt.compile(
        summary_part_first=summary_part1,
        summary_part_second=summary_part2
    )

    openai_obs_combine = analyze_obs.start_observation(
        name="openai_combine_summaries", as_type="generation",
        model=MAJORDOMO_AI_MODEL,
        input={
            "messages": combine_messages,
            "task": "combine_summaries",
            "part1_length": len(summary_part1),
            "part2_length": len(summary_part2)
        },
        metadata={"analysis_step": "3_combine", "summary_type": "final_unified"}
    )

    headers_combine = get_combined_headers(openai_obs_combine)
    print("📤 Injected headers (Combine):", headers_combine)

    response_combined = client.chat.completions.create(
        model=MAJORDOMO_AI_MODEL,
        messages=combine_messages,
        temperature=0.7,
        max_tokens=1024,
        extra_headers=headers_combine
    )

    final_summary = response_combined.choices[0].message.content
    openai_obs_combine.update(output={
        "final_summary": final_summary,
        "tokens_used": response_combined.usage.total_tokens if response_combined.usage else None
    })
    openai_obs_combine.end()
    print(f"✅ Final combined summary complete ({len(final_summary)} chars)")

    # ---- Wrap up parent observation ----
    execution_time = time.time() - start_time
    analyze_obs.update(output={
        "analysis_completed": True,
        "processing_method": "split_analysis",
        "parts_processed": 2,
        "summaries": {
            "part1_summary": summary_part1,
            "part2_summary": summary_part2,
            "final_combined_summary": final_summary
        },
        "execution_metrics": {
            "total_execution_time": execution_time,
            "openai_calls_made": 3,
            "logs_part1_chars": len(logs_part1),
            "logs_part2_chars": len(logs_part2),
            "final_summary_chars": len(final_summary)
        }
    })
    analyze_obs.end()

    # ---- Update state ----
    state["summary"] = final_summary
    state["part1_summary"] = summary_part1
    state["part2_summary"] = summary_part2

    return state

def display_summary(state: State) -> State:
    global workflow_trace
    print("✅ Log Analysis Results:\n")
    
    # Get all summaries
    final_summary = state.get("summary", "No final summary available")
    part1_summary = state.get("part1_summary", "No part 1 summary available")
    part2_summary = state.get("part2_summary", "No part 2 summary available")
    
    obs = workflow_trace.start_observation(
        name="display_summary", as_type="span",
        input={
            "final_summary_length": len(final_summary),
            "part1_summary_length": len(part1_summary),
            "part2_summary_length": len(part2_summary),
            "display_target": "console"
        },
        metadata={
            "function": "display_summary", 
            "output_method": "print",
            "analysis_method": "split_processing"
        }
    )
    
    # # Display the results
    # print("=" * 80)
    # print("🔍 PART 1 ANALYSIS:")
    # print("-" * 40)
    # print(part1_summary)
    # print("\n")
    
    # print("🔍 PART 2 ANALYSIS:")
    # print("-" * 40)
    # print(part2_summary)
    # print("\n")
    
    # print("🎯 FINAL COMBINED SUMMARY:")
    # print("-" * 40)
    # print(final_summary)
    # print("=" * 80)
    
    # Update observation with output details
    obs.update(output={
        "summaries_displayed": {
            "part1_displayed": True,
            "part2_displayed": True, 
            "final_summary_displayed": True
        },
        "display_method": "console_print",
        "total_characters_displayed": len(part1_summary) + len(part2_summary) + len(final_summary),
        "display_sections": 3
    })
    obs.end()
    return state

# ------------------ Workflow Graph ------------------
workflow = StateGraph(State)
workflow.add_node("fetch_logs", fetch_logs)
workflow.add_node("analyze_logs", analyze_logs)
workflow.add_node("display_summary", display_summary)

workflow.add_edge(START, "fetch_logs")
workflow.add_edge("fetch_logs", "analyze_logs")
workflow.add_edge("analyze_logs", "display_summary")
workflow.add_edge("display_summary", END)

# Compile workflow once
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# ------------------ API Schema ------------------
class WorkflowRequest(BaseModel):
    index: str = "k8s-logs"

# ------------------ API Endpoint ------------------
@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    """Trigger the log analysis workflow."""
    try:
        global workflow_trace
        workflow_start = time.time()

        # Create Langfuse observation for HTTP request
        http_obs = langfuse.start_observation(
            name="POST /run-workflow",
            as_type="span",
            input={
                "http_method": "POST",
                "endpoint": "/run-workflow",
                "request_body": request.dict(),
                "content_type": "application/json",
                "timestamp": workflow_start
            },
            metadata={
                "http_route": "/run-workflow",
                "service": "log-analyzer-agent",
                "request_type": "api_call"
            }
        )

        # Create Langfuse trace for workflow (child of HTTP request)
        workflow_trace = http_obs.start_observation(
            name="rca_agent_workflow",
            as_type="trace",
            input={
                "workflow_type": "elasticsearch_log_analysis", 
                "elasticsearch_index": request.index,
                "query_parameters": {
                    "size": 10,
                    "sort": [{"timestamp": {"order": "desc"}}],
                    "query": {"match_all": {}}
                },
                "timestamp": workflow_start,
                "user_request": request.dict()
            },
            metadata={
                "start_time": workflow_start, 
                "elasticsearch_index": request.index,
                "service": "log-analyzer-agent"
            }
        )

        # Run workflow
        final_state = app_graph.invoke(
            {"logs": "", "main_trace_id": workflow_trace.trace_id},
            config={"stream": False, "configurable": {"thread_id": "log_analysis_agent"}}
        )

        # Calculate execution metrics
        execution_time = time.time() - workflow_start
        summary = final_state.get("summary", "No summary available")
        logs_processed = len(final_state.get("logs", ""))

        # Update workflow trace
        workflow_trace.update(
            output={
                "workflow_completed": True,
                "execution_metrics": {
                    "total_execution_time_seconds": execution_time,
                    "logs_processed_chars": logs_processed,
                    "summary_length_chars": len(summary)
                },
                "results": {
                    "final_summary": summary,
                    "logs_found": logs_processed > 0,
                    "analysis_status": "completed"
                }
            },
            metadata={
                "status": "completed",
                "end_time": time.time(),
                "execution_duration": execution_time
            }
        )
        workflow_trace.end()

        # Prepare API response
        api_response = {
            "status": "success",
            "trace_id": workflow_trace.trace_id,
            "summary": summary,
            "execution_time": execution_time,
            "analysis_method": "split_processing",
            "details": {
                "part1_summary": final_state.get("part1_summary"),
                "part2_summary": final_state.get("part2_summary"),
                "final_combined_summary": summary,
                "openai_calls_made": 3
            }
        }

        # Update HTTP observation with output
        http_obs.update(
            output={
                "http_status_code": 200,
                "response_body": api_response,
                "execution_time_seconds": execution_time,
                "workflow_trace_id": workflow_trace.trace_id,
                "content_type": "application/json"
            },
            metadata={
                "status": "completed",
                "response_size_chars": len(str(api_response))
            }
        )
        http_obs.end()
        langfuse.flush()

        return api_response

    except Exception as e:
        # Prepare error response
        error_response = {"status": "error", "error": str(e)}
        
        # Update HTTP observation on error
        if 'http_obs' in locals():
            http_obs.update(
                output={
                    "http_status_code": 500,
                    "response_body": error_response,
                    "error_details": {
                        "error_message": str(e),
                        "error_type": type(e).__name__
                    }
                },
                metadata={"status": "error", "error": str(e)}
            )
            http_obs.end()
        
        # Update workflow trace on error
        if 'workflow_trace' in locals():
            workflow_trace.update(
                output={
                    "workflow_completed": False,
                    "error_details": {
                        "error_message": str(e),
                        "error_type": type(e).__name__,
                        "execution_time_before_error": time.time() - workflow_start if 'workflow_start' in locals() else 0
                    }
                },
                metadata={
                    "status": "error", 
                    "error": str(e),
                    "end_time": time.time()
                }
            )
            workflow_trace.end()
        
        langfuse.flush()
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
    
# the traces how flow the data to collector
# languages jaeger 
# state as input output
