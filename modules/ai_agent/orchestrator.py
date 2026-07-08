import streamlit as st
import json
import traceback
import numpy as np
from modules.ai_agent.gemini_client import GeminiClient
from modules.ai_agent.prompts import SYSTEM_INSTRUCTION
from modules.ai_agent.model_tools import train_and_evaluate_models
from modules.run_manager import update_metadata


def _make_serializable(obj):
    """Recursively convert numpy/pandas types to plain Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):  # generic numpy scalar
        return obj.item()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


def get_tool_map():
    return {
        "train_and_evaluate_models": train_and_evaluate_models,
    }


def run_ai_pipeline(
    api_key: str,
    cleaned_file_path: str,
    target_column: str,
    problem_type: str,
    profile: dict,
    artifacts: dict,
    cloud_sync: bool
):
    """Run the AI Agent pipeline for model training and evaluation.
    
    Returns a dict with keys: success, best_model, metrics, ai_summary
    """
    tool_map = get_tool_map()
    tools = list(tool_map.values())

    try:
        client = GeminiClient(api_key=api_key, tools=tools, system_instruction=SYSTEM_INSTRUCTION)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    # Build a concise profile summary for the AI
    profile_summary = {
        "rows": profile.get("rows", "unknown"),
        "columns": profile.get("columns", "unknown"),
        "numerical_columns": profile.get("numerical_columns", []),
        "categorical_columns": profile.get("categorical_columns", []),
        "missing_values": profile.get("total_missing", 0),
    }

    initial_prompt = f"""
A pre-cleaned dataset is ready for model training.

Dataset Profile:
{json.dumps(profile_summary, indent=2)}

Target Column: {target_column}
Detected Problem Type: {problem_type}

File Paths:
- Cleaned dataset: {cleaned_file_path}
- Save metrics to: {artifacts['metrics_csv']}
- Save best model to: {artifacts['best_model_joblib']}

Please analyze this profile, select ALL appropriate models for '{problem_type}', and execute the training tool.
Use the exact model names from your instructions.
"""

    with st.status("🤖 AI Agent is selecting and training models...", expanded=True) as status:
        try:
            from google.genai import types
            response = client.send_message(initial_prompt)
            max_iterations = 15
            iteration = 0
            ai_summary = ""
            best_model_name = None
            metrics_result = None
            raw_training_result = None

            while iteration < max_iterations:
                iteration += 1

                function_calls = []
                if hasattr(response, 'function_calls') and response.function_calls:
                    function_calls = response.function_calls

                if not function_calls:
                    # Extract final text
                    try:
                        ai_summary = response.text or ""
                    except (ValueError, AttributeError):
                        if hasattr(response, 'candidates') and response.candidates:
                            for part in response.candidates[0].content.parts:
                                if hasattr(part, 'text') and part.text:
                                    ai_summary += part.text

                    status.update(label="✅ AI Agent completed model training", state="complete")
                    break

                responses_for_agent = []
                for fc in function_calls:
                    fn_name = fc.name
                    fn_args = {}
                    if fc.args:
                        try:
                            fn_args = dict(fc.args)
                        except Exception:
                            fn_args = {k: v for k, v in fc.args.items()}

                    st.write(f"🛠️ **AI Tool Call:** `{fn_name}`")
                    st.caption(f"Arguments: {fn_args}")

                    try:
                        func = tool_map[fn_name]
                        result = func(**fn_args)
                        result_clean = _make_serializable(result)

                        # Capture metrics and full raw result for dashboard rendering
                        if fn_name == "train_and_evaluate_models" and isinstance(result_clean, dict):
                            if result_clean.get("status") == "success":
                                best_model_name = result_clean.get("best_model_name")
                                metrics_result = result_clean.get("metrics")
                                raw_training_result = result # Save the raw object with sklearn pipeline!

                        result_str = str(result_clean)[:500]
                        st.write(f"✅ **Result:** {result_str}")
                        responses_for_agent.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": result_clean}
                            )
                        )
                    except KeyError:
                        error_msg = f"Unknown tool: {fn_name}"
                        st.error(f"❌ {error_msg}")
                        responses_for_agent.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"error": error_msg}
                            )
                        )
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ **Error executing {fn_name}:** {error_msg}")
                        traceback.print_exc()
                        responses_for_agent.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"error": error_msg}
                            )
                        )

                status.update(label=f"AI Agent analyzing results... (step {iteration})", state="running")
                response = client.send_message(responses_for_agent)
            else:
                status.update(label="AI Agent reached maximum iterations", state="complete")

            # Update metadata
            if best_model_name:
                update_metadata(artifacts["base_dir"], {
                    "mode": "AI Agent",
                    "best_model": best_model_name,
                    "task_type": problem_type
                })

            return {
                "success": best_model_name is not None,
                "best_model": best_model_name,
                "metrics": metrics_result,
                "ai_summary": ai_summary,
                "raw_result": raw_training_result
            }

        except Exception as e:
            status.update(label="AI Agent Encountered an Error", state="error")
            st.error(f"Pipeline failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
