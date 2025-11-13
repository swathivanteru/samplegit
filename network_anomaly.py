import streamlit as st
import os
import pandas as pd
import numpy as np
import json
import re
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_debug, set_verbose, set_llm_cache
import plotly.express as px
from langchain.agents import tool, initialize_agent, AgentType
from pymongo import MongoClient
import asyncio
from fastmcp import Client

# --- Stylish Header ---
st.markdown(
    """
    <style>
    /* Apply gradient background to entire app */
    .stApp {
        background: linear-gradient(to right, #ffffff, #ebeaf0);
        background-size: cover;
        min-height: 100vh;
    }

    /* Make text more readable, optional */
    .stApp * {
        color: #333333;  /* Change text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>🚀 Network Traffic Anomaly Dashboard</h1>", unsafe_allow_html=True)

# LangChain settings
set_llm_cache(InMemoryCache())
set_debug(True)
set_verbose(True)

# Set NVIDIA API Key
os.environ["NVIDIA_API_KEY"] = st.secrets["API_KEY"]

MCP_URL = "http://localhost:8000/sse"

# -------------------------------
# MCP Client Functions
# -------------------------------
async def fetch_devices_async(limit=100):
    async with Client(MCP_URL) as client:
        result = await client.call_tool("fetch_devices", {"request": {"limit": limit}})
        return result.structured_content.get("devices", [])

async def update_device_async(device_id, updates):
    async with Client(MCP_URL) as client:
        result = await client.call_tool("update_device", {"request": {"device_id": device_id, "updates": updates}})
        return result.structured_content
# -------------------------------
# Fetch Data from MCP (MongoDB)
# -------------------------------
devices = asyncio.run(fetch_devices_async(limit=500))
df2 = pd.DataFrame(devices)
if df2.empty:
    st.warning("No data found in MongoDB.")
    st.stop()

# Sidebar for row selection
st.sidebar.header("🔧 Row Selection")
row_start_2 = st.sidebar.number_input("Start Row: ", min_value=0, max_value=len(df2)-1, value=0)
row_end_2 = st.sidebar.number_input("End Row: ", min_value=row_start_2+1, max_value=len(df2), value=row_start_2+10)

# Prepare DataFrames
df2_plot = df2.iloc[row_start_2:row_end_2].copy()

if "device_id" not in df2_plot.columns:
    st.error("Dataset 2 does not have 'Device ID' column. Please check dataset.")
else:
    df2_plot["device_id"] = df2_plot["device_id"].astype(str)
selected_data_2 = df2_plot.to_dict(orient="records")

# Prompt Template
initial_prompt = PromptTemplate.from_template("""
You are a network security expert. Analyze the following two sets of network traffic data and identify any unusual or suspicious patterns, even if all rows are labeled benign.
focus on packet_loss_rate in Dataset2 {data2} ignore anomaly_label column


Dataset 2:
{data2}

Return ONLY valid JSON:
{{
  "dataset2_anomalies": ["device_ids from Dataset 2 that appear unusual"]
}}
If no anomalies are found, select the most unusual row based on numeric features.                                              
Do not include any explanation outside the JSON.
""")

# Initialize LLM
llm = ChatNVIDIA(
    model="meta/llama-3.1-405b-instruct",
    nvidia_api_key=os.environ["NVIDIA_API_KEY"],
    max_tokens=512,
    temperature=0.2
)

# LLM Invocation
with st.spinner("🔍 Analyzing data with LLM..."):
    initial_chain = initial_prompt | llm
    initial_response = initial_chain.invoke({"data2": selected_data_2}, config={"cache": False})

# Parse JSON
try:
    raw_response = str(initial_response.content)
    json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        llm_json = json.loads(json_str)
        anomalous_device_ids_2 = llm_json.get("dataset2_anomalies", [])
    else:
        st.error("No valid JSON found in LLM response.")
        anomalous_device_ids_2 = []
except Exception as e:
    st.error(f"Failed to parse LLM response: {e}")
    anomalous_device_ids_2 = []

# Display LLM Output
st.markdown(
    """
    <style>
    .stExpanderHeader {
        font-size: 20px;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
    """,
    unsafe_allow_html=True
)
# Tag anomalies

df2_plot["anomaly"] = df2_plot["device_id"].apply(
    lambda x: "Anomalous" if x in anomalous_device_ids_2 else "Normal"
)
# Anomaly log
@tool
def generate_anomaly_log(input: str) -> str:
    """
    Generate anomaly explanations and recommendations in JSON format.
    """
    return llm.invoke(input).content

# 3. Initialize Agent
# -------------------------------
agent = initialize_agent(
    tools=[generate_anomaly_log],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2
)
upper_quantile = df2_plot["packet_loss_rate"].quantile(0.90)
# Initialize log file

log_file = "anomaly_logs.csv"
with open(log_file, "w") as f:
    f.write("dataset,id,reason,occurrence,suggestion\n")

st.markdown("""
    <style>
    .status-box {
        border: 2px solid #228B22;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f9f9f9;
        font-family: Arial, sans-serif;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .status-title {
        font-size: 18px;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 10px;
    }
    .status-text {
        font-size: 16px;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.subheader("🔍 Real-Time Anomaly Detection & Logging (Dataset 2)")
status_placeholder = st.empty()

anomalous_ids = []
for i, row in enumerate(selected_data_2):
    device_id = row.get("device_id", "Unknown")
    packet_loss_rate = row.get("packet_loss_rate", 0)

    # Build HTML for the box
    html_content = f"""
    <div class="status-box">
        <div class="status-title">Processing Row {i+1}/{len(selected_data_2)}</div>
        <div class="status-text"><b>Device ID:</b> {device_id}</div>
        <div class="status-text"><b>Packet Loss Rate:</b> {packet_loss_rate}</div>
    </div>
    """
    status_placeholder.markdown(html_content, unsafe_allow_html=True)

    # Filter based on quantiles
    if packet_loss_rate <= upper_quantile:
        st.info(f"✅ Row {i+1} is normal.")
        continue

    # Build prompt with quantile context
    single_prompt = f"""
    You are a network security expert. Analyze the following row from Dataset 2.
    Rules:
    - If packet_loss_rate > 0.90, mark as anomaly.
    - If device_behavior_score < 0.3, mark as unhealthy.
    - If anomaly detected AND device is healthy (device_behavior_score >= 0.3), mark as blocked.

    Return ONLY valid JSON list. If the row is normal, return an empty list.
    Return ONLY valid JSON list. Do NOT include any explanation, reasoning, or text outside the JSON block.
    Format:
    [
      {{
        "dataset": "Dataset 2",
        "id": "device_id",
        "reason": "...",
        "occurrence": "...",
        "suggestion": "..."
      }}
    ]
    Row: {json.dumps(row)}
    """

    try:
        response = agent.run(single_prompt)
        # st.text(f"🔍 Raw response for {device_id}: {response}")

        with open("raw_llm_responses.txt", "a") as debug_log:
            debug_log.write(f"Device {device_id}: {response}\n")
        if "Agent stopped" in response:
            # st.warning(f"⚠️ Agent stopped for device_id: {device_id}. Retrying with simplified prompt...")
            response = llm.invoke(single_prompt).content
  
        if response.strip().startswith("["):
            # clean_json = extract_json(response)
            parsed = json.loads(response)
            if parsed :

                anomalous_ids.append(device_id)
                st.warning(f"⚠️ Anomaly detected in device_id: {parsed[0]['id']}")
                # is_anomalous = parsed[0]["is_anomalous"]
                # unhealthy = parsed[0]["unhealthy"]
                # blocked = parsed[0]["blocked"]
                
                # collection.update_one(
                #     {"device_id": device_id},
                #     {"$set": {
                #         "is_anomalous": is_anomalous,
                #         "unhealthy": unhealthy,
                #         "blocked": blocked
                #     }}
                # )
                with open(log_file, "a") as f:
                    for entry in parsed:
                        f.write(f"{entry['dataset']},{entry['id']},{entry['reason']},{entry['occurrence']},{entry['suggestion']}\n")
            else:
                st.info("✅ No anomaly detected.")
        else:
            st.warning(f"⚠️ Invalid response format for device_id: {device_id}")
        
    except json.JSONDecodeError:
        st.error(f"❌ JSON parsing error for device_id: {device_id}")
    except Exception as e:
        st.error(f"❌ Error processing row {i+1}: {e}")

try:
    anomaly_log = json.loads(response)
    log_json_str = json.dumps(anomaly_log, indent=2)
    st.success("📄 Anomaly log generated successfully!")

    # Download button
    st.download_button(
        label="📥 Download Anomaly Log",
        data=log_json_str,
        file_name="anomaly_log.json",
        mime="application/json"
    )
except Exception as e:
    st.error(f"Failed to parse anomaly log: {e}")

# Device Health check and restarting

# async def call_restart_tool(device_id: str, score: float):
#     async with Client("http://localhost:8000/sse") as client:
#         result = await client.call_tool("restart_device", {
#             "request": {  # Wrap arguments under 'request'
#                 "device_id": device_id,
#                 "device_behavior_score": score
#             }
#         })
#         return result.structured_content.get("restarted", False)
async def call_restart_tool(device_id, score):
    async with Client(MCP_URL) as client:
        result = await client.call_tool("restart_device", {"request": {"device_id": device_id, "device_behavior_score": score}})
        return result.structured_content.get("restarted", False)

def restart_device_async(device_id, score):
    try:
        return asyncio.run(call_restart_tool(device_id, score))
    except Exception:
        return False
df2_plot["is_anomalous"] = df2_plot["device_id"].isin(anomalous_ids)

if "device_behavior_score" in df2_plot.columns:
    df2_plot["unhealthy"] = df2_plot["device_behavior_score"] < 0.3
    df2_plot["restart_triggered"] = df2_plot.apply(
    lambda row: restart_device_async(row["device_id"], row["device_behavior_score"])
    if row.get("is_anomalous") and row.get("unhealthy") else False,
    axis=1
)
    df2_plot["blocked"] = df2_plot.apply(
    lambda row: True if row["is_anomalous"] and not row["unhealthy"] else False,
    axis=1
)
for _, row in df2_plot.iterrows():
    updates = {
        "is_anomalous": bool(row["is_anomalous"]),
        "unhealthy": bool(row["unhealthy"]),
        "blocked": bool(row["blocked"]),
        "restart_triggered": bool(row["restart_triggered"])
    }
    asyncio.run(update_device_async(row["device_id"], updates))

    # Display in Streamlit
st.subheader("🩺 Device Health & MCP Actions")
st.dataframe(df2_plot[[
        "device_id", "packet_loss_rate", "device_behavior_score",
        "is_anomalous", "unhealthy", "restart_triggered", "blocked"
    ]])

# Pie chart
anomalous_count = df2_plot["is_anomalous"].sum()
unhealthy_count = df2_plot["unhealthy"].sum()
blocked_count = df2_plot["blocked"].sum()
normal_count = len(df2_plot) - anomalous_count
labels = ["Anomalous", "Unhealthy", "Blocked", "Normal"]
values = [anomalous_count, unhealthy_count, blocked_count, normal_count]
fig = px.pie(names=labels, values=values, title="Device Status Distribution", hole=0.3)
fig.update_traces(textinfo='label+percent')
st.plotly_chart(fig)

st.dataframe(df2)
# 9. Severity Scoring Heatmap
visualization_option = st.sidebar.radio(
    "Select Visualization:",
    ["Severity Heatmap","Chat with Bot"]
)

# Convert selected_data_2 to DataFrame
df2 = pd.DataFrame(selected_data_2)
if visualization_option == "Severity Heatmap":
    if "packet_loss_rate" in df2.columns:
        # Calculate severity based on deviation from mean
        df2["severity"] = df2["packet_loss_rate"].apply(lambda x: x if x > 0.9 else 0)
        df2["severity_norm"] = (
            df2["severity"] / df2["severity"].max() if df2["severity"].max() > 0 else 0
        )

        # Prepare heatmap data
        heatmap_df2 = df2[["device_id", "severity_norm"]].rename(columns={
            "device_id": "ID",
            "severity_norm": "Severity"
        })
        heatmap_df2["Dataset"] = "Dataset 2"

        # Generate Plotly heatmap
        import plotly.express as px
        fig_heatmap = px.density_heatmap(
            heatmap_df2,
            x="ID",
            y="Dataset",
            z="Severity",
            color_continuous_scale="Reds",
            title="Anomaly Severity Heatmap"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Severity heatmap could not be generated. Missing numeric columns.")

# st.subheader("🚫 Blocked Devices Logic")

# if "device_behavior_score" in df2_plot.columns:
#     # Mark anomalies
#     df2_plot["is_anomalous"] = df2_plot["device_id"].isin(anomalous_ids)
#     df2_plot["healthy"] = df2_plot["device_behavior_score"] >= 0.3

#     # Initialize blocked column
#     df2_plot["blocked"] = False
#     df2_plot["block_reason"] = ""

#     for idx, row in df2_plot.iterrows():
#         if row["is_anomalous"] and row["healthy"]:
#             # Mark as blocked
#             df2_plot.at[idx, "blocked"] = True
#     st.subheader("🔒 Blocked Devices Table")
#     st.dataframe(df2_plot[["device_id", "is_anomalous","timestamp", "packet_loss_rate", "device_behavior_score", "unhealthy", "restart_triggered", "blocked"]])
# else:
#     st.warning("⚠️ 'device_behavior_score' column not found in Dataset 2.")
# --- Chat Section ---
elif visualization_option == "Severity Heatmap":
    st.subheader("💬 Chat with LLM")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question about the datasets...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        chat_prompt = PromptTemplate.from_template("""
    You are a network security expert. Based on the following datasets, answer the user's question.


    Dataset 2:
    {data2}

    User Question:
    {question}

    Respond clearly and concisely.
    """)

        chat_chain = chat_prompt | llm
        chat_response = chat_chain.invoke({
            "data2": selected_data_2,
            "question": user_input
        }, config={"cache": False})

        st.session_state.chat_history.append({"role": "assistant", "content": chat_response.content})

# Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])