"""
Call Analysis Dashboard - Streamlit visualization for two-bot conversation analytics.

Visualizes metrics from call_history.json and analysis reports including:
- Infrastructure metrics (latency, connection times, TTFB)
- Application metrics (sentiment, message flow, duration)
- Historical trends across multiple calls
- Per-call detailed breakdowns

Run with: streamlit run dashboard.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from database import (
    get_call_history_for_dashboard,
    get_recordings_grouped_by_session,
    init_database,
)


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Call Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# DATA LOADING

def load_call_history(path: Optional[str] = None, use_sqlite: bool = True) -> dict:
    """Load call history data from SQLite database or JSON file."""
    # Try SQLite first if enabled
    if use_sqlite:
        try:
            init_database()
            data = get_call_history_for_dashboard()
            if data.get("total_calls", 0) > 0:
                return data
        except Exception as e:
            st.warning(f"SQLite load failed, falling back to JSON: {e}")

    # Fall back to JSON
    if path is None:
        path = Path(__file__).parent / "call_history.json"
    else:
        path = Path(path)

    if not path.exists():
        return {
            "total_calls": 0,
            "call_history": [],
            "all_calls_avg_latency": 0,
            "all_calls_avg_bot_connection_time": 0,
            "all_calls_avg_sentiment": 0,
            "all_calls_avg_duration": 0,
            "all_calls_avg_message_duration": 0,
            "all_calls_total_duration": 0,
            "topic_counts": {}
        }

    with open(path, "r") as f:
        return json.load(f)


def create_call_history_df(data: dict) -> pd.DataFrame:
    """Convert call history to a DataFrame."""
    if not data.get("call_history"):
        return pd.DataFrame()

    df = pd.DataFrame(data["call_history"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["call_number"] = range(1, len(df) + 1)
    return df


# HEADER & KEY METRICS

def render_header():
    """Render the dashboard header."""
    st.title("📊 Two-Bot Conversation Analysis Dashboard")
    st.markdown("""
    This dashboard visualizes metrics from automated conversations between two AI bots
    (Sarah - Customer and Marcus - Travel Agent). It helps you understand conversation quality,
    system performance, and trends over time.
    """)
    st.divider()


def render_key_metrics(data: dict, df: pd.DataFrame):
    """Render the key metrics overview section."""
    st.header("📈 Key Metrics Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Calls",
            value=data.get("total_calls", 0),
            help="Total number of conversations analyzed"
        )

    with col2:
        avg_duration = data.get("all_calls_avg_duration", 0)
        st.metric(
            label="Avg Duration",
            value=f"{avg_duration:.1f}s",
            delta=f"{avg_duration/60:.1f} min",
            help="Average conversation length"
        )

    with col3:
        avg_latency = data.get("all_calls_avg_latency", 0)
        st.metric(
            label="Avg Latency",
            value=f"{avg_latency:.2f}s",
            delta="Good" if avg_latency < 2 else "Needs improvement",
            delta_color="normal" if avg_latency < 2 else "inverse",
            help="Average time between user message and bot response"
        )

    with col4:
        avg_sentiment = data.get("all_calls_avg_sentiment", 0)
        sentiment_label = "Positive" if avg_sentiment > 0.3 else "Neutral" if avg_sentiment > -0.3 else "Negative"
        # Get sentiment emoji
        sentiment_emoji = "😊" if avg_sentiment > 0.3 else "😐" if avg_sentiment > -0.3 else "😟"
        st.metric(
            label="Avg Sentiment",
            value=f"{avg_sentiment:.2f}",
            delta=f"{sentiment_emoji} {sentiment_label}",
            delta_color="normal" if avg_sentiment > 0 else "inverse",
            help="Conversation sentiment score (-1 to 1)"
        )

    with col5:
        avg_conn = data.get("all_calls_avg_bot_connection_time", 0)
        st.metric(
            label="Avg Connection",
            value=f"{avg_conn:.2f}s",
            help="Average time for bots to connect"
        )

    # Show latest call's quality scores if available
    if not df.empty and "professionalism" in df.columns:
        st.markdown("---")
        st.subheader("Latest Call Quality Scores")
        latest = df.iloc[-1]

        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        with qcol1:
            prof = latest.get("professionalism", 0) * 100
            st.metric("Professionalism", f"{prof:.0f}%", help="How professional the conversation was")
        with qcol2:
            help_score = latest.get("helpfulness", 0) * 100
            st.metric("Helpfulness", f"{help_score:.0f}%", help="How helpful the responses were")
        with qcol3:
            eng = latest.get("engagement", 0) * 100
            st.metric("Engagement", f"{eng:.0f}%", help="How engaged both parties were")
        with qcol4:
            res = latest.get("resolution", 0) * 100
            st.metric("Resolution", f"{res:.0f}%", help="Whether issues were properly addressed")


# INFRASTRUCTURE METRICS TAB

def render_infrastructure_metrics(df: pd.DataFrame):
    """Render infrastructure-level metrics visualizations."""
    st.header("🔧 Infrastructure Metrics")

    st.markdown("""
    **What you're seeing:** These metrics measure the technical performance of the system -
    how quickly bots connect, respond, and maintain conversations. Lower values indicate
    better performance.
    """)

    if df.empty:
        st.warning("No call history data available yet. Run some conversations first!")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Message Latency Over Time
        fig = px.line(
            df,
            x="call_number",
            y="avg_latency",
            title="Message Latency Trend",
            labels={"call_number": "Call #", "avg_latency": "Latency (seconds)"},
            markers=True
        )
        fig.add_hline(
            y=df["avg_latency"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {df['avg_latency'].mean():.2f}s"
        )
        fig.update_layout(
            hovermode="x unified",
            yaxis_title="Latency (seconds)",
            xaxis_title="Call Number"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Message Latency** measures the time between when one bot finishes speaking
        and when the other bot responds. Lower latency = more natural conversation flow.
        """)

    with col2:
        # Bot Connection Time Over Time
        fig = px.bar(
            df,
            x="call_number",
            y="avg_bot_connection_time",
            title="Bot Connection Time by Call",
            labels={"call_number": "Call #", "avg_bot_connection_time": "Connection Time (seconds)"},
            color="avg_bot_connection_time",
            color_continuous_scale="RdYlGn_r"
        )
        fig.update_layout(
            yaxis_title="Connection Time (seconds)",
            xaxis_title="Call Number",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Bot Connection Time** shows how long it takes each bot to join the Daily room
        and become ready for conversation. Faster connections = quicker call starts.
        """)


# APPLICATION METRICS TAB

def render_app_level_metrics(df: pd.DataFrame):
    """Render application-level metrics visualizations."""
    st.header("💬 Application Metrics")

    st.markdown("""
    **What you're seeing:** These metrics measure the quality and characteristics of
    the actual conversations - how long they last, how positive they are, and how
    engaged the participants are.
    """)

    if df.empty:
        st.warning("No call history data available yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Sentiment Over Time with color coding
        fig = px.scatter(
            df,
            x="call_number",
            y="sentiment",
            size="message_count",
            color="sentiment",
            title="Conversation Sentiment by Call",
            labels={
                "call_number": "Call #",
                "sentiment": "Sentiment Score",
                "message_count": "Messages"
            },
            color_continuous_scale="RdYlGn",
            range_color=[-1, 1]
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Neutral")
        fig.update_layout(
            yaxis_title="Sentiment Score (-1 to 1)",
            xaxis_title="Call Number"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Sentiment Score** is analyzed by AI to determine the emotional tone of the conversation.
        - **Positive (> 0.3)**: Friendly, helpful, satisfied
        - **Neutral (-0.3 to 0.3)**: Professional, informational
        - **Negative (< -0.3)**: Frustrated, confused, unhappy

        Bubble size represents the number of messages exchanged.
        """)

    with col2:
        # Call Duration with Message Count
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=df["call_number"],
                y=df["duration_seconds"],
                name="Duration (s)",
                marker_color="steelblue"
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df["call_number"],
                y=df["message_count"],
                name="Messages",
                mode="lines+markers",
                marker_color="orange",
                line=dict(width=2)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Call Duration vs Message Count",
            xaxis_title="Call Number",
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Duration (seconds)", secondary_y=False)
        fig.update_yaxes(title_text="Message Count", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Duration vs Messages** helps identify conversation efficiency:
        - High messages + short duration = fast-paced, efficient
        - Low messages + long duration = may indicate pauses or issues
        - The ratio helps identify natural conversation flow
        """)


# SENTIMENT ANALYSIS TAB

def render_sentiment_details(df: pd.DataFrame):
    """Render detailed sentiment analysis visualizations."""
    st.header("😊 Sentiment Analysis")

    st.markdown("""
    **What you're seeing:** Detailed breakdown of conversation sentiment including
    quality dimensions and per-speaker analysis.
    """)

    if df.empty:
        st.warning("No call history data available yet.")
        return

    # Check if detailed sentiment data is available
    has_detailed = "professionalism" in df.columns

    col1, col2 = st.columns(2)

    with col1:
        if has_detailed:
            # Quality dimensions radar/bar chart
            latest_call = df.iloc[-1]
            quality_metrics = {
                "Professionalism": latest_call.get("professionalism", 0) * 100,
                "Helpfulness": latest_call.get("helpfulness", 0) * 100,
                "Engagement": latest_call.get("engagement", 0) * 100,
                "Resolution": latest_call.get("resolution", 0) * 100,
            }

            fig = go.Figure(go.Bar(
                x=list(quality_metrics.values()),
                y=list(quality_metrics.keys()),
                orientation='h',
                marker_color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'],
                text=[f"{v:.0f}%" for v in quality_metrics.values()],
                textposition='auto',
            ))
            fig.update_layout(
                title="Latest Call - Quality Dimensions",
                xaxis_title="Score (%)",
                yaxis_title="",
                xaxis_range=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("""
            **Quality Dimensions** break down sentiment into specific aspects:
            - **Professionalism**: How professional the conversation was
            - **Helpfulness**: How helpful the responses were
            - **Engagement**: How engaged both parties were
            - **Resolution**: Whether issues were properly addressed
            """)
        else:
            # Fallback: Sentiment trend line
            fig = px.line(
                df,
                x="call_number",
                y="sentiment",
                title="Sentiment Trend Over Time",
                labels={"call_number": "Call #", "sentiment": "Sentiment Score"},
                markers=True
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(yaxis_range=[-1, 1])
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if has_detailed:
            # Quality dimensions trend over time
            quality_cols = ["professionalism", "helpfulness", "engagement", "resolution"]
            available_cols = [c for c in quality_cols if c in df.columns]

            if available_cols and len(df) > 1:
                fig = go.Figure()
                colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
                for i, col in enumerate(available_cols):
                    fig.add_trace(go.Scatter(
                        x=df["call_number"],
                        y=df[col] * 100,
                        name=col.capitalize(),
                        mode="lines+markers",
                        line=dict(color=colors[i])
                    ))

                fig.update_layout(
                    title="Quality Dimensions Trend",
                    xaxis_title="Call Number",
                    yaxis_title="Score (%)",
                    yaxis_range=[0, 100],
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption("""
                **Quality Trend** shows how conversation quality metrics evolve over time.
                Consistent high scores indicate reliable service quality.
                """)
            else:
                # Speaker sentiment comparison for latest call
                speaker_data = latest_call.get("speaker_sentiments", {})
                if speaker_data:
                    speakers = list(speaker_data.keys())
                    scores = list(speaker_data.values())

                    fig = go.Figure(go.Bar(
                        x=speakers,
                        y=scores,
                        marker_color=['#3498db', '#e74c3c'],
                        text=[f"{s:.2f}" for s in scores],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title="Latest Call - Per-Speaker Sentiment",
                        xaxis_title="Speaker",
                        yaxis_title="Sentiment Score",
                        yaxis_range=[-1, 1],
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            # Sentiment distribution
            fig = px.histogram(
                df,
                x="sentiment",
                nbins=10,
                title="Sentiment Score Distribution",
                labels={"sentiment": "Sentiment Score", "count": "Frequency"},
                color_discrete_sequence=["#2ecc71"]
            )
            fig.update_layout(xaxis_range=[-1, 1])
            st.plotly_chart(fig, use_container_width=True)


# MESSAGE ANALYSIS TAB

def render_message_analysis(df: pd.DataFrame):
    """Render message-level analysis visualizations."""
    st.header("📝 Message Analysis")

    st.markdown("""
    **What you're seeing:** Detailed breakdown of message patterns and timing within
    conversations.
    """)

    if df.empty:
        st.warning("No call history data available yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Average Message Duration Trend
        fig = px.area(
            df,
            x="call_number",
            y="avg_message_duration",
            title="Average Message Duration Trend",
            labels={"call_number": "Call #", "avg_message_duration": "Duration (seconds)"},
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(
            yaxis_title="Avg Message Duration (seconds)",
            xaxis_title="Call Number"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Message Duration** shows the average time between consecutive messages.
        Consistent values suggest smooth turn-taking. Spikes may indicate processing
        delays or longer responses.
        """)

    with col2:
        # Messages per Call Distribution
        # Calculate appropriate bin size based on data range
        if len(df) > 0 and "message_count" in df.columns:
            msg_min = df["message_count"].min()
            msg_max = df["message_count"].max()
            msg_range = msg_max - msg_min

            # Use bins of size 5 for better readability
            nbins = max(5, int(msg_range / 5) + 1)

            fig = px.histogram(
                df,
                x="message_count",
                nbins=nbins,
                title="Message Count Distribution",
                color_discrete_sequence=["#00CC96"]
            )
            fig.update_layout(
                xaxis_title="Messages per Call",
                yaxis_title="Number of Calls",
                bargap=0.1
            )

            # Add mean line
            mean_msgs = df["message_count"].mean()
            fig.add_vline(
                x=mean_msgs,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {mean_msgs:.0f}"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            st.caption(f"""
            **Message Distribution** shows how many messages typically occur per call.
            - Min: {msg_min:.0f} | Max: {msg_max:.0f} | Avg: {mean_msgs:.1f}
            - A narrow distribution indicates consistent conversation lengths.
            """)
        else:
            st.info("No message count data available.")


# CUSTOMER TOPICS TAB

def render_customer_topics(data: dict, df: pd.DataFrame):
    """Render customer topics/questions analysis."""
    st.header("🏷️ Customer Topics")

    st.markdown("""
    **What you're seeing:** The most frequently asked topics by customers across all calls.
    This helps identify common customer interests and questions.
    """)

    topic_counts = data.get("topic_counts", {})

    if not topic_counts:
        st.info("No topic data available yet. Topics will be extracted from conversations after the next call.")
        return

    # Sort topics by count and get top 7
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:7]

    if not sorted_topics:
        st.info("No topics recorded yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart of top topics
        topics_df = pd.DataFrame(sorted_topics, columns=["Topic", "Count"])

        fig = go.Figure(go.Bar(
            x=topics_df["Count"],
            y=topics_df["Topic"],
            orientation='h',
            marker_color=px.colors.qualitative.Set2[:len(topics_df)],
            text=topics_df["Count"],
            textposition='auto',
        ))
        fig.update_layout(
            title="Top 7 Customer Topics",
            xaxis_title="Number of Mentions",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),  # Highest count at top
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Top Topics** shows the most frequently discussed subjects by customers.
        Higher counts indicate common areas of interest or concern.
        """)

    with col2:
        # Word cloud style visualization using a treemap
        all_topics = list(topic_counts.items())
        if len(all_topics) > 0:
            treemap_df = pd.DataFrame(all_topics, columns=["Topic", "Count"])

            fig = px.treemap(
                treemap_df,
                path=["Topic"],
                values="Count",
                title="Topic Frequency Map",
                color="Count",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=350,
            )
            fig.update_traces(
                textinfo="label+value",
                textfont_size=14,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("""
            **Topic Map** visualizes topic frequency - larger boxes indicate more frequently
            discussed topics. This helps quickly identify dominant customer interests.
            """)

    # Show recent call topics if available
    if not df.empty and "customer_topics" in df.columns:
        st.subheader("Recent Call Topics")
        recent_calls = df.tail(5)
        for _, row in recent_calls.iterrows():
            topics = row.get("customer_topics", [])
            if topics and isinstance(topics, list) and len(topics) > 0:
                timestamp = row.get("timestamp", "Unknown")
                if hasattr(timestamp, 'strftime'):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
                topic_tags = " ".join([f"`{t}`" for t in topics])
                st.markdown(f"**{timestamp}**: {topic_tags}")


# RECORDINGS TAB

def get_recordings(use_sqlite: bool = True) -> List[Dict]:
    """Get list of audio recordings from SQLite database or recordings directory."""
    # Try SQLite first if enabled
    if use_sqlite:
        try:
            init_database()
            recordings = get_recordings_grouped_by_session()
            if recordings:
                return recordings
        except Exception as e:
            st.warning(f"SQLite recordings load failed, falling back to filesystem: {e}")

    # Fall back to filesystem scan
    recordings_dir = Path(__file__).parent / "recordings"

    if not recordings_dir.exists():
        return []

    recordings = []
    wav_files = list(recordings_dir.glob("*.wav"))

    # Group recordings by timestamp
    recording_groups = {}
    for wav_file in wav_files:
        # Parse filename: type_timestamp.wav (e.g., merged_20240315_123456.wav)
        parts = wav_file.stem.split("_", 1)
        if len(parts) == 2:
            rec_type, timestamp = parts[0], parts[1]
            if timestamp not in recording_groups:
                recording_groups[timestamp] = {
                    "timestamp": timestamp,
                    "files": {},
                    "datetime": None
                }
                # Parse datetime from timestamp
                try:
                    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    recording_groups[timestamp]["datetime"] = dt
                except ValueError:
                    pass

            recording_groups[timestamp]["files"][rec_type] = {
                "path": str(wav_file),
                "size_kb": wav_file.stat().st_size / 1024,
                "name": wav_file.name
            }

    # Convert to sorted list (newest first)
    for timestamp, group in recording_groups.items():
        recordings.append(group)

    recordings.sort(key=lambda x: x["timestamp"], reverse=True)
    return recordings


def get_raw_log_content() -> Optional[str]:
    """Get the content of the raw conversation log file."""
    log_path = Path(__file__).parent / "two_bots.log"
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                return f.read()
        except Exception:
            return None
    return None


def render_recordings():
    """Render the audio recordings tab."""
    st.header("🎙️ Audio Recordings")

    st.markdown("""
    **What you're seeing:** Audio recordings from bot conversations. Each conversation
    produces up to three recordings:
    - **Merged**: Combined audio from both Sarah and Marcus
    - **Sarah**: Sarah's audio track only (the customer)
    - **Marcus**: Marcus's audio track only (the travel agent)
    """)

    recordings = get_recordings()

    if not recordings:
        st.info("No recordings available yet. Run a conversation to generate audio recordings.")
        st.markdown("""
        Recordings are automatically saved when conversations run. They will appear in the
        `recordings/` directory with the following naming convention:
        - `merged_YYYYMMDD_HHMMSS.wav` - Combined audio
        - `sarah_YYYYMMDD_HHMMSS.wav` - Sarah's audio
        - `marcus_YYYYMMDD_HHMMSS.wav` - Marcus's audio
        """)
        return

    st.success(f"Found {len(recordings)} recording session(s)")

    # Display recordings
    for i, recording in enumerate(recordings):
        timestamp = recording["timestamp"]
        dt = recording.get("datetime")
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else timestamp

        with st.expander(f"📅 Recording from {dt_str}", expanded=(i == 0)):
            files = recording["files"]

            col1, col2, col3 = st.columns(3)

            # Merged recording
            with col1:
                st.subheader("🔊 Merged")
                if "merged" in files:
                    file_info = files["merged"]
                    st.caption(f"Size: {file_info['size_kb']:.1f} KB")
                    try:
                        with open(file_info["path"], "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            label="Download",
                            data=audio_bytes,
                            file_name=file_info["name"],
                            mime="audio/wav",
                            key=f"download_merged_{timestamp}"
                        )
                    except Exception as e:
                        st.error(f"Error loading audio: {e}")
                else:
                    st.caption("Not available")

            # Sarah's recording
            with col2:
                st.subheader("👩 Sarah")
                if "sarah" in files:
                    file_info = files["sarah"]
                    st.caption(f"Size: {file_info['size_kb']:.1f} KB")
                    try:
                        with open(file_info["path"], "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            label="Download",
                            data=audio_bytes,
                            file_name=file_info["name"],
                            mime="audio/wav",
                            key=f"download_sarah_{timestamp}"
                        )
                    except Exception as e:
                        st.error(f"Error loading audio: {e}")
                else:
                    st.caption("Not available")

            # Marcus's recording
            with col3:
                st.subheader("👨 Marcus")
                if "marcus" in files:
                    file_info = files["marcus"]
                    st.caption(f"Size: {file_info['size_kb']:.1f} KB")
                    try:
                        with open(file_info["path"], "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            label="Download",
                            data=audio_bytes,
                            file_name=file_info["name"],
                            mime="audio/wav",
                            key=f"download_marcus_{timestamp}"
                        )
                    except Exception as e:
                        st.error(f"Error loading audio: {e}")
                else:
                    st.caption("Not available")

            # Show file paths for reference
            st.divider()
            st.caption("📁 **File Paths:**")
            for rec_type, file_info in files.items():
                st.caption(f"  {rec_type}: `{file_info['name']}`")

    # Raw Audio Log Section
    st.divider()
    st.subheader("📜 Raw Conversation Log")

    st.markdown("""
    **Note:** The raw log file (`two_bots.log`) is overwritten each conversation.
    The log shown below corresponds to the **most recent** conversation session.
    """)

    log_content = get_raw_log_content()

    if log_content:
        log_path = Path(__file__).parent / "two_bots.log"
        log_size_kb = log_path.stat().st_size / 1024
        log_lines = log_content.count('\n') + 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Log File", "two_bots.log")
        with col2:
            st.metric("Size", f"{log_size_kb:.1f} KB")
        with col3:
            st.metric("Lines", f"{log_lines:,}")

        # Log filtering options
        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            log_filter = st.selectbox(
                "Filter log by:",
                ["All", "TTS Only", "Transcriptions", "Errors/Warnings", "Bot Events"],
                key="log_filter"
            )

        # Apply filter
        if log_filter == "All":
            filtered_log = log_content
        elif log_filter == "TTS Only":
            filtered_log = "\n".join([
                line for line in log_content.split("\n")
                if "Generating TTS" in line or "TTFB" in line
            ])
        elif log_filter == "Transcriptions":
            filtered_log = "\n".join([
                line for line in log_content.split("\n")
                if "Forwarded transcription" in line or "TranscriptionFrame" in line
            ])
        elif log_filter == "Errors/Warnings":
            filtered_log = "\n".join([
                line for line in log_content.split("\n")
                if "ERROR" in line or "WARNING" in line or "error" in line.lower()
            ])
        elif log_filter == "Bot Events":
            filtered_log = "\n".join([
                line for line in log_content.split("\n")
                if "Sarah" in line or "Marcus" in line
            ])
        else:
            filtered_log = log_content

        if filtered_log.strip():
            # Show log in scrollable code block
            st.code(filtered_log, language="log", line_numbers=True)

            # Download button for full log
            st.download_button(
                label="📥 Download Full Log",
                data=log_content,
                file_name="two_bots.log",
                mime="text/plain",
                key="download_log"
            )
        else:
            st.info(f"No log entries match the '{log_filter}' filter.")
    else:
        st.warning("No raw log file found. Run a conversation to generate logs.")


# CORRELATION ANALYSIS TAB

def render_correlation_analysis(df: pd.DataFrame):
    """Render correlation and relationship analysis."""
    st.header("🔍 Relationship Analysis")

    st.markdown("""
    **What you're seeing:** How different metrics relate to each other. This helps
    identify patterns and potential areas for optimization.
    """)

    if df.empty or len(df) < 2:
        st.warning("Need at least 2 calls to show correlation analysis.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Latency vs Sentiment
        fig = px.scatter(
            df,
            x="avg_latency",
            y="sentiment",
            size="message_count",
            color="duration_seconds",
            title="Latency vs Sentiment",
            labels={
                "avg_latency": "Avg Latency (s)",
                "sentiment": "Sentiment Score",
                "message_count": "Messages",
                "duration_seconds": "Duration (s)"
            },
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            xaxis_title="Average Latency (seconds)",
            yaxis_title="Sentiment Score"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Latency vs Sentiment** explores whether faster responses correlate with
        better conversation sentiment. Ideally, lower latency should correlate with
        more positive sentiment (points cluster in bottom-right).
        """)

    with col2:
        # Correlation Heatmap
        numeric_cols = ["duration_seconds", "avg_latency", "avg_bot_connection_time",
                       "sentiment", "message_count", "avg_message_duration"]
        available_cols = [c for c in numeric_cols if c in df.columns]

        if len(available_cols) >= 2:
            corr_matrix = df[available_cols].corr()

            # Rename for display
            display_names = {
                "duration_seconds": "Duration",
                "avg_latency": "Latency",
                "avg_bot_connection_time": "Connection",
                "sentiment": "Sentiment",
                "message_count": "Messages",
                "avg_message_duration": "Msg Duration"
            }
            corr_matrix.index = [display_names.get(c, c) for c in corr_matrix.index]
            corr_matrix.columns = [display_names.get(c, c) for c in corr_matrix.columns]

            fig = px.imshow(
                corr_matrix,
                title="Metric Correlations",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                text_auto=".2f"
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("""
            **Correlation Matrix** shows relationships between metrics:
            - **+1 (Blue)**: Strong positive correlation (both increase together)
            - **0 (White)**: No correlation
            - **-1 (Red)**: Strong negative correlation (one increases, other decreases)
            """)


# WRITTEN ANALYSIS TAB

def load_analysis_reports() -> list[dict]:
    """Load all analysis markdown files from the directory."""
    reports_dir = Path(__file__).parent
    analysis_files = sorted(reports_dir.glob("analysis_*.md"), reverse=True)

    reports = []
    for filepath in analysis_files:
        # Extract timestamp from filename (analysis_YYYYMMDD_HHMMSS.md)
        filename = filepath.stem
        try:
            timestamp_str = filename.replace("analysis_", "")
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_time = "Unknown"

        with open(filepath, "r") as f:
            content = f.read()

        reports.append({
            "filename": filepath.name,
            "timestamp": formatted_time,
            "content": content
        })

    return reports


def render_written_analysis():
    """Render the written analysis reports tab."""
    st.header("📄 Written Analysis Reports")

    st.markdown("""
    **What you're seeing:** AI-generated analysis reports from conversation logs.
    These reports provide detailed insights into conversation quality, turn-taking,
    context handling, and technical performance.
    """)

    reports = load_analysis_reports()

    if not reports:
        st.warning("No analysis reports found. Run the analyzer to generate reports.")
        st.code("python call_analyzer.py", language="bash")
        return

    # Report selector
    report_options = [f"{r['timestamp']} - {r['filename']}" for r in reports]
    selected_idx = st.selectbox(
        "Select Analysis Report",
        range(len(report_options)),
        format_func=lambda x: report_options[x],
        help="Choose an analysis report to view. Reports are sorted by date (newest first)."
    )

    selected_report = reports[selected_idx]

    # Display metadata
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Report Date", selected_report["timestamp"].split()[0])
    with col2:
        st.metric("Report Time", selected_report["timestamp"].split()[1] if " " in selected_report["timestamp"] else "N/A")

    st.divider()

    # Display the markdown content
    st.markdown(selected_report["content"])

    # Download button
    st.divider()
    st.download_button(
        label="📥 Download Report",
        data=selected_report["content"],
        file_name=selected_report["filename"],
        mime="text/markdown"
    )


# HISTORICAL SUMMARY & SIDEBAR

def render_historical_summary(data: dict, df: pd.DataFrame):
    """Render historical summary and insights."""
    st.header("📊 Historical Summary")

    if df.empty:
        st.info("Start running conversations to see historical trends and insights.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Cumulative metrics over time
        df_sorted = df.sort_values("timestamp")
        df_sorted["cumulative_duration"] = df_sorted["duration_seconds"].cumsum()
        df_sorted["cumulative_messages"] = df_sorted["message_count"].cumsum()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df_sorted["timestamp"],
                y=df_sorted["cumulative_duration"] / 60,
                name="Total Duration (min)",
                fill="tozeroy",
                line=dict(color="steelblue")
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df_sorted["timestamp"],
                y=df_sorted["cumulative_messages"],
                name="Total Messages",
                line=dict(color="orange", dash="dash")
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Cumulative Usage Over Time",
            hovermode="x unified"
        )
        fig.update_xaxes(title_text="Date/Time")
        fig.update_yaxes(title_text="Total Duration (minutes)", secondary_y=False)
        fig.update_yaxes(title_text="Total Messages", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Quick Stats")

        total_duration_min = data.get("all_calls_total_duration", 0) / 60
        total_messages = df["message_count"].sum() if "message_count" in df.columns else 0

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Calls | **{data.get('total_calls', 0)}** |
        | Total Duration | **{total_duration_min:.1f} min** |
        | Total Messages | **{total_messages}** |
        | Avg Messages/Call | **{df['message_count'].mean():.1f}** |
        | Best Latency | **{df['avg_latency'].min():.2f}s** |
        | Worst Latency | **{df['avg_latency'].max():.2f}s** |
        """)

        # Performance insights
        st.subheader("Insights")

        insights = []

        if len(df) >= 2:
            # Trend analysis
            recent_latency = df.tail(2)["avg_latency"].mean()
            older_latency = df.head(len(df)//2)["avg_latency"].mean() if len(df) > 2 else recent_latency

            if recent_latency < older_latency * 0.9:
                insights.append("✅ Latency improving over time")
            elif recent_latency > older_latency * 1.1:
                insights.append("⚠️ Latency degrading - investigate")

            avg_sentiment = df["sentiment"].mean()
            if avg_sentiment > 0.5:
                insights.append("✅ Consistently positive sentiment")
            elif avg_sentiment < 0:
                insights.append("⚠️ Review conversation quality")

        if not insights:
            insights.append("📊 Collecting more data for insights...")

        for insight in insights:
            st.markdown(insight)


def render_sidebar(data: dict):
    """Render the sidebar with controls and info."""
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")

        # Data refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Data source indicator
        st.header("💾 Data Source")
        db_path = Path(__file__).parent / "call_analytics.db"
        if db_path.exists():
            st.success("SQLite Database Active")
            st.caption(f"DB: `call_analytics.db`")
            st.caption(f"Size: {db_path.stat().st_size / 1024:.1f} KB")
        else:
            st.info("Using JSON fallback")
            st.caption("DB will be created on next run")

        st.divider()

        st.header("📖 About This Dashboard")
        st.markdown("""
        This dashboard visualizes metrics from the **Two-Bot Conversation Demo** -
        an automated dialog system where:

        - **Sarah** (Customer) calls to inquire about a trip
        - **Marcus** (Travel Agent) provides information
        - A **Monitor** LLM decides when to end the call

        **Data Sources:**
        - `call_analytics.db` - SQLite database (primary)
        - `call_history.json` - JSON fallback
        - `two_bots.log` - Raw conversation logs
        - `analysis_*.md` - AI-generated analysis reports
        """)

        st.divider()

        st.header("🎯 Key Metrics Explained")

        with st.expander("Infrastructure Metrics"):
            st.markdown("""
            - **Latency**: Time between messages (lower = better)
            - **Connection Time**: Time to join Daily room
            - **TTFB**: Time to first byte for LLM/TTS
            """)

        with st.expander("App-Level Metrics"):
            st.markdown("""
            - **Sentiment**: Conversation emotional tone (-1 to +1)
            - **Duration**: Total call length
            - **Messages**: Number of conversation turns
            """)

        st.divider()

        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"Total calls in database: {data.get('total_calls', 0)}")


# MAIN ENTRY POINT

def main():
    """Main dashboard entry point."""
    # Load data
    data = load_call_history()
    df = create_call_history_df(data)

    # Render components
    render_sidebar(data)
    render_header()
    render_key_metrics(data, df)

    st.divider()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔧 Infrastructure",
        "💬 Application",
        "😊 Sentiment",
        "📝 Messages",
        "🏷️ Topics",
        "🎙️ Recordings",
        "🔍 Correlations",
        "📄 Written Analysis"
    ])

    with tab1:
        render_infrastructure_metrics(df)

    with tab2:
        render_app_level_metrics(df)

    with tab3:
        render_sentiment_details(df)

    with tab4:
        render_message_analysis(df)

    with tab5:
        render_customer_topics(data, df)

    with tab6:
        render_recordings()

    with tab7:
        render_correlation_analysis(df)

    with tab8:
        render_written_analysis()

    st.divider()
    render_historical_summary(data, df)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **Two-Bot Conversation Analysis Dashboard** | Built with Streamlit & Plotly
    Data generated by `call_analyzer.py` from Pipecat conversation logs.
    """)


if __name__ == "__main__":
    main()
