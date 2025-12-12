"""
Call Analyzer - Post-session analysis of two-bot conversations.

Reads session logs from two_bots.log, extracts conversation data, and uses OpenAI to
generate an insightful analysis of the call including:
- Conversation flow and turn-taking
- Topic coverage and context handling
- Technical metrics (latency, TTS/LLM performance)
- Overall quality assessment

Adapted for loguru-formatted logs from the two_bots demo.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import openai
from dotenv import load_dotenv

from database import init_database, add_call as db_add_call

load_dotenv(override=True)

log = logging.getLogger(__name__)


# =============================================================================
# REGEX PATTERNS FOR LOG PARSING
# =============================================================================

class LogPatterns:
    """Regex patterns for parsing loguru-formatted logs."""

    # Timestamp: HH:MM:SS or HH:MM:SS.mmm
    TIMESTAMP = r"(\d{2}:\d{2}:\d{2}(?:\.\d{3})?)"

    # TTS generation: "Sarah: Generating TTS [Hello there]"
    TTS_GENERATION = r"(\w+): Generating TTS \[([^\]]+)\]"

    # LLM generation: "OpenAILLMService#0: Generating chat from universal context"
    LLM_GENERATION = r"OpenAILLMService#(\d+): Generating chat from universal context"

    # Bot speaking events
    BOT_STARTED_SPEAKING = r"Bot started speaking"
    BOT_STOPPED_SPEAKING = r"Bot stopped speaking"
    USER_INTERRUPTION = r"User started speaking"

    # Performance metrics
    LLM_TTFB = r"OpenAILLMService#(\d+) TTFB: ([\d.]+)"
    TTS_TTFB = r"(\w+) TTFB: ([\d.]+)"
    LLM_TOKENS = r"OpenAILLMService#(\d+) prompt tokens: (\d+), completion tokens: (\d+)"
    TTS_CHARS = r"(\w+) usage characters: (\d+)"

    # Bot identification: "Sarah | Starting bot"
    BOT_NAME_LOG = r"\| ([A-Za-z]+)\s+\| (.+)$"
    BOT_START = r"(\w+) \| Starting bot"

    # Participant events
    PARTICIPANT_JOINED = r"First participant joined: (\w+)"
    PARTICIPANT_LEFT = r"Participant left: (\w+)"

    # Connection tracking
    JOINING = r"Joining (https://[^\s]+)"
    JOINED = r"Joined (https://[^\s]+)"

    # Transcription forwarding for latency measurement
    TRANSCRIPTION_FORWARDED = r"(\w+) \| Forwarded transcription from (\w+):\s*(.+)$"


# Follow-up question indicators for conversation analysis
FOLLOW_UP_INDICATORS = [
    r"\?",
    r"what about",
    r"how about",
    r"could you",
    r"can you",
    r"tell me more",
    r"and what",
    r"also",
    r"additionally",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: datetime
    speaker: str  # The bot that generated this TTS
    text: str  # The actual spoken text
    duration_seconds: float = 0.0
    is_follow_up_question: bool = False


@dataclass
class InfrastructureMetrics:
    """Infrastructure-level metrics for a call."""
    # Message latency (time between user message and bot response)
    message_latencies: List[float] = field(default_factory=list)
    avg_message_latency: float = 0.0

    # Bot connection times (how long it takes each bot to connect)
    bot_connection_times: Dict[str, float] = field(default_factory=dict)
    avg_bot_connection_time: float = 0.0

    # LLM and TTS performance
    avg_llm_ttfb: float = 0.0
    avg_tts_ttfb: float = 0.0


@dataclass
class SentimentAnalysis:
    """Detailed sentiment analysis for a conversation."""
    # Overall sentiment (-1 to 1 scale: negative to positive)
    overall_score: float = 0.0
    overall_label: str = "neutral"  # negative, neutral, positive

    # Per-speaker sentiment
    speaker_sentiments: Dict[str, float] = field(default_factory=dict)
    speaker_labels: Dict[str, str] = field(default_factory=dict)

    # Sentiment breakdown by category (0.0 to 1.0)
    professionalism: float = 0.0
    helpfulness: float = 0.0
    engagement: float = 0.0
    resolution: float = 0.0

    # Key emotional moments
    positive_highlights: List[str] = field(default_factory=list)
    negative_highlights: List[str] = field(default_factory=list)

    # Customer topics/questions asked (e.g., "beaches", "hotels", "prices")
    customer_topics: List[str] = field(default_factory=list)


@dataclass
class AppLevelMetrics:
    """Application-level metrics for a call."""
    # Conversation sentiment (-1 to 1 scale: negative to positive)
    conversation_sentiment: float = 0.0
    sentiment_label: str = "neutral"  # negative, neutral, positive

    # Detailed sentiment analysis
    sentiment_analysis: Optional[SentimentAnalysis] = None

    # Follow-up questions
    follow_up_question_count: int = 0
    follow_up_questions: List[str] = field(default_factory=list)

    # Message counts
    messages_sent: int = 0  # Bot messages
    messages_received: int = 0  # User/other bot messages

    # Duration metrics
    call_duration_seconds: float = 0.0
    avg_message_duration_seconds: float = 0.0
    message_durations: List[float] = field(default_factory=list)


@dataclass
class HistoricalMetrics:
    """Aggregated metrics across all calls."""
    total_calls: int = 0

    # Infrastructure averages across all calls
    all_calls_avg_latency: float = 0.0
    all_calls_avg_bot_connection_time: float = 0.0

    # App-level averages across all calls
    all_calls_avg_sentiment: float = 0.0
    all_calls_total_duration: float = 0.0
    all_calls_avg_duration: float = 0.0
    all_calls_avg_message_duration: float = 0.0

    # Historical data for trend analysis
    call_history: List[Dict[str, Any]] = field(default_factory=list)

    # Customer topic frequency tracking (topic -> count)
    topic_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class CallMetrics:
    """Aggregated metrics from the call."""
    total_duration_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Per-bot metrics
    bot_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Turn-taking metrics
    total_turns: int = 0
    interruptions: int = 0

    # Performance metrics
    llm_ttfb_values: List[float] = field(default_factory=list)
    tts_ttfb_values: List[float] = field(default_factory=list)

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Enhanced metrics
    infrastructure: InfrastructureMetrics = field(default_factory=InfrastructureMetrics)
    app_level: AppLevelMetrics = field(default_factory=AppLevelMetrics)


# =============================================================================
# HISTORICAL DATA STORAGE
# =============================================================================

class HistoricalDataStore:
    """Manages persistent storage of call metrics for historical analysis."""

    DEFAULT_STORE_FILE = "call_history.json"

    def __init__(self, store_path: Optional[str] = None):
        """Initialize the data store."""
        if store_path:
            self.store_path = Path(store_path)
        else:
            self.store_path = Path(__file__).parent / self.DEFAULT_STORE_FILE

    def load(self) -> HistoricalMetrics:
        """Load historical metrics from storage."""
        if not self.store_path.exists():
            return HistoricalMetrics()

        try:
            with open(self.store_path, "r") as f:
                data = json.load(f)
            return HistoricalMetrics(
                total_calls=data.get("total_calls", 0),
                all_calls_avg_latency=data.get("all_calls_avg_latency", 0.0),
                all_calls_avg_bot_connection_time=data.get("all_calls_avg_bot_connection_time", 0.0),
                all_calls_avg_sentiment=data.get("all_calls_avg_sentiment", 0.0),
                all_calls_total_duration=data.get("all_calls_total_duration", 0.0),
                all_calls_avg_duration=data.get("all_calls_avg_duration", 0.0),
                all_calls_avg_message_duration=data.get("all_calls_avg_message_duration", 0.0),
                call_history=data.get("call_history", []),
                topic_counts=data.get("topic_counts", {})
            )
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load historical data: {e}")
            return HistoricalMetrics()

    def save(self, historical: HistoricalMetrics):
        """Save historical metrics to storage."""
        data = {
            "total_calls": historical.total_calls,
            "all_calls_avg_latency": historical.all_calls_avg_latency,
            "all_calls_avg_bot_connection_time": historical.all_calls_avg_bot_connection_time,
            "all_calls_avg_sentiment": historical.all_calls_avg_sentiment,
            "all_calls_total_duration": historical.all_calls_total_duration,
            "all_calls_avg_duration": historical.all_calls_avg_duration,
            "all_calls_avg_message_duration": historical.all_calls_avg_message_duration,
            "call_history": historical.call_history[-100:],  # Keep last 100 calls
            "topic_counts": historical.topic_counts
        }
        try:
            with open(self.store_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            log.error(f"Failed to save historical data: {e}")

    def add_call(self, metrics: CallMetrics, sentiment: SentimentAnalysis) -> HistoricalMetrics:
        """Add a new call's metrics and update historical averages."""
        historical = self.load()

        # Create call record with detailed sentiment data
        timestamp = metrics.start_time.isoformat() if metrics.start_time else datetime.now().isoformat()
        call_record = {
            "timestamp": timestamp,
            "duration_seconds": metrics.total_duration_seconds,
            "avg_latency": metrics.infrastructure.avg_message_latency,
            "avg_bot_connection_time": metrics.infrastructure.avg_bot_connection_time,
            "sentiment": sentiment.overall_score,
            "sentiment_label": sentiment.overall_label,
            "message_count": metrics.total_turns,
            "avg_message_duration": metrics.app_level.avg_message_duration_seconds,
            # Detailed sentiment breakdown
            "professionalism": sentiment.professionalism,
            "helpfulness": sentiment.helpfulness,
            "engagement": sentiment.engagement,
            "resolution": sentiment.resolution,
            "speaker_sentiments": sentiment.speaker_sentiments,
            # Customer topics from this call
            "customer_topics": sentiment.customer_topics,
        }
        historical.call_history.append(call_record)

        # Update topic counts
        for topic in sentiment.customer_topics:
            topic_lower = topic.lower()
            historical.topic_counts[topic_lower] = historical.topic_counts.get(topic_lower, 0) + 1

        # Update running averages
        n = historical.total_calls
        historical.total_calls += 1

        # Incremental average formula: new_avg = old_avg + (new_value - old_avg) / n
        if n == 0:
            historical.all_calls_avg_latency = metrics.infrastructure.avg_message_latency
            historical.all_calls_avg_bot_connection_time = metrics.infrastructure.avg_bot_connection_time
            historical.all_calls_avg_sentiment = sentiment.overall_score
            historical.all_calls_avg_duration = metrics.total_duration_seconds
            historical.all_calls_avg_message_duration = metrics.app_level.avg_message_duration_seconds
        else:
            historical.all_calls_avg_latency += (metrics.infrastructure.avg_message_latency - historical.all_calls_avg_latency) / (n + 1)
            historical.all_calls_avg_bot_connection_time += (metrics.infrastructure.avg_bot_connection_time - historical.all_calls_avg_bot_connection_time) / (n + 1)
            historical.all_calls_avg_sentiment += (sentiment.overall_score - historical.all_calls_avg_sentiment) / (n + 1)
            historical.all_calls_avg_duration += (metrics.total_duration_seconds - historical.all_calls_avg_duration) / (n + 1)
            historical.all_calls_avg_message_duration += (metrics.app_level.avg_message_duration_seconds - historical.all_calls_avg_message_duration) / (n + 1)

        historical.all_calls_total_duration += metrics.total_duration_seconds

        # Save to JSON (legacy)
        self.save(historical)

        # Also save to SQLite database
        try:
            init_database()
            db_add_call(
                timestamp=timestamp,
                duration_seconds=metrics.total_duration_seconds,
                avg_latency=metrics.infrastructure.avg_message_latency,
                avg_bot_connection_time=metrics.infrastructure.avg_bot_connection_time,
                sentiment=sentiment.overall_score,
                sentiment_label=sentiment.overall_label,
                message_count=metrics.total_turns,
                avg_message_duration=metrics.app_level.avg_message_duration_seconds,
                professionalism=sentiment.professionalism,
                helpfulness=sentiment.helpfulness,
                engagement=sentiment.engagement,
                resolution=sentiment.resolution,
                speaker_sentiments=sentiment.speaker_sentiments,
                customer_topics=sentiment.customer_topics
            )
            log.info("Call data saved to SQLite database")
        except Exception as e:
            log.warning(f"Failed to save call data to SQLite: {e}")

        return historical


# =============================================================================
# CALL ANALYZER
# =============================================================================

class CallAnalyzer:
    """
    Analyzes session logs to provide insights about bot conversations.

    Parses loguru-formatted logs from the two_bots demo and extracts:
    - Conversation turns and timing
    - Performance metrics (TTFB, latency)
    - Sentiment analysis via OpenAI
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """
        Initialize the analyzer.

        Args:
            openai_api_key: OpenAI API key for generating analysis
            model: OpenAI model to use for analysis
        """
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.service_to_bot: Dict[int, str] = {}
        self.historical_store = HistoricalDataStore()

    def find_session_log(self, logs_dir: str) -> Optional[Path]:
        """
        Find the session log file.

        Args:
            logs_dir: Directory containing session logs

        Returns:
            Path to the session log file, or None if not found
        """
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            log.warning(f"Logs directory not found: {logs_dir}")
            return None

        # Look for two_bots.log first (our demo format)
        two_bots_log = logs_path / "two_bots.log"
        if two_bots_log.exists() and two_bots_log.stat().st_size > 100:
            return two_bots_log

        # Fall back to session.log
        session_log = logs_path / "session.log"
        if session_log.exists() and session_log.stat().st_size > 100:
            return session_log

        log.warning(f"No session logs found in {logs_dir}")
        return None

    def parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse a timestamp string into a datetime object."""
        from datetime import date
        today = date.today()
        try:
            # Try HH:MM:SS.mmm format first
            t = datetime.strptime(ts_str, "%H:%M:%S.%f")
            return t.replace(year=today.year, month=today.month, day=today.day)
        except ValueError:
            try:
                # Try HH:MM:SS format
                t = datetime.strptime(ts_str, "%H:%M:%S")
                return t.replace(year=today.year, month=today.month, day=today.day)
            except ValueError:
                try:
                    # Try full datetime format as fallback
                    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    try:
                        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        return None

    def parse_logs(self, log_content: str) -> tuple[List[ConversationTurn], CallMetrics]:
        """
        Parse log content and extract conversation data.

        Args:
            log_content: Raw log file content

        Returns:
            Tuple of (conversation_turns, metrics)
        """
        turns = []
        metrics = CallMetrics()
        patterns = LogPatterns

        lines = log_content.split("\n")
        bot_names: List[str] = []
        speaking_start: Dict[int, datetime] = {}
        joining_time: Optional[datetime] = None
        last_transcription_time: Optional[datetime] = None

        for line in lines:
            # Extract timestamp
            ts_match = re.search(patterns.TIMESTAMP, line)
            timestamp = None
            if ts_match:
                timestamp = self.parse_timestamp(ts_match.group(1))
                if timestamp:
                    if metrics.start_time is None:
                        metrics.start_time = timestamp
                    metrics.end_time = timestamp

            # Track bot names from "Starting bot" messages
            bot_start_match = re.search(patterns.BOT_START, line)
            if bot_start_match:
                bot_name = bot_start_match.group(1)
                if bot_name not in bot_names:
                    bot_names.append(bot_name)
                    self.service_to_bot[len(bot_names) - 1] = bot_name
                    metrics.bot_metrics[bot_name] = {
                        "speaking_time": 0.0,
                        "turn_count": 0,
                        "tts_characters": 0,
                        "llm_tokens": {"prompt": 0, "completion": 0},
                    }

            # Track TTS generation (what the bot actually said)
            tts_match = re.search(patterns.TTS_GENERATION, line)
            if tts_match and timestamp:
                speaker = tts_match.group(1)
                text = tts_match.group(2)

                # Initialize bot metrics if needed
                if speaker not in metrics.bot_metrics:
                    metrics.bot_metrics[speaker] = {
                        "speaking_time": 0.0,
                        "turn_count": 0,
                        "tts_characters": 0,
                        "llm_tokens": {"prompt": 0, "completion": 0},
                    }

                turns.append(ConversationTurn(timestamp=timestamp, speaker=speaker, text=text))
                metrics.total_turns += 1
                metrics.bot_metrics[speaker]["turn_count"] += 1

            # Track bot speaking start/stop
            if re.search(patterns.BOT_STARTED_SPEAKING, line) and timestamp:
                for idx in self.service_to_bot:
                    speaking_start[idx] = timestamp

            if re.search(patterns.BOT_STOPPED_SPEAKING, line) and timestamp:
                for idx, start in list(speaking_start.items()):
                    duration = (timestamp - start).total_seconds()
                    bot_name = self.service_to_bot.get(idx)
                    if bot_name and bot_name in metrics.bot_metrics:
                        metrics.bot_metrics[bot_name]["speaking_time"] += duration
                speaking_start.clear()

            # Track interruptions
            if re.search(patterns.USER_INTERRUPTION, line):
                metrics.interruptions += 1

            # Track LLM TTFB
            llm_ttfb_match = re.search(patterns.LLM_TTFB, line)
            if llm_ttfb_match:
                metrics.llm_ttfb_values.append(float(llm_ttfb_match.group(2)))

            # Track TTS TTFB
            tts_ttfb_match = re.search(patterns.TTS_TTFB, line)
            if tts_ttfb_match:
                bot_name = tts_ttfb_match.group(1)
                if bot_name in metrics.bot_metrics:
                    metrics.tts_ttfb_values.append(float(tts_ttfb_match.group(2)))

            # Track LLM token usage
            tokens_match = re.search(patterns.LLM_TOKENS, line)
            if tokens_match:
                service_idx = int(tokens_match.group(1))
                bot_name = self.service_to_bot.get(service_idx)
                if bot_name and bot_name in metrics.bot_metrics:
                    metrics.bot_metrics[bot_name]["llm_tokens"]["prompt"] += int(tokens_match.group(2))
                    metrics.bot_metrics[bot_name]["llm_tokens"]["completion"] += int(tokens_match.group(3))

            # Track TTS character usage
            chars_match = re.search(patterns.TTS_CHARS, line)
            if chars_match:
                bot_name = chars_match.group(1)
                if bot_name in metrics.bot_metrics:
                    metrics.bot_metrics[bot_name]["tts_characters"] += int(chars_match.group(2))

            # Track bot connection time (Joining -> Joined)
            if re.search(patterns.JOINING, line) and timestamp:
                joining_time = timestamp

            if re.search(patterns.JOINED, line) and timestamp and joining_time:
                connection_time = (timestamp - joining_time).total_seconds()
                if 0 < connection_time < 30:
                    metrics.infrastructure.message_latencies.append(connection_time)
                    if "bot_connection" not in metrics.infrastructure.bot_connection_times:
                        metrics.infrastructure.bot_connection_times["bot_connection"] = []
                    metrics.infrastructure.bot_connection_times["bot_connection"].append(connection_time)
                joining_time = None

            # Track transcription forwarding for message latency
            transcription_match = re.search(patterns.TRANSCRIPTION_FORWARDED, line)
            if transcription_match and timestamp:
                last_transcription_time = timestamp

            # Track LLM generation start as response to transcription
            if re.search(patterns.LLM_GENERATION, line) and timestamp and last_transcription_time:
                latency = (timestamp - last_transcription_time).total_seconds()
                if 0 < latency < 30:
                    metrics.infrastructure.message_latencies.append(latency)
                last_transcription_time = None

            # Track errors and warnings
            if "ERROR" in line:
                metrics.errors.append(line.strip())
            if len(metrics.warnings) < 20 and "WARNING" in line:
                metrics.warnings.append(line.strip())

        # Calculate total duration
        if metrics.start_time and metrics.end_time:
            metrics.total_duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()

        # Calculate average message latency
        if metrics.infrastructure.message_latencies:
            metrics.infrastructure.avg_message_latency = (
                sum(metrics.infrastructure.message_latencies) / len(metrics.infrastructure.message_latencies)
            )

        # Calculate average bot connection time
        connection_times = metrics.infrastructure.bot_connection_times.get("bot_connection", [])
        if connection_times:
            metrics.infrastructure.avg_bot_connection_time = sum(connection_times) / len(connection_times)

        # Calculate message durations (time between consecutive messages)
        if len(turns) > 1:
            message_durations = []
            for i in range(1, len(turns)):
                if turns[i].timestamp and turns[i-1].timestamp:
                    duration = (turns[i].timestamp - turns[i-1].timestamp).total_seconds()
                    # Only include reasonable durations (0 to 60 seconds)
                    if 0 < duration < 60:
                        message_durations.append(duration)

            if message_durations:
                metrics.app_level.message_durations = message_durations
                metrics.app_level.avg_message_duration_seconds = sum(message_durations) / len(message_durations)

        return turns, metrics

    def build_conversation_transcript(self, turns: List[ConversationTurn]) -> str:
        """Build a readable transcript from conversation turns."""
        transcript_lines = []
        current_speaker = None
        current_text_parts = []
        current_time = None

        for turn in turns:
            if turn.speaker == current_speaker:
                # Same speaker continues, append text
                current_text_parts.append(turn.text)
            else:
                # New speaker, flush previous
                if current_speaker and current_text_parts:
                    time_str = current_time.strftime("%H:%M:%S") if current_time else "??:??:??"
                    full_text = " ".join(current_text_parts)
                    transcript_lines.append(f"[{time_str}] {current_speaker}: \"{full_text}\"")

                current_speaker = turn.speaker
                current_text_parts = [turn.text]
                current_time = turn.timestamp

        # Flush last speaker
        if current_speaker and current_text_parts:
            time_str = current_time.strftime("%H:%M:%S") if current_time else "??:??:??"
            full_text = " ".join(current_text_parts)
            transcript_lines.append(f"[{time_str}] {current_speaker}: \"{full_text}\"")

        return "\n".join(transcript_lines)

    def analyze_sentiment(self, turns: List[ConversationTurn]) -> SentimentAnalysis:
        """
        Perform detailed sentiment analysis on the conversation.

        Returns:
            SentimentAnalysis object with comprehensive sentiment data
        """
        analysis = SentimentAnalysis()
        transcript = self.build_conversation_transcript(turns)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert sentiment analyzer for customer service conversations.
Analyze the conversation and respond with ONLY a JSON object in this exact format:
{
    "overall_score": <float -1.0 to 1.0>,
    "overall_label": "<negative|neutral|positive>",
    "speaker_sentiments": {
        "<speaker_name>": <float -1.0 to 1.0>
    },
    "professionalism": <float 0.0 to 1.0>,
    "helpfulness": <float 0.0 to 1.0>,
    "engagement": <float 0.0 to 1.0>,
    "resolution": <float 0.0 to 1.0>,
    "positive_highlights": ["<quote>"],
    "negative_highlights": ["<quote>"],
    "customer_topics": ["<topic>"]
}

Score guidelines:
- overall_score: -1.0 (very negative) to 1.0 (very positive)
- professionalism: How professional the conversation was
- helpfulness: How helpful the responses were
- engagement: How engaged both parties were
- resolution: Whether issues were resolved (1.0 = fully resolved)
- highlights: Short quotes (max 50 chars) showing emotional moments
- customer_topics: List of 1-7 key topics/subjects the CUSTOMER asked about or expressed interest in.
  Examples: "hotel", "price", "dates", "beaches", "restaurants", "activities", "transportation", "weather", "duration", "amenities", "location", "booking", "cancellation", "reviews"
  Use single lowercase words. Only include topics actually discussed by the customer."""
                    },
                    {"role": "user", "content": f"Analyze this conversation:\n\n{transcript[:4000]}"}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result = response.choices[0].message.content.strip()
            # Clean up potential markdown formatting
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            result = result.strip()

            data = json.loads(result)

            # Populate analysis object
            analysis.overall_score = max(-1.0, min(1.0, float(data.get("overall_score", 0.0))))
            analysis.overall_label = data.get("overall_label", "neutral")

            # Per-speaker sentiments
            speaker_data = data.get("speaker_sentiments", {})
            for speaker, score in speaker_data.items():
                analysis.speaker_sentiments[speaker] = max(-1.0, min(1.0, float(score)))
                if score > 0.3:
                    analysis.speaker_labels[speaker] = "positive"
                elif score < -0.3:
                    analysis.speaker_labels[speaker] = "negative"
                else:
                    analysis.speaker_labels[speaker] = "neutral"

            # Category scores
            analysis.professionalism = max(0.0, min(1.0, float(data.get("professionalism", 0.5))))
            analysis.helpfulness = max(0.0, min(1.0, float(data.get("helpfulness", 0.5))))
            analysis.engagement = max(0.0, min(1.0, float(data.get("engagement", 0.5))))
            analysis.resolution = max(0.0, min(1.0, float(data.get("resolution", 0.5))))

            # Highlights
            analysis.positive_highlights = data.get("positive_highlights", [])[:5]
            analysis.negative_highlights = data.get("negative_highlights", [])[:5]

            # Customer topics (normalize to lowercase, limit to 7)
            raw_topics = data.get("customer_topics", [])
            analysis.customer_topics = [t.lower().strip() for t in raw_topics if isinstance(t, str)][:7]

        except Exception as e:
            log.warning(f"Sentiment analysis failed: {e}")
            # Set default neutral values
            analysis.overall_score = 0.0
            analysis.overall_label = "neutral"

        return analysis

    def generate_analysis(
        self,
        turns: List[ConversationTurn],
        metrics: CallMetrics,
        raw_logs: str
    ) -> tuple[str, SentimentAnalysis]:
        """
        Use OpenAI to generate a comprehensive analysis of the call.

        Returns:
            Tuple of (analysis_report, sentiment_analysis)
        """
        transcript = self.build_conversation_transcript(turns)

        # Perform sentiment analysis
        sentiment = self.analyze_sentiment(turns)
        metrics.app_level.conversation_sentiment = sentiment.overall_score
        metrics.app_level.sentiment_label = sentiment.overall_label
        metrics.app_level.sentiment_analysis = sentiment

        # Calculate averages
        avg_llm_ttfb = sum(metrics.llm_ttfb_values) / len(metrics.llm_ttfb_values) if metrics.llm_ttfb_values else 0
        avg_tts_ttfb = sum(metrics.tts_ttfb_values) / len(metrics.tts_ttfb_values) if metrics.tts_ttfb_values else 0

        # Build metrics summary
        metrics_summary = f"""
## Call Metrics

- **Total Duration**: {metrics.total_duration_seconds:.1f} seconds ({metrics.total_duration_seconds/60:.1f} minutes)
- **Start Time**: {metrics.start_time.strftime('%Y-%m-%d %H:%M:%S') if metrics.start_time else 'N/A'}
- **End Time**: {metrics.end_time.strftime('%Y-%m-%d %H:%M:%S') if metrics.end_time else 'N/A'}

### Turn Statistics
- Total conversation turns (TTS segments): {metrics.total_turns}
- Interruptions detected: {metrics.interruptions}

### Performance Metrics
- Average LLM TTFB: {avg_llm_ttfb:.3f} seconds
- Average TTS TTFB: {avg_tts_ttfb:.3f} seconds

### Per-Bot Metrics
"""
        for bot_name, bot_data in metrics.bot_metrics.items():
            metrics_summary += f"""
**{bot_name}**:
- Speaking time: {bot_data['speaking_time']:.1f} seconds
- TTS segments: {bot_data['turn_count']}
- TTS characters: {bot_data['tts_characters']}
- LLM tokens: {bot_data['llm_tokens']['prompt']} prompt, {bot_data['llm_tokens']['completion']} completion
"""

        # Add sentiment analysis section
        metrics_summary += f"""
### Sentiment Analysis
- **Overall Score**: {sentiment.overall_score:.2f} ({sentiment.overall_label})
- **Professionalism**: {sentiment.professionalism:.0%}
- **Helpfulness**: {sentiment.helpfulness:.0%}
- **Engagement**: {sentiment.engagement:.0%}
- **Resolution**: {sentiment.resolution:.0%}

#### Per-Speaker Sentiment
"""
        for speaker, score in sentiment.speaker_sentiments.items():
            label = sentiment.speaker_labels.get(speaker, "neutral")
            metrics_summary += f"- **{speaker}**: {score:.2f} ({label})\n"

        if sentiment.positive_highlights:
            metrics_summary += "\n#### Positive Moments\n"
            for highlight in sentiment.positive_highlights[:3]:
                metrics_summary += f"- \"{highlight[:80]}\"\n"

        if sentiment.negative_highlights:
            metrics_summary += "\n#### Areas for Improvement\n"
            for highlight in sentiment.negative_highlights[:3]:
                metrics_summary += f"- \"{highlight[:80]}\"\n"

        if metrics.errors:
            metrics_summary += f"\n### Errors ({len(metrics.errors)})\n"
            for err in metrics.errors[:10]:
                metrics_summary += f"- {err[:200]}\n"

        # Truncate raw logs for context
        log_lines = raw_logs.split("\n")
        if len(log_lines) > 200:
            truncated_logs = "\n".join(log_lines[:100]) + "\n\n... [truncated] ...\n\n" + "\n".join(log_lines[-100:])
        else:
            truncated_logs = raw_logs

        prompt = f"""You are an AI conversation analyst. Analyze the following two-bot conversation session and provide a comprehensive review.

## Conversation Transcript
{transcript}

{metrics_summary}

## Sample Log Entries (for technical context)
```
{truncated_logs[:5000]}
```

Please provide a detailed analysis report covering:

1. **Conversation Summary**: What was the conversation about? What topics were discussed?

2. **Turn-Taking Quality**:
   - How well did the bots take turns?
   - Were there any overlaps or awkward patterns?
   - Did the conversation flow naturally?

3. **Context Handling**:
   - Did the bots maintain context throughout the conversation?
   - Were responses relevant to what was said?
   - Any cases of context loss or confusion?

4. **Technical Performance**:
   - How were the LLM and TTS latencies?
   - Any notable errors or warnings?
   - Were there any interruptions or speaking overlaps?

5. **Conversation Quality**:
   - Was the conversation natural and engaging?
   - Did the bots build on each other's points?
   - How was the pacing?

6. **Recommendations**:
   - What could be improved?
   - Any patterns that suggest issues?

Format your response as a well-structured markdown report.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing AI conversation logs and providing actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            analysis = response.choices[0].message.content
            return analysis, sentiment

        except Exception as e:
            log.error(f"OpenAI analysis failed: {e}")
            return f"Analysis generation failed: {e}\n\nRaw metrics:\n{metrics_summary}", sentiment

    def analyze_session(self, log_file_path: str) -> tuple[str, str, SentimentAnalysis, "CallMetrics"]:
        """
        Analyze a session from its log file.

        Returns:
            Tuple of (analysis_report, cli_summary, sentiment_analysis, call_metrics)
        """
        log.info(f"Analyzing session from: {log_file_path}")

        with open(log_file_path, "r") as f:
            raw_logs = f.read()

        turns, metrics = self.parse_logs(raw_logs)

        log.info(f"Parsed {len(turns)} conversation turns")
        log.info(f"Total duration: {metrics.total_duration_seconds:.1f} seconds")

        analysis, sentiment = self.generate_analysis(turns, metrics, raw_logs)

        # Build a quick metrics summary for CLI output
        cli_summary = f"""
{'='*60}
CALL ANALYSIS COMPLETE
{'='*60}

Duration: {metrics.total_duration_seconds:.1f}s ({metrics.total_duration_seconds/60:.1f} min)
TTS Segments: {metrics.total_turns}
Interruptions: {metrics.interruptions}
Errors: {len(metrics.errors)}

SENTIMENT ANALYSIS
------------------
Overall: {sentiment.overall_score:.2f} ({sentiment.overall_label})
Professionalism: {sentiment.professionalism:.0%}
Helpfulness: {sentiment.helpfulness:.0%}
Engagement: {sentiment.engagement:.0%}
Resolution: {sentiment.resolution:.0%}

Per-Speaker:
"""
        for speaker, score in sentiment.speaker_sentiments.items():
            label = sentiment.speaker_labels.get(speaker, "neutral")
            cli_summary += f"  {speaker}: {score:.2f} ({label})\n"

        cli_summary += "\nBot Statistics:\n"
        for bot_name, bot_data in metrics.bot_metrics.items():
            cli_summary += f"  {bot_name}: {bot_data['turn_count']} segments, {bot_data['speaking_time']:.1f}s speaking, {bot_data['tts_characters']} chars\n"

        return analysis, cli_summary, sentiment, metrics

    def save_report(self, analysis: str, output_path: str):
        """Save the analysis report to a file."""
        with open(output_path, "w") as f:
            f.write(analysis)
        log.info(f"Analysis report saved to: {output_path}")


# CONVENIENCE FUNCTIONS

def analyze_latest_session(
    logs_dir: str,
    openai_api_key: str,
    output_dir: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to analyze the current session.

    Args:
        logs_dir: Directory containing session logs
        openai_api_key: OpenAI API key
        output_dir: Optional directory to save the report

    Returns:
        Path to the saved report, or None if analysis failed
    """
    analyzer = CallAnalyzer(openai_api_key)

    log_file = analyzer.find_session_log(logs_dir)
    if not log_file:
        log.error("No session log found to analyze")
        return None

    analysis, cli_summary, sentiment, metrics = analyzer.analyze_session(str(log_file))

    # Update historical data store
    store = HistoricalDataStore(Path(logs_dir) / "call_history.json")
    store.add_call(metrics, sentiment)
    log.info("Updated historical call data")

    # Print summary to CLI
    print(cli_summary)
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60 + "\n")
    print(analysis)

    # Save report
    output_dir = output_dir or str(log_file.parent)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"analysis_{timestamp}.md"

    # Build full report with header
    full_report = f"""# Call Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Log File**: {log_file.name}

---

{analysis}
"""

    analyzer.save_report(full_report, str(report_path))

    return str(report_path)


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        exit(1)

    # Run analysis
    report_path = analyze_latest_session(str(script_dir), api_key)

    if report_path:
        print(f"\n Report saved to: {report_path}")
    else:
        print("\n Analysis failed")
