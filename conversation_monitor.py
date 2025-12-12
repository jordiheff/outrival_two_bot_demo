#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""
Conversation Monitor - A third LLM that listens to the conversation and decides when to end it.

This processor collects TranscriptionFrames from both bots and periodically
evaluates whether the conversation has reached a natural conclusion point.
When it determines the conversation should end, it signals the customer bot (Sarah)
to wrap up the call.
"""

import asyncio
from typing import Awaitable, Callable, List, Optional

from loguru import logger
from openai import AsyncOpenAI

from pipecat.frames.frames import Frame, TranscriptionFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# SYSTEM PROMPT FOR CONVERSATION ANALYSIS

MONITOR_SYSTEM_PROMPT = """You are a conversation analyst monitoring a call between \
a customer (Sarah) and a travel agent (Marcus).

Your job is to determine if the conversation has reached a natural conclusion point \
where Sarah should politely wrap up the call.

The conversation should end when:
1. Sarah has received answers to all her main questions (trip dates, hotel name, price, trip duration)
2. The conversation has reached a natural pause or winding-down phase
3. Both parties seem satisfied with the information exchanged
4. There's repetition or the conversation is going in circles
5. More than 15 prompts have occurred

The conversation should NOT end if:
1. Sarah still has unanswered questions
2. Marcus is in the middle of providing important information
3. There are unresolved topics being actively discussed

Respond with ONLY one word: "END" if the conversation should end, or "CONTINUE" if it should continue."""


# CONVERSATION MONITOR

class ConversationMonitor(FrameProcessor):
    """
    Monitors the conversation between two bots using TranscriptionFrames.

    Uses a third LLM to analyze the conversation and determine when it's
    appropriate to end the call. When ready to end, it triggers a callback
    to signal the customer bot.
    """

    def __init__(
        self,
        openai_api_key: str,
        on_should_end_call: Optional[Callable[[], Awaitable[None]]] = None,
        check_interval_seconds: float = 5.0,
        min_exchanges_before_check: int = 4,
        model: str = "gpt-4o-mini",
        **kwargs
    ):
        """
        Initialize the conversation monitor.

        Args:
            openai_api_key: OpenAI API key for the monitoring LLM
            on_should_end_call: Async callback to invoke when conversation should end
            check_interval_seconds: How often to check if conversation should end
            min_exchanges_before_check: Minimum conversation turns before checking
            model: OpenAI model to use for analysis
        """
        super().__init__(**kwargs)

        self._client = AsyncOpenAI(api_key=openai_api_key)
        self._model = model
        self._on_should_end_call = on_should_end_call
        self._check_interval = check_interval_seconds
        self._min_exchanges = min_exchanges_before_check

        # Conversation history
        self._transcript: List[dict] = []
        self._last_check_index = 0
        self._should_end = False
        self._check_task: Optional[asyncio.Task] = None
        self._running = False

        self._logger = logger.bind(bot_name="Monitor")

    async def start(self):
        """Start the periodic checking loop."""
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        self._logger.info("Conversation monitor started")

    async def stop(self):
        """Stop the periodic checking loop."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Conversation monitor stopped")

    async def _check_loop(self):
        """Periodically check if the conversation should end."""
        while self._running:
            await asyncio.sleep(self._check_interval)

            # Only check if we have enough exchanges and haven't already decided to end
            if len(self._transcript) >= self._min_exchanges and not self._should_end:
                # Only analyze new content since last check
                if len(self._transcript) > self._last_check_index:
                    should_end = await self._analyze_conversation()
                    self._last_check_index = len(self._transcript)

                    if should_end:
                        self._should_end = True
                        self._logger.info("Monitor: Conversation should end - triggering callback")
                        if self._on_should_end_call:
                            await self._on_should_end_call()

    async def _analyze_conversation(self) -> bool:
        """
        Use LLM to analyze if the conversation has reached a natural end point.

        Returns:
            True if the conversation should end, False otherwise
        """
        transcript_text = "\n".join([
            f"{turn['speaker']}: {turn['text']}"
            for turn in self._transcript
        ])

        user_prompt = f"Here is the conversation so far:\n\n{transcript_text}\n\nShould Sarah wrap up the call now?"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": MONITOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )

            answer = response.choices[0].message.content.strip().upper()
            self._logger.debug(f"Monitor analysis result: {answer}")

            return answer == "END"

        except Exception as e:
            self._logger.error(f"Monitor analysis failed: {e}")
            return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process frames to collect transcription data.

        Captures TranscriptionFrame (from STT) and TTSTextFrame (bot responses)
        to build a complete picture of the conversation.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # This is speech-to-text from either bot
            self._transcript.append({
                "speaker": frame.user_id if frame.user_id else "Unknown",
                "text": frame.text,
                "type": "transcription"
            })
            self._logger.trace(f"Captured transcription: {frame.user_id}: {frame.text}")

        elif isinstance(frame, TTSTextFrame):
            # This is text being sent to TTS (what the bot is about to say)
            # Note: We track this as a backup, but TranscriptionFrames are primary
            pass

        # Always pass the frame through
        await self.push_frame(frame, direction)

    def get_transcript(self) -> List[dict]:
        """Return the collected transcript."""
        return self._transcript.copy()
