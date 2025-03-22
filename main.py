from __future__ import annotations
from dotenv import load_dotenv
import os

load_dotenv()
import logging

from livekit import api
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)

from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, google
from livekit.plugins.rime import TTS


logger = logging.getLogger("myagent")
logger.setLevel(logging.INFO)

google_llm = google.LLM(
    model="gemini-2.0-flash-exp",
    temperature="0.8",
)

rime_tts = TTS(
    model="mist",
    speaker="rainforest",
    speed_alpha=0.9,
    reduce_latency=True,
)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")
    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.messages.append(
        llm.ChatMessage(
            content="""
TALK ABOUT KITTENS OBSESSIVELY
            """,
            role="system",
        )
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        # flexibility to use any models
        stt=deepgram.STT(model="nova-2-general"),
        llm=google_llm,
        tts=rime_tts,
        # intial ChatContext with system prompt
        chat_ctx=initial_chat_ctx,
        # whether the agent can be interrupted
        allow_interruptions=True,
        # sensitivity of when to interrupt
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        # minimal silence duration to consider end of turn
        min_endpointing_delay=0.5,
    )

    logger.info(f"Agent connected to room: {ctx.room.name}")
    logger.info(f"Local participant identity: {ctx.room.local_participant.identity}")
    assistant.start(ctx.room)
    logger.info("starting agent")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
