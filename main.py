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
    # model="gemini-2.0-flash-thinking-exp-01-21",
    model="gemini-2.0-flash"
)

rime_tts = TTS(
    model="mistv2",
    speaker="cove",
    speed_alpha=0.9,
    reduce_latency=True,
    pause_between_brackets=True,
    phonemize_between_brackets=True,
)


async def entrypoint(ctx: JobContext):
	logger.info("starting entrypoint")
	async def get_video_track(room: rtc.Room):
		"""Find and return the first available remote video track in the room."""
		for participant_id, participant in room.remote_participants.items():
			for track_id, track_publication in participant.track_publications.items():
				if track_publication.track and isinstance(
					track_publication.track, rtc.RemoteVideoTrack
				):
					logger.info(
						f"Found video track {track_publication.track.sid} "
						f"from participant {participant_id}"
					)
					return track_publication.track
		raise ValueError("No remote video track found in the room")


	async def get_latest_image(room: rtc.Room):
		"""Capture and return a single frame from the video track."""
		video_stream = None
		try:
			video_track = await get_video_track(room)
			video_stream = rtc.VideoStream(video_track)
			async for event in video_stream:
				logger.debug("Captured latest video frame")
				return event.frame
		except Exception as e:
			logger.error(f"Failed to get latest image: {e}")
			return None
		finally:
			if video_stream:
				await video_stream.aclose()


	async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
		"""
		Callback that runs right before the LLM generates a response.
		Captures the current video frame and adds it to the conversation context.
		"""
		latest_image = await get_latest_image(ctx.room)
		if latest_image:
			image_content = [llm.ChatImage(image=latest_image)]
			chat_ctx.messages.append(llm.ChatMessage(role="user", content=image_content))
			logger.debug("Added latest frame to conversation context")
	initial_chat_ctx = llm.ChatContext()
	initial_chat_ctx.messages.append(
		llm.ChatMessage(
			content="""
	<GUIDELINES>
	- You are a joke generator. The user will share a news/story/event and you need to generate a joke about it and respond.
	- When you see an image in the conversation, naturally incorporate what you see
	- The text you generate will go straight to a text-to-speech model, generate only the spoken response.
	</GUIDELINES>

	<TTS_SETTINGS>
	The text-to-speech model you are using supports custom pauses, custom pronounciation, and spelling.
	You can use the following tags to control the text-to-speech model:

	<750> for a pause of 750 milliseconds

	spell() to spell a name 

	CUSTOM PRONOUNCIATIONS:

	Vowels
	symbol	use in an english word
	@	bat
	a	hot
	A	butt
	W	about
	x	comma
	Y	bite
	E	bet
	R	bird, foreword
	e	bait
	I	bit
	i	beat
	o	boat
	O	boy
	U	book
	u	boot
	N	button


	Consonants
	symbol	use in an english word
	b	buy
	C	China
	d	die
	D	thy
	f	fight
	g	guy
	h	high
	J	jive
	k	kite
	l	lie
	m	my
	n	nigh
	G	sing
	p	pie
	r	rye
	s	sigh
	S	shy
	t	tie
	T	thigh
	v	vie
	w	wise
	y	yacht
	z	zoo
	Z	pleasure

	Stress
	For primary stress, put 1 before the relevant vowel. For example, comma would be {k1am0x}

	For seconadary stress, put 2 in front of the relevant vowel. For example, auctioneer would be {2akS0In1ir}

	All other vowels shoud have a 0 in front of them.


	</TTS_SETTINGS>
	<EXAMPLES>
	Customer: Hello, is this a hair salon?
	Assistant: Hello, hello. 
	</EXAMPLES>

			""",
			role="system",
		)
	)

	await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

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
		before_llm_cb=before_llm_cb,
	)

	logger.info(f"Agent connected to room: {ctx.room.name}")
	logger.info(f"Local participant identity: {ctx.room.local_participant.identity}")
	assistant.start(ctx.room)
	await assistant.say("What's up with you lately? Tell me a story")
	logger.info("starting agent")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
