from __future__ import annotations
from dotenv import load_dotenv
import os

load_dotenv()
import logging
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions

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
from google import genai
from google.genai import types

from typing import Annotated, Union
from exa_py import Exa


gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
logger = logging.getLogger("myagent")
logger.setLevel(logging.INFO)

google_llm = google.LLM(
    # model="gemini-2.0-flash-thinking-exp-01-21",
    model="gemini-2.0-flash",
    tool_choice="required",
)

rime_tts = TTS(
    model="mistv2",
    speaker="cove",
    speed_alpha=0.9,
    reduce_latency=True,
    pause_between_brackets=True,
    phonemize_between_brackets=True,
)

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


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
                logger.info(f"Captured latest video frame: {type(event.frame.data)}")
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
        logger.info("before_llm_cb")
        latest_image = await get_latest_image(ctx.room)

        latest_user_message = [m for m in chat_ctx.messages if m.role == "user"][-1]
        logger.info(f"latest user message: {latest_user_message.content}")
        if latest_image:
            image_content = [llm.ChatImage(image=latest_image)]
            chat_ctx.messages.append(
                llm.ChatMessage(role="user", content=image_content)
            )
            logger.debug("Added latest frame to conversation context")

        # 1. Association Agent - Generate joke suggestions based on user query and image
        association_prompt = f"""
		Based on the user's message and the image provided, generate a list of 20 joke suggestions.
		Be creative and diverse in your suggestions. Each joke should be brief and humorous.
		Format your response as a numbered list.
		```
		User message: {latest_user_message.content if hasattr(latest_user_message.content, '__str__') else 'No text content'}
		```
		"""
        logger.info(f"association_prompt: {association_prompt}")
        image_bytes = encode(
            latest_image,
            EncodeOptions(
                format="JPEG",
                resize_options=ResizeOptions(
                    width=512, height=512, strategy="scale_aspect_fit"
                ),
            ),
        )
        association_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                association_prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
            ],
        )

        joke_suggestions = association_response.text
        logger.info(f"Generated joke suggestions: {joke_suggestions}")

        # 2. Humor Judge - Select the funniest joke from the suggestions
        humor_judge_prompt = f"""
		You are a humor judge. Below is a list of joke suggestions.
		Select the single funniest joke from the list and explain briefly why it's the funniest.
		Use the following framework to judge:
		The best jokes are following one of these styles:
		- Sarcasm
		- Absurd
		- Dry
		- Exaggeration
		- Popular reference
		- Dark
		- Wordplay

		Use the examples of good jokes below to help you judge the jokes.
		EXAMPLES:
		EXAMPLE1:
		INPUT:
		```
		Screenshot with a bunch of factors for energy consumption of different programming languages and the headline "Python consumes 76 times more energy and is 72 times slower than C."
		```
		JOKE STYLE: self-deprecation
		JOKE:
		```
		yes but what about my energy and desire to live when i code in C
		```
		OR 
		JOKE STYLE: exaggeration
		JOKE:
		```
		and when you do it in assembly you actually create a perpetuum mobile and generate energy out of nothing
		```

		EXAMPLE2:
		INPUT:
		```
		Screenshot of a tweet that says "Simone Biles is better at gymnastics than I am at sitting at home drinking watching TV. And I'm pretty good at that"
		```
		JOKE STYLE: sarcasm
		JOKE:
		```
		She's 27 she's probably better at drinking than you are mate.. and at social media since we're at it  
		```


		EXAMPLE4:
		INPUT:
		```
		wendys
		4h4 hours ago
		My plan for today was to get wendys breakfast and thats pretty much it
		```
		JOKE STYLE: Dark
		JOKE:
		```
		Said Melanie Vicks, 32, who died tragically later that day from stomach bleeding
		```

		EXAMPLE5:
		INPUT:
		```
		rombesk
		2h2 hours ago
		What company could go bankrupt tomorrow and it would be good for the world?
		```
		JOKE STYLE: Dark
		JOKE:
		```
		Make a wish foundation
		```

		EXAMPLE6:
		INPUT:
		```
		ginnyhogan_
		3h3 hours ago
		Id like to marry rich, but honestly, I'd also settle for marrying poor.
		```
		JOKE STYLE: self-depration
		JOKE:
		```
		I am so lonely, i'm at the stage where i'd settle for getting a physical touch by a homeless person
		```
		
		Joke suggestions:
		{joke_suggestions}
		"""

        humor_judge_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=humor_judge_prompt,
        )

        selected_joke = humor_judge_response.text.strip()
        logger.info(f"Selected joke: {selected_joke}")

        # Add the selected joke as a system message to guide the main LLM response
        chat_ctx.messages.append(
            llm.ChatMessage(
                role="user",
                content=f"Use this joke as inspiration for your response: {selected_joke}",
            )
        )

    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.messages.append(
        llm.ChatMessage(
            content="""
	You are a conversational assistant.
	You are given the latest user message along with a suggestion for a joke.
	You also have the image from the user's camera.
	Your task is to use this joke suggestion and the image when appropriate and convert it into a joke text for a text-to-speech model.

	<TTS_SETTINGS>
	The text-to-speech model you are using supports custom pauses, custom pronounciation, and spelling.
	You can use the following tags to control the text-to-speech model:

	<n> for a pause of n milliseconds

	Custom pronounciations:
	A custom pronounciation of any word can be generated.
	Here is the table for vowels and consonants:
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

	To accentuate a certain vowel within a word, use stress numeric tags:

	For primary stress, put 1 before the relevant vowel. For example, comma would be {k1am0x}

	For seconadary stress, put 2 in front of the relevant vowel. For example, auctioneer would be {2akS0In1ir}

	All other vowels shoud have a 0 in front of them.
	</TTS_SETTINGS>
	<EXAMPLES>
	One trick is to prolong a vowel by using the stress tags.

	User message: Mars landing is expected in 2035
	Joke suggestion: I'll be waiting so freaking long for the Tesla stock to climb up again
	TTS input:     I'll be waiting {s0o2o1o0o0o0o0o} freaking long for the Tesla stock to climb up again

	Another trick is to intentially change the pronounciation of a word
	User message: I had some wine with friends last night
	Joke suggestion: Were the friends you had 3 bottles of wine with imaginary? 
	TTS input: Were the friends you had 3 bottles of {v1In2o} with imaginary? 

	One more trick is to imitate a stutter to mimick the user's word
	User message: I decided to take on pilates
	Joke suggestion: Oh you think pilates is going to help with your body shape? I doubt it
	TTS input: Oh you think {p2Il1A1@1at0is} is going to help with your body shape? I doubt it

	</EXAMPLES>

	RETURN THE TTS INPUT ONLY NOTHING ELSE!
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
        before_llm_cb=before_llm_cb,
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
    await assistant.say("What's up with you lately? Tell me a story")
    logger.info("starting agent")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
