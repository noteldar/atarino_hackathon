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
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

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
            chat_ctx.messages.append(
                llm.ChatMessage(role="user", content=image_content)
            )
            logger.debug("Added latest frame to conversation context")

        latest_user_message = [m for m in chat_ctx.messages if m.role == "user"][-1]

        # 1. Association Agent - Generate joke suggestions based on user query and image
        association_prompt = f"""
        Based on the user's message and the image provided, generate a list of 20 joke suggestions.
        Be creative and diverse in your suggestions. Each joke should be brief and humorous.
        Format your response as a numbered list.
        
        User message: {latest_user_message.content if hasattr(latest_user_message.content, '__str__') else 'No text content'}
        """

        association_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=association_prompt,
        )

        joke_suggestions = association_response.text
        logger.debug(f"Generated joke suggestions: {joke_suggestions}")

        # 2. Humor Judge - Select the funniest joke from the suggestions
        humor_judge_prompt = f"""
        You are a humor judge. Below is a list of joke suggestions.
        Select the single funniest joke from the list and explain briefly why it's the funniest.
        Return only the selected joke without numbering or explanation.
        
        Joke suggestions:
        {joke_suggestions}
        """

        humor_judge_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=humor_judge_prompt,
        )

        selected_joke = humor_judge_response.text.strip()
        logger.debug(f"Selected joke: {selected_joke}")

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
    Your task is to use this joke and convert it into a text for a text-to-speech model.

	<TTS_SETTINGS>
	The text-to-speech model you are using supports custom pauses, custom pronounciation, and spelling.
	You can use the following tags to control the text-to-speech model:

	<n> for a pause of n milliseconds

	Custom pronounciation - here's a list of words and their custom pronounciation:
1. **Pizza**: {p1i0zx} (sounds like "peeza")
2. **Banana**: {bx0n1a0nx} (emphasis on the second syllable)
3. **Robot**: {r1o0b0At} (with "but" ending)
4. **Computer**: {kxm0p1u0tx} (compressed pronunciation)
5. **Potato**: {px0t1e0to} (with long A sound)
6. **Sandwich**: {s1@0nW0wiC} (with unusual vowel shifts)
7. **Balloon**: {bx0l1u0n} (exaggerated "oo" sound)
8. **Dinosaur**: {d1Y0nx0sOr} (with "sore" ending)
9. **Elephant**: {1i0lW0fxnt} (first syllable emphasis)
10. **Telephone**: {t1E0lW0fon} (altered vowels)
11. **Internet**: {1i0ntR0nEt} (strange first syllable)
12. **Cucumber**: {k1u0kAm0bR} (exaggerated first syllable)
13. **Spaghetti**: {spx0g1E0ti} (emphasis on the E)
14. **Bicycle**: {b1Y0sx0kWl} (altered vowels)
15. **Chocolate**: {C1o0kx0lxt} (altered first syllable)
16. **Hamburger**: {h1@0mbR0gR} (compressed with bird-like endings)
17. **Octopus**: {1A0ktW0pUs} (altered first vowel)
18. **Microwave**: {m1Y0krW0wev} (exaggerated first syllable)
19. **Pineapple**: {p1Y0n@0pl} (with shorter ending)
20. **Kangaroo**: {k@0Gx0r1u} (emphasis on final syllable)
21. **Crocodile**: {kr1A0kW0dYl} (with unusual first syllable)
22. **Butterfly**: {b1O0tR0flY} (with "boy" sound at start)
23. **Helicopter**: {h1E0lW0kA0ptR} (compressed)
24. **Umbrella**: {Am0br1E0lx} (emphasis on second syllable)
25. **Avocado**: {1@0vW0kY0do} (altered first syllable)
26. **Hippopotamus**: {h2i0pW0p1A0tx0mWs} (emphasis shift)
27. **Watermelon**: {w1O0tR0mE0lxn} (with "boy" sound at start)
28. **Refrigerator**: {rW0fr2i0jR1e0tR} (emphasis on third syllable)
29. **Calculator**: {k1@0lkyW0le0tR} (compressed)
30. **Caterpillar**: {k1@0tR0p2i0lR} (shortened)
31. **Television**: {t1E0lW0v2i0ZWn} (altered vowels)
32. **Alligator**: {2@0lW0g1e0tR} (emphasis shift)
33. **Restaurant**: {r1E0stW0rAnt} (altered vowels)
34. **Gymnasium**: {j2i0mn1e0zWm} (compressed)
35. **Tornado**: {tOr0n1e0do} (emphasis shift)
36. **Mosquito**: {m1A0sk0wi0to} (altered first syllable)
37. **Pajamas**: {px0j1a0mxz} (emphasis on middle syllable)
38. **Penguin**: {p1E0Gw0In} (altered vowels)
39. **Skeleton**: {sk1i0lW0tAn} (with long E sound)
40. **Astronaut**: {1@0strW0nOt} (with "boy" ending)
41. **Saxophone**: {s1@0ksW0fon} (altered vowels)
42. **Helicopter**: {h1E0lY0kA0ptR} (with "bite" sound)
43. **Cinnamon**: {s2i0nW0m1A0n} (altered stress and vowels)
44. **Platypus**: {pl1@0tW0pUs} (altered vowels)
45. **Rattlesnake**: {r1@0tWl0snek} (compressed)
46. **Broccoli**: {br1A0kW0li} (with short O sound)
47. **Cauliflower**: {k1O0lW0flY0R} (with "boy" first syllable)
48. **Jellyfish**: {j1E0lW0fIS} (compressed)
49. **Flamingo**: {flW0m1i0Go} (emphasis on second syllable)
50. **Hippopotamus**: {h2i0pW0b1A0tW0mWs} (altered consonant in middle)
	</TTS_SETTINGS>
	<EXAMPLES>
    User message: Mars landing is expected in 2035
	Joke suggestion: It looks like i will be waiting for a long time for the Tesla stock to climb up again
	TTS input:  It looks like i will be waiting for a {loooooooong} time for the Tesla stock to climb up again
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
