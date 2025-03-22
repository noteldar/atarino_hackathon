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
            chat_ctx.messages.append(
                llm.ChatMessage(role="user", content=image_content)
            )
            logger.debug("Added latest frame to conversation context")

    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.messages.append(
        llm.ChatMessage(
            content="""
	<GUIDELINES>
	- You are a joke generator. The user will share a news/story/event and you need to generate a joke about it and respond.
	- When you see an image in the conversation, naturally incorporate what you see
	- The text you generate will go straight to a text-to-speech model, generate only the spoken response.
    - Include in each response, a pause and a word with a custom pronunciation of a word from a list below.
	</GUIDELINES>

	<TTS_SETTINGS>
	The text-to-speech model you are using supports custom pauses, custom pronounciation, and spelling.
	You can use the following tags to control the text-to-speech model:

	<750> for a pause of 750 milliseconds

	spell() to spell a name 

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
