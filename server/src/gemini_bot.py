#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Literal, TypedDict, Union

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Function handlers for the LLM
async def get_movies(function_name, tool_call_id, args, llm, context, result_callback):
    """Handler for fetching current movies."""
    logger.debug("Calling TMDB API: get_movies")
    try:
        await result_callback([
            {
                "id": 1,
                "title": "Inception",
                "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO."
            },
            {
                "id": 2,
                "title": "The Matrix",
                "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
            },
        ])
    except Exception as e:
        await result_callback({"success": False, "error": str(e)})

async def get_upcoming_movies(function_name, tool_call_id, args, llm, context, result_callback):
    """Handler for fetching upcoming movies."""
    logger.debug("Calling TMDB API: get_upcoming_movies")
    try:
        await result_callback([
            {
                "id": 3,
                "title": "The Shawshank Redemption",
                "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."
            },
        ])
    except Exception as e:
        await result_callback({"success": False, "error": str(e)})

tools = [
    {
        "function_declarations": [
            {
                "name": "get_current_movies",
                "description": "Show current movies in theaters",
                "parameters": None,  # Specify None for no parameters
            },
            {
                "name": "get_upcoming_movies",
                "description": "Show movies coming soon",
                "parameters": None,  # Specify None for no parameters,
            }
        ]
    }
]

system_instruction = """
You are a friendly movie expert. Your responses will be converted to audio, so avoid special characters. Always use the available functions to progress the conversation naturally.

Start by greeting the user. Help the user learn more about movies. You can:
- Use get_current_movies to see what's playing now
- Use get_upcoming_movies to see what's coming soon

After showing details or recommendations, ask if they'd like to explore another movie or end the conversation.
"""


async def main():
    """Main function to set up and run the movie explorer bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Movie Explorer Bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        # Initialize the Gemini Multimodal Live model
        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            system_instruction=system_instruction,
            tools=tools,
        )
        llm.register_function("get_current_movies", get_movies)
        llm.register_function("get_upcoming_movies", get_upcoming_movies)

        context = OpenAILLMContext(
            [{"role": "user", "content": "Say hello."}],
        )
        context_aggregator = llm.create_context_aggregator(context)

        # TODO: add the RTVI events for Pipecat client UI

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                context_aggregator.user(),  # User responses
                llm,  # LLM
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
