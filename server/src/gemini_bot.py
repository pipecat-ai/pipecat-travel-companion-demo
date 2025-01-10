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
# TODO: should refactor this function to retrieve from RTVI
async def get_my_current_location(function_name, tool_call_id, args, llm, context, result_callback):
    logger.debug("Calling get_my_current_location")
    try:
        await result_callback({
            "lat": "-27.501604",
            "lon": "-48.489933"
        })
    except Exception as e:
        await result_callback({"success": False, "error": str(e)})

async def set_restaurant_location(function_name, tool_call_id, arguments, llm, context, result_callback):
    lat = arguments["lat"]
    lon = arguments["lon"]
    logger.debug(f"Calling set_restaurant_location with arguments {lat},{lon}")
    await result_callback({})

tools = [
    {
        "function_declarations": [
            {
                "name": "get_my_current_location",
                "description": "Retrieves the user current location",
                "parameters": None,  # Specify None for no parameters
            },
            {
                "name": "set_restaurant_location",
                "description": "Sets the location of the chosen restaurant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "string",
                            "description": "Latitude of the location",
                        },
                        "lon": {
                            "type": "string",
                            "description": "Longitude of the location",
                        },
                    },
                    "required": ["lat", "lon"],
                },
            },
        ],
        'google_search': {}
    }
]

system_instruction = """
You are a travel companion, and your responses will be converted to audio, so keep them simple and avoid special characters or complex formatting.

You can:
- Use get_my_current_location to determine the user's current location. Once retrieved, inform the user of the city they are in, rather than providing coordinates.
- Use google_search to check the weather and share it with the user. Describe the temperature in Celsius and Fahrenheit.
- Use google_search to recommend restaurants that are nearby to the user's location, less than 10km. 
- Use set_restaurant_location to share the location of a selected restaurant with the user.
- Use google_search to provide recent and relevant news from the user's current location.

Answer any user questions with accurate, concise, and conversational responses.
"""


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Latest news!",
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
        llm.register_function("get_my_current_location", get_my_current_location)
        llm.register_function("set_restaurant_location", set_restaurant_location)

        context = OpenAILLMContext(
            [{"role": "user", "content": """
            Start with a warm greeting and a brief introduction of yourself. 
            Use the google_search tool to retrieve the current date. If there's any discrepancy, prioritize the most recent date. 
            Ensure you wait for the updated date before sharing it with the user in a friendly and conversational tone.
            """}],
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
