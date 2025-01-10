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
            "lat": "-27.501586",
            "lon": "-48.489710"
        })
    except Exception as e:
        await result_callback({"success": False, "error": str(e)})

tools = [
    {
        "function_declarations": [
            {
                "name": "get_my_current_location",
                "description": "Retrieves the user current location",
                "parameters": None,  # Specify None for no parameters
            },
        ],
        'google_search': {}
    }
]

system_instruction = """
You are a travel companion. Your responses will be converted to audio, so avoid using special characters or overly complex formating. 
Always use the google_search API to check which day is today. If there is a discrepancy always use the most recent date.

You can:
- Use get_my_current_location to retrieve my current location. When speaking to the user, inform them about the neighborhood and city they are in, rather than providing coordinates.
- Use google_search to check how is the weather.
- Use google_search to recommend restaurants.
- Use google_search to provide the most recent and relevant news from my current location.
- Answer any questions the user may have, ensuring your responses are accurate and concise.
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

        context = OpenAILLMContext(
            [{"role": "user", "content": "Start by greeting the user warmly, introducing yourself, and mentioning the current day. Be friendly and engaging to set a positive tone for the interaction."}],
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
