\"\"\"Gemini Live session manager — handles audio streaming, function calls, search.\"\"\"

import asyncio
import json
import os
import time
from typing import AsyncGenerator, Callable, Awaitable

from google import genai
from google.genai import types

from ai_common import (  # noqa: F401 - re-exported for callers and tests
    ASSISTANT_GENDER,
    ASSISTANT_LANGUAGE,
    ASSISTANT_NAME,
    ASSISTANT_RESPONSE_LANGUAGE,
    ASSISTANT_SPEAKING_STYLE,
    DEBUG_LOGGING,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    FOLLOW_UP_RESOLUTION_PROMPT,
    INPUT_LANGUAGE_LOCK_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    build_persona_prompt,
    build_prompt,
    build_tool_specs,
    debug_log,
    web_search,
)

def _cfg(name: str, default: str) -> str:
    \"\"\"bashio hands unset add-on options through as the literal string "null".\"\"\"
    value = os.getenv(name, "").strip()
    return default if not value or value.lower() == "null" else value


GEMINI_MODEL = _cfg("GEMINI_MODEL", "gemini-3.8-live")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Charon")
RECEIVE_IDLE_TIMEOUT_AFTER_FUNCTION = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AFTER_FUNCTION", "1.5"))
RECEIVE_IDLE_TIMEOUT_AFTER_AUDIO = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AFTER_AUDIO", "1.2"))
RECEIVE_IDLE_TIMEOUT_GENERAL = float(os.getenv("RECEIVE_IDLE_TIMEOUT_GENERAL", "8.0"))
# Tools are async: the result is spoken in a fresh turn ~0.5-1s after we send it.
RECEIVE_IDLE_TIMEOUT_AWAITING_REPLY = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AWAITING_REPLY", "5.0"))
GEMINI_SILENCE_DURATION_MS = int(os.getenv("GEMINI_SILENCE_DURATION_MS", "3000"))
# Quiet confirmations: a successful plain device action is
# acknowledged with a short chime instead of a spoken sentence.
QUIET_CONFIRMATIONS = os.getenv("QUIET_CONFIRMATIONS", "false").lower() in ("1", "true", "yes", "on")
QUIET_TOOLS = frozenset({
    "control_device", "control_room", "activate_scene", "run_script",
    "set_climate", "control_vacuum", "cancel_timer", "stop_timer_alarm",
})

# Gemini 3.8 Live runs tools NON_BLOCKING: the model is free to speak while a call
# runs, so it must be told not to claim success before the result, and to cover
# slow lookups with a filler.
ASYNC_TOOLS_PROMPT = (
    "\n\nNarzędzia wykonują się w tle. Po wywołaniu narzędzia sterującego NIE potwierdzaj "
    "wykonania, zanim dostaniesz wynik — potwierdź dopiero na podstawie wyniku. "
    "Wyjątek: zanim wywołasz search_web albo inne narzędzie pobierające informacje z internetu "
    "lub zewnętrznego serwisu, powiedz najpierw jedno bardzo krótkie zdanie, np. „Już sprawdzam.”, "
    "a dopiero potem wywołaj narzędzie. Przy sterowaniu urządzeniami, scenami i timerami "
    "nic nie mów przed wywołaniem."
)


def make_confirm_chime(sample_rate: int = 24000) -> bytes:
    \"\"\"Short soft two-note rising chime, PCM16 mono at Gemini's output rate.\"\"\"
    import numpy as np
    notes = [(0.07, 880.0), (0.11, 1318.5)]
    parts = []
    for duration, freq in notes:
        t = np.arange(int(sample_rate * duration)) / sample_rate
        envelope = np.minimum(1.0, t / 0.008) * np.exp(-t * 18)
        parts.append(np.sin(2 * np.pi * freq * t) * envelope)
    wave = np.concatenate(parts + [np.zeros(int(sample_rate * 0.05))])
    return (wave * 7000).astype(np.int16).tobytes()


CONFIRM_CHIME = make_confirm_chime()


def is_quiet_success(name: str, result) -> bool:
    \"\"\"A plain device action that clearly worked — nothing worth saying aloud.\"\"\"
    if name not in QUIET_TOOLS or not isinstance(result, dict):
        return False
    if result.get("status") != "ok" or result.get("no_change"):
        return False
    # e.g. "cancel the timer" when none was running: let the model explain.
    return result.get("count", 1) != 0


def build_tools(room_keys: list[str], vacuum_enabled: bool = False) -> list:
    \"\"\"Adapt the shared tool catalogue to Gemini's FunctionDeclaration type.\"\"\"
    declarations = [
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters=spec["parameters"],
            behavior=types.Behavior.NON_BLOCKING,
        )
        for spec in build_tool_specs(room_keys, vacuum_enabled)
    ]
    return [types.Tool(function_declarations=declarations)]


class GeminiSession:
    \"\"\"Manages a Gemini Live session with streaming audio.\"\"\"

    def __init__(self, client: genai.Client, entity_list: str, room_lights: dict,
                 ha_context: str, history: list,
                 on_function_call: Callable,
                 voice: str | None = None,
                 on_responding: Callable | None = None,
                 vacuum_enabled: bool = False,
                 local_area_id: str = ""):
        self.model = GEMINI_MODEL
        self.client = client
        self.entity_list = entity_list
        self.room_lights = room_lights
        self.ha_context = ha_context
        self.history = history
        self.on_function_call = on_function_call
        self.voice = voice or GEMINI_VOICE
        self.on_responding = on_responding
        self.vacuum_enabled = vacuum_enabled
        self.local_area_id = local_area_id

    def _build_prompt(self) -> str:
        prompt = build_prompt(self.entity_list, self.ha_context, self.history,
                              self.local_area_id)
        return prompt + ASYNC_TOOLS_PROMPT

    async def stream_audio(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        on_audio_out: Callable[[bytes], Awaitable[None]],
    ) -> str:
        \"\"\"Stream audio to Gemini, stream response audio back via callback.

        Returns summary of what happened (for history).
        \"\"\"
        room_keys = list(self.room_lights.keys())
        prompt = self._build_prompt()

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice),
                ),
                language_code=ASSISTANT_LANGUAGE,
            ),
            system_instruction=types.Content(parts=[types.Part(text=prompt)]),
            tools=build_tools(room_keys, self.vacuum_enabled),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
        )

        function_calls_made = []
        response_text = ""
        last_receive_time = time.monotonic()
        active_tool_tasks = set()

        debug_log(f"  [gemini] model={self.model}")
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            input_transcript_parts = []
            output_transcript_parts = []
            send_done = False

            async def execute_tool_background(name: str, args: dict, call_id: str):
                try:
                    if self.on_responding:
                        self.on_responding(True)
                    
                    result = await self.on_function_call(name, args)
                    debug_log(f"  [gemini] Wynik funkcji {name}: {result}")

                    await session.send_tool_response(
                        function_responses=[
                            types.FunctionResponse(
                                name=name,
                                id=call_id,
                                response={"result": result}
                            )
                        ]
                    )

                    if QUIET_CONFIRMATIONS and is_quiet_success(name, result):
                        debug_log("  [gemini] Akcja udana (Ciche potwierdzenie) - odtwarzam dźwięk.")
                        await on_audio_out(CONFIRM_CHIME)

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    debug_log(f"  [gemini] Wyjątek w zadaniu narzędzia {name}: {e}")

            async def send_audio():
                nonlocal send_done
                chunk_n = 0
                try:
                    async for chunk in audio_chunks:
                        chunk_n += 1
                        if chunk_n == 1:
                            debug_log("  [gemini] Rozpoczęto aktywność użytkownika. Przesyłam audio...")
                            await session.send_realtime_input(activity_start=types.ActivityStart())
                        
                        await session.send_realtime_input(
                            media_chunks=[types.Blob(data=chunk, mime_type="audio/pcm")]
                        )
                    
                    if chunk_n > 0:
                        debug_log("  [gemini] Koniec audio użytkownika. Zamykam aktywność (ActivityEnd)...")
                        await session.send_realtime_input(activity_end=types.ActivityEnd())
                except Exception as e:
                    debug_log(f"  [gemini] Błąd podczas wysyłania audio: {e}")
                finally:
                    send_done = True

            async def receive_responses():
                nonlocal response_text, last_receive_time
                try:
                    async for response in session.receive():
                        last_receive_time = time.monotonic()
                        
                        if response.input_transcription:
                            for part in response.input_transcription.parts:
                                if part.text:
                                    input_transcript_parts.append(part.text)

                        if response.tool_call and response.tool_call.function_calls:
                            for call in response.tool_call.function_calls:
                                name = call.name
                                args = call.args
                                call_id = call.id
                                debug_log(f"  [gemini] Model żąda wywołania funkcji: {name}({args})")
                                function_calls_made.append(f"{name}({json.dumps(args)})")

                                if self.on_function_call:
                                    task = asyncio.create_task(execute_tool_background(name, args, call_id))
                                    active_tool_tasks.add(task)
                                    task.add_done_callback(active_tool_tasks.discard)

                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    if self.on_responding:
                                        self.on_responding(True)
                                    await on_audio_out(part.inline_data.data)
                                
                                if part.text:
                                    response_text += part.text
                                    output_transcript_parts.append(part.text)

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    debug_log(f"  [gemini] Błąd w pętli odbierania: {e}")

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_responses())

            try:
                while not send_done or active_tool_tasks or (time.monotonic() - last_receive_time < RECEIVE_IDLE_TIMEOUT_GENERAL):
                    await asyncio.sleep(0.1)
                    
                    if active_tool_tasks and not response_text:
                        if time.monotonic() - last_receive_time > RECEIVE_IDLE_TIMEOUT_AWAITING_REPLY:
                            break
            finally:
                send_task.cancel()
                receive_task.cancel()
                
                if active_tool_tasks:
                    for task in active_tool_tasks:
                        task.cancel()
                    await asyncio.gather(*active_tool_tasks, return_exceptions=True)
                    
                await asyncio.gather(send_task, receive_task, return_exceptions=True)

        user_said = "".join(input_transcript_parts).strip() or "[Audio Input]"
        model_said = response_text.strip() or "[Audio/Action Response]"
        tools_str = f" | Wywołane narzędzia: {', '.join(function_calls_made)}" if function_calls_made else ""
        
        return f"Użytkownik: {user_said} -> Asystent: {model_said}{tools_str}"
