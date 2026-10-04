#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini Live Session Manager

* Strumieniowo przesyła PCM (16‑bit, 16 kHz, mono) do modelu
  `gemini‑3.8‑live`.
* Obsługuje **non‑blocking tools** – wywołania funkcji są wykonywane w tle,
  a model kontynuuje wypowiedź.
* Po każdej zakończonej turze nie zamyka połączenia, lecz czeka na kolejne
  wypowiedzi użytkownika.  Gdy nie wykryje żadnej aktywności przez
  **15 s** (wartość konfigurowalna) sesja jest zamykana.
* Zwracany wynik jest typem `GeminiTurnResult` zawierającym:
  - tekst wypowiedzi modelu,
  - listę wywołanych funkcji (czytelny łańcuch),
  - surowe kawałki audio do odtworzenia po stronie klienta.

Wymagane zależności (pip):
    google‑generativeai, structlog, numpy, aiohttp, python‑dotenv
oraz dostęp do klucza API Gemini (`GEMINI_API_KEY`) i ewentualnie
klucza SerpAPI (`SERPAPI_KEY`) używanego w `ai_common.web_search`.

"""

# ----------------------------------------------------------------------
#  IMPORTY
# ----------------------------------------------------------------------
from __future__ import annotations

import asyncio
import json
import os
import time
import structlog
from dataclasses import dataclass, field
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Tuple,
)

# Google Gemini SDK
from google.generativeai import genai, types

# Wspólne helpery z Twojego projektu (muszą znajdować się w PYTHONPATH)
from ai_common import (
    ASSISTANT_LANGUAGE,
    QUIET_CONFIRMATIONS,
    build_tool_specs,
    debug_log,
    web_search,
)

# ----------------------------------------------------------------------
#  KONFIGURACJA I STALE
# ----------------------------------------------------------------------
log = structlog.get_logger(__name__)


def _env(name: str, default: str) -> str:
    """
    Pobiera wartość ze środowiska. Jeśli zmienna jest pusta,
    równa „null” (w sensie BashIO) lub nie istnieje – zwraca ``default``.
    """
    val = (os.getenv(name) or "").strip()
    return default if not val or val.lower() == "null" else val


# Model i głos – można nadpisać zmiennymi środowiskowymi
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.8-live")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Charon")

# PCM‑rate – musi zgadzać się z tym, co koduje front‑end.
PCM_RATE = 16000  # Hz

# --------------------------- timeouty (sekundy) -----------------------
class IdleTimeouts:
    """Timeouty wykorzystywane w pętli odbioru."""
    AFTER_FUNCTION = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AFTER_FUNCTION", "1.5"))
    AFTER_AUDIO = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AFTER_AUDIO", "1.2"))
    GENERAL = float(os.getenv("RECEIVE_IDLE_TIMEOUT_GENERAL", "8.0"))
    AWAITING_REPLY = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AWAITING_REPLY", "5.0"))


# Cisza, po której sesja zostaje zamknięta (wartość w milisekundach → sekundy)
SILENCE_TIMEOUT_SEC = float(os.getenv("GEMINI_SILENCE_DURATION_MS", "15000")) / 1000.0


# --------------------------- dźwięk potwierdzenia --------------------
def make_confirm_chime(sample_rate: int = 24000) -> bytes:
    """Krótki, dwutonowy dźwięk potwierdzający (PCM16 mono)."""
    import numpy as np

    notes = [(0.07, 880.0), (0.11, 1318.5)]
    parts = []
    for dur, freq in notes:
        t = np.arange(int(sample_rate * dur)) / sample_rate
        envelope = np.minimum(1.0, t / 0.008) * np.exp(-t * 18)
        parts.append(np.sin(2 * np.pi * freq * t) * envelope)
    wave = np.concatenate(parts + [np.zeros(int(sample_rate * 0.05))])
    return (wave * 7000).astype(np.int16).tobytes()


CONFIRM_CHIME = make_confirm_chime()


def _is_quiet_tool(name: str, result: dict) -> bool:
    """
    Czy narzędzie jest „ciche” – zwraca ``True`` wtedy, gdy model nie musi
    wypowiadać żadnego zdania, a jedynie odtworzyć sygnał dźwiękowy.
    """
    QUIET_TOOLS = {
        "control_device",
        "control_room",
        "activate_scene",
        "run_script",
        "set_climate",
        "control_vacuum",
        "cancel_timer",
        "stop_timer_alarm",
    }
    if name not in QUIET_TOOLS or not isinstance(result, dict):
        return False
    if result.get("status") != "ok" or result.get("no_change"):
        return False
    # ``count`` == 0 oznacza, że nic się nie zmieniło – nie wymaga wypowiedzi.
    return result.get("count", 1) != 0


# ----------------------------------------------------------------------
#  DANE STANU SESJI
# ----------------------------------------------------------------------
@dataclass
class SessionState:
    """Wszystkie pola, które zmieniają się w czasie trwania jednej sesji."""
    # Transkrypcje
    input_transcript: List[str] = field(default_factory=list)
    output_transcript: List[str] = field(default_factory=list)

    # Zgromadzone audio, które trzeba odtworzyć po stronie klienta
    audio_out_chunks: List[bytes] = field(default_factory=list)

    # Lista wywołanych funkcji w formie czytelnego stringa
    pending_fc: List[str] = field(default_factory=list)

    # Licznik uruchomionych asynchronicznie narzędzi
    tools_running: int = 0
    tools_done: asyncio.Event = field(default_factory=asyncio.Event)

    # Informacje o bieżącej turze modelu
    turn_open: bool = False
    reply_owed: bool = False          # model już wypowiedział, ale czeka na wynik narzędzia
    responding_signaled: bool = False

    # Timestampy (float – wynik time.monotonic())
    last_response_ts: float = 0.0
    last_audio_ts: float = 0.0

    # Czy wysłano już wszystkie dane audio od klienta?
    send_done: bool = False

    # Event ustawiany przy każdej aktywności (audio, turn, tool)
    activity_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Czy sesja ma się już zamknąć (po wykryciu ciszy)?
    shutdown_requested: bool = False


# ----------------------------------------------------------------------
#  REZULTAT JEDNEJ TURY / SESJI
# ----------------------------------------------------------------------
@dataclass
class GeminiTurnResult:
    """Co zwraca metoda ``stream_audio``."""
    spoken_text: str                # pełny tekst wypowiedzi modelu
    tool_calls: str                 # spis wywołań narzędzi, np. "search_web(query='…')"
    audio_chunks: List[bytes]       # surowe PCM/Opus do odtworzenia w UI


# ----------------------------------------------------------------------
#  FUNKCJE POMOCNICZE
# ----------------------------------------------------------------------
def _build_tool_decls(room_keys: Tuple[str, ...], vacuum_enabled: bool) -> List[types.Tool]:
    """
    Przekształca specyfikację narzędzi z ``ai_common.build_tool_specs`` na
    ``types.FunctionDeclaration`` zgodne z Gemini.
    Wszystkie narzędzia mają zachowanie NON_BLOCKING.
    """
    decls = [
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters=spec["parameters"],
            behavior=types.Behavior.NON_BLOCKING,
        )
        for spec in build_tool_specs(list(room_keys), vacuum_enabled)
    ]
    return [types.Tool(function_declarations=decls)]


def _choose_idle_timeout(state: SessionState) -> float:
    """
    Zależnie od bieżącego stanu wybiera najodpowiedniejszy timeout
    (z klasy ``IdleTimeouts``).
    """
    if state.tools_running:
        return IdleTimeouts.GENERAL
    if state.reply_owed:
        return IdleTimeouts.AWAITING_REPLY
    if state.pending_fc and state.audio_out_chunks:
        return IdleTimeouts.AFTER_AUDIO
    if state.pending_fc:
        return IdleTimeouts.AFTER_FUNCTION
    return IdleTimeouts.GENERAL


async def _silence_watcher(state: SessionState) -> None:
    """
    Czeka na ``SILENCE_TIMEOUT_SEC`` sekund bez żadnej aktywności.
    Gdy timeout wygaśnie – ustawia ``state.shutdown_requested = True``,
    co powoduje zamknięcie połączenia.
    """
    while not state.shutdown_requested:
        try:
            await asyncio.wait_for(state.activity_event.wait(),
                                   timeout=SILENCE_TIMEOUT_SEC)
            # Aktywność została wykryta – czyścimy flagę i czekamy dalej
            state.activity_event.clear()
        except asyncio.TimeoutError:
            log.info("gemini.silence_timeout",
                     timeout=SILENCE_TIMEOUT_SEC,
                     session=getattr(state, "session_id", "unknown"))
            state.shutdown_requested = True
            break


# ----------------------------------------------------------------------
#  KLASA GEMINISESSION
# ----------------------------------------------------------------------
class GeminiSession:
    """
    Zarządza jedną sesją Gemini Live.
    - Odbiera strumień PCM od klienta.
    - Przesyła go do modelu (manualne VAD).
    - Odbiera odpowiedzi audio + tekst, a także wywołania funkcji.
    - Wykonuje funkcje asynchronicznie, odsyła ich wyniki do modelu.
    - Po 15 s ciszy zamyka sesję.
    """

    def __init__(
        self,
        client: genai.Client,
        entity_list: str,
        room_lights: dict,
        ha_context: str,
        history: list,
        on_function_call: Callable[[str, dict], Awaitable[dict]],
        voice: str | None = None,
        on_responding: Callable[[], None] | None = None,
        vacuum_enabled: bool = False,
        local_area_id: str = "",
    ) -> None:
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

        # Krótkie ID sesji – przydatne w logach/metricach
        self.session_id = os.urandom(6).hex()
        self._t0: float = 0.0

    # ------------------------------------------------------------------
    # 1️⃣ Budowanie promptu i konfiguracji LiveConnect
    # ------------------------------------------------------------------
    def _build_prompt(self) -> str:
        """
        Składa pełny prompt systemowy (z historią, kontekstem itp.)
        i dopina informacje o zachowaniu NON_BLOCKING tools.
        """
        from ai_common import build_prompt

        prompt = build_prompt(
            self.entity_list,
            self.ha_context,
            self.history,
            self.local_area_id,
        )

        async_tools_prompt = (
            "\n\nNarzędzia wykonują się w tle. Po wywołaniu narzędzia sterującego "
            "NIE potwierdzaj wykonania, zanim dostaniesz wynik — potwierdź dopiero "
            "na podstawie wyniku. "
            "Wyjątek: zanim wywołasz search_web albo inne narzędzie pobierające "
            "informacje z internetu, powiedz najpierw jedno bardzo krótkie zdanie, "
            "np. „Już sprawdzam.”, a dopiero potem wywołaj narzędzie. "
            "Przy sterowaniu urządzeniami, scenami i timerami nic nie mów przed "
            "wywołaniem."
        )
        return prompt + async_tools_prompt

    def _live_config(self, room_keys: Tuple[str, ...]) -> types.LiveConnectConfig:
        """Konfiguracja LiveConnect – audio, VAD, system‑prompt i tools."""
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice,
                    ),
                ),
                language_code=ASSISTANT_LANGUAGE,
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self._build_prompt())]
            ),
            tools=_build_tool_decls(room_keys, self.vacuum_enabled),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
        )

    # ------------------------------------------------------------------
    # 2️⃣ Wysyłanie audio do Gemini (ręczne VAD)
    # ------------------------------------------------------------------
    async def _send_audio(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        sess: types.LiveConnect,
        state: SessionState,
    ) -> None:
        """Przesyła podany strumień PCM do Gemini, sygnalizując start/end VAD."""
        chunk_no = 0
        async for chunk in audio_chunks:
            chunk_no += 1
            if chunk_no == 1:
                log.debug(
                    "gemini.send.start",
                    session=self.session_id,
                )
                # pierwszy pakiet → sygnał początku wypowiedzi użytkownika
                await sess.send_realtime_input(activity_start=types.ActivityStart())
            await sess.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={PCM_RATE}",
                )
            )
            # Każdy przychodzący fragment PCM liczy się jako aktywność
            state.activity_event.set()

        # Koniec strumienia – ręczny VAD
        if chunk_no:
            await sess.send_realtime_input(activity_end=types.ActivityEnd())
        else:
            await sess.send_realtime_input(audio_stream_end=True)

        state.send_done = True
        log.debug(
            "gemini.send.end",
            chunks=chunk_no,
            session=self.session_id,
        )

    # ------------------------------------------------------------------
    # 3️⃣ Wykonywanie batcha funkcji (tool‑call)
    # ------------------------------------------------------------------
    async def _run_tool_batch(
        self,
        calls: List[types.FunctionCall],
        sess: types.LiveConnect,
        state: SessionState,
    ) -> None:
        """
        Uruchamia wszystkie funkcje z jednego wywołania ``tool_call`` równolegle.
        Po zakończeniu odsyła wyniki do Gemini (z opcjonalnym ``SILENT``).
        """

        async def _exec_one(fc: types.FunctionCall) -> dict:
            args = dict(fc.args)
            try:
                if fc.name == "search_web":
                    # własna metoda, korzystająca z web_search()
                    return await asyncio.wait_for(self._do_search(args.get("query", "")),
                                                timeout=8)
                # domyślna funkcja – wywołana z kodu Home‑Assistant
                return await asyncio.wait_for(
                    self.on_function_call(fc.name, args), timeout=8
                )
            except asyncio.TimeoutError:
                return {"status": "error", "message": f"{fc.name} timed out"}
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "tool.exception",
                    name=fc.name,
                    error=str(exc),
                    session=self.session_id,
                )
                return {"status": "error", "message": str(exc)}

        # ---------------------------------
        # Uruchomienie wszystkich funkcji
        # ---------------------------------
        results = await asyncio.gather(*[_exec_one(fc) for fc in calls])

        # Czy wszystkie wyniki są "ciche" (tylko dźwięk)?
        quiet = QUIET_CONFIRMATIONS and all(
            _is_quiet_tool(fc.name, res) for fc, res in zip(calls, results)
        )
        extra = {}
        if quiet:
            extra["scheduling"] = types.FunctionResponseScheduling.SILENT

        await sess.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response=res,
                    **extra,
                )
                for fc, res in zip(calls, results)
            ]
        )
        log.debug(
            "gemini.tools.sent",
            quiet=quiet,
            elapsed_ms=int((time.monotonic() - self._t0) * 1000),
            session=self.session_id,
        )

        if quiet:
            # odtwórz jedynie dźwięk potwierdzający
            state.audio_out_chunks.append(CONFIRM_CHIME)
            await self._audio_sink(CONFIRM_CHIME)
        else:
            state.reply_owed = True
            state.last_response_ts = time.monotonic()

        # Updating counters
        state.tools_running -= 1
        state.tools_done.set()

    # ------------------------------------------------------------------
    # 4️⃣ Odbiór i obsługa wiadomości od Gemini
    # ------------------------------------------------------------------
    async def _receive_loop(
        self,
        sess: types.LiveConnect,
        state: SessionState,
        audio_sink: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """
        Główna pętla odbierająca wiadomości z Gemini.
        - Zapamiętuje transkrypcje.
        - Zbiera audio i tekst modelu.
        - Wykrywa wywołania narzędzi i uruchamia ``_run_tool_batch``.
        - Po każdej istotnej aktywności wywołuje ``state.activity_event.set()``.
        """
        self._audio_sink = audio_sink

        next_msg_fut: asyncio.Future | None = None
        msgs = sess.receive().__aiter__()

        while not state.shutdown_requested:
            timeout = _choose_idle_timeout(state)

            if next_msg_fut is None:
                next_msg_fut = asyncio.ensure_future(msgs.__anext__())
            tools_waiter = asyncio.ensure_future(state.tools_done.wait())

            done, _ = await asyncio.wait(
                {next_msg_fut, tools_waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            tools_waiter.cancel()

            # -----------------------------------------------------------
            #  a) Zakończyły się wszystkie narzędzia -> ewentualny koniec
            # -----------------------------------------------------------
            if tools_waiter in done:
                state.tools_done.clear()
                if (
                    state.tools_running == 0
                    and not state.reply_owed
                    and not state.turn_open
                ):
                    log.debug(
                        "gemini.quiet_tool_end",
                        session=self.session_id,
                    )
                    break
                continue

            # -----------------------------------------------------------
            #  b) Timeout – brak nowych wiadomości od Gemini
            # -----------------------------------------------------------
            if next_msg_fut not in done:
                log.debug(
                    "gemini.idle_timeout",
                    timeout=timeout,
                    state=state,
                    session=self.session_id,
                )
                break

            # -----------------------------------------------------------
            #  c) Otrzymano wiadomość
            # -----------------------------------------------------------
            try:
                msg = next_msg_fut.result()
            except StopAsyncIteration:
                # iterator zakończył się po turze – wracamy do nasłuchiwania
                next_msg_fut = None
                if state.tools_running or state.reply_owed:
                    msgs = sess.receive().__aiter__()
                    continue
                break
            finally:
                next_msg_fut = None

            # Każda przyjęta wiadomość jest **aktywnością**
            state.activity_event.set()

            # ----------------- debugowanie pól wiadomości -----------------
            msg_fields = [
                n
                for n in (
                    "setup_complete",
                    "server_content",
                    "tool_call",
                    "tool_call_cancellation",
                    "usage_metadata",
                    "go_away",
                    "session_resumption_update",
                )
                if getattr(msg, n, None) is not None
            ]
            if msg.server_content:
                msg_fields += [
                    n
                    for n in ("generation_complete", "turn_complete", "interrupted")
                    if getattr(msg.server_content, n, None)
                ]

            log.debug(
                "gemini.msg",
                fields=msg_fields,
                session=self.session_id,
            )

            sc = msg.server_content
            # -----------------------------------------------------------
            #   Server content – transkrypcje i audio modelu
            # -----------------------------------------------------------
            if sc:
                if sc.input_transcription and sc.input_transcription.text:
                    state.input_transcript.append(sc.input_transcription.text)
                if sc.output_transcription and sc.output_transcription.text:
                    state.output_transcript.append(sc.output_transcription.text)

                # -----------------------------------------------------------------
                # Model turn – audio (inline_data) i ewentualny tekst (part.text)
                # -----------------------------------------------------------------
                if sc.model_turn:
                    state.turn_open = True
                    if not state.responding_signaled:
                        state.responding_signaled = True
                        if self.on_responding:
                            self.on_responding()
                        log.debug(
                            "gemini.responding",
                            session=self.session_id,
                        )
                    for part in sc.model_turn.parts:
                        if part.inline_data:
                            state.audio_out_chunks.append(part.inline_data.data)
                            state.last_audio_ts = time.monotonic()
                            await audio_sink(part.inline_data.data)
                        if part.text:
                            state.output_transcript.append(part.text)

                # ------------------------------------------------------------
                # Turn zakończona – ale nie zamykamy sesji, wracamy do nasłuchiwania
                # ------------------------------------------------------------
                if sc.turn_complete or sc.generation_complete:
                    state.turn_open = False
                    # Jeśli po narzędziu model już wypowiedział coś i audio już dotarło,
                    # uznajemy, że zaległa odpowiedź została "spłacona".
                    if state.reply_owed and state.last_audio_ts > state.last_response_ts:
                        state.reply_owed = False
                    # Jeśli nadal czekamy na wynik narzędzia lub na kolejny obrót,
                    # nie przerywamy pętli – po prostu czekamy dalej.
                    if state.tools_running or state.reply_owed:
                        log.debug(
                            "gemini.turn_done_waiting",
                            tools=state.tools_running,
                            reply_owed=state.reply_owed,
                            session=self.session_id,
                        )
                        continue
                    # Normalny koniec tury – wracamy do nasłuchiwania nowych
                    # zdarzeń (audio lub tool‑call).  Nie wychodzimy z pętli.
                    continue

            # -----------------------------------------------------------
            #   Tool call – uruchomiamy batch w tle
            # -----------------------------------------------------------
            if msg.tool_call:
                state.turn_open = True
                if not state.responding_signaled:
                    state.responding_signaled = True
                    if self.on_responding:
                        self.on_responding()
                    log.debug(
                        "gemini.tool_call_received",
                        session=self.session_id,
                    )
                for fc in msg.tool_call.function_calls:
                    state.pending_fc.append(f"{fc.name}({dict(fc.args)})")
                state.tools_running += 1
                asyncio.create_task(self._run_tool_batch(msg.tool_call.function_calls,
                                                        sess,
                                                        state))

        # -----------------------------------------------------------
        # Po wyjściu z pętli – czekamy (max 10 s) na wszystkie wciąż uruchomione narzędzia
        # -----------------------------------------------------------
        if state.tools_running:
            log.debug(
                "gemini.waiting_for_tools",
                remaining=state.tools_running,
                session=self.session_id,
            )
            try:
                await asyncio.wait_for(state.tools_done.wait(), timeout=10)
            except asyncio.TimeoutError:
                log.warning(
                    "gemini.tools_still_running_after_timeout",
                    remaining=state.tools_running,
                    session=self.session_id,
                )

    # ------------------------------------------------------------------
    # 5️⃣ Publiczna metoda – jednorazowa sesja (z czatem) / odtwarzanie audio
    # ------------------------------------------------------------------
    async def stream_audio(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        on_audio_out: Callable[[bytes], Awaitable[None]],
    ) -> GeminiTurnResult:
        """
        Strumieniuje podany audio do Gemini Live, obsługuje tool‑calls i zwraca
        rezultat po zakończeniu sesji (lub po wykryciu 15 s ciszy).
        """
        self._t0 = time.monotonic()
        state = SessionState()
        # Dzięki ``session_id`` możemy szybciej identyfikować logi
        state.session_id = self.session_id

        config = self._live_config(tuple(self.room_lights.keys()))

        async with self.client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config,
        ) as sess:
            # Uruchamiamy watchdog monitorujący ciszę
            watcher = asyncio.create_task(_silence_watcher(state))

            # Uruchamiamy dwa główne zadania – wysyłanie i odbiór
            send_task = asyncio.create_task(self._send_audio(audio_chunks, sess, state))
            recv_task = asyncio.create_task(self._receive_loop(sess, state, on_audio_out))

            # Czekamy, aż jedno z trzech zadań zakończy się:
            #   * watchdog wykryje ciszę,
            #   * wysyłanie audio się skończy (klient zamknął połączenie),
            #   * odbiór napotka nieodrecoverable błąd.
            while not state.shutdown_requested:
                done, _ = await asyncio.wait(
                    {send_task, recv_task, watcher},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if watcher in done:
                    # Cisza → zamykamy sesję
                    state.shutdown_requested = True
                    break
                if send_task in done or recv_task in done:
                    # Błąd w jednej z części – przerywamy natychmiast
                    state.shutdown_requested = True
                    break

            # Sprzątanie: zamykamy połączenie z Gemini i anulujemy watchdog
            await sess.close()
            watcher.cancel()

        # Przygotowanie wyniku do zwrócenia
        spoken = "".join(state.output_transcript).strip()
        tools = " ".join(state.pending_fc).strip()
        log.info(
            "gemini.session_finished",
            duration_ms=int((time.monotonic() - self._t0) * 1000),
            spoken_len=len(spoken),
            tool_calls=len(state.pending_fc),
            session=self.session_id,
        )
        return GeminiTurnResult(
            spoken_text=spoken,
            tool_calls=tools,
            audio_chunks=state.audio_out_chunks,
        )

    # ------------------------------------------------------------------
    # 6️⃣ Prosty wrapper do wyszukiwania w internecie (używany jako tool)
    # ------------------------------------------------------------------
    async def _do_search(self, query: str) -> dict:
        """Wykorzystuje ``ai_common.web_search`` – zwraca dict wyników."""
        return await web_search(query, self.client)


# ----------------------------------------------------------------------
#  PRZYKŁADOWE UŻYCIE (do wklejenia w innym module)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Minimalny test uruchamiany „na sucho”.  Nie wymaga prawdziwego
    strumienia audio – po prostu odczytuje kilka przykładowych PCM‑chunków
    z pliku i wyświetla wynik w konsoli.
    """
    import sys

    async def dummy_function(name: str, args: dict) -> dict:
        """Przykładowa funkcja, którą można podpiąć do ``on_function_call``."""
        print(f"[dummy] called {name} with {args!r}")
        await asyncio.sleep(0.1)
        return {"status": "ok", "message": f"Executed {name}"}

    async def fake_audio_source() -> AsyncGenerator[bytes, None]:
        """Generuje kilka losowych chunków PCM (16‑bit, mono, 16 kHz)."""
        for _ in range(5):
            # 0.2 s czystego szumu – wystarczy do demonstracji
            yield os.urandom(int(0.2 * PCM_RATE * 2))  # 2 bajty na próbkę
            await asyncio.sleep(0.2)

    async def audio_sink(chunk: bytes) -> None:
        """W tym przykładzie po prostu liczy liczbę odebranych chunków."""
        print(f"[audio_sink] got {len(chunk)} B")

    async def run_demo():
        client = genai.GenerativeModel(GEMINI_MODEL).client
        session = GeminiSession(
            client=client,
            entity_list="",
            room_lights={},
            ha_context="",
            history=[],
            on_function_call=dummy_function,
        )
        result = await session.stream_audio(fake_audio_source(), audio_sink)
        print("\n=== RESULT ===")
        print("Spoken:", result.spoken_text)
        print("Tools :", result.tool_calls)
        print("Audio chunks:", len(result.audio_chunks))

    asyncio.run(run_demo())
