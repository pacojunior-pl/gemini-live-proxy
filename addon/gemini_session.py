#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini Live Session – refaktoryzowany i gotowy do produkcji.

* Strumieniujemy PCM (16‑bit, 16 kHz, mono) do modelu gemini‑3.8‑live.
* Narzędzia (search, HA‑actions) są uruchamiane asynchronicznie (NON_BLOCKING).
* Po każdej turze modelu nie zamykamy połączenia – czekamy na kolejne
  wypowiedzi użytkownika.  Sesja zostaje zamknięta po 15 s ciszy
  (wartość konfigurowalna przez env `GEMINI_SILENCE_DURATION_MS`).
* W razie limitu 429 przy `search_web` używamy DuckDuckGo jako fallback
  i zwracamy przyjazny komunikat.
* Logowanie jest realizowane przez ``structlog``.
"""

# ----------------------------------------------------------------------
# IMPORTY
# ----------------------------------------------------------------------
import asyncio
import json
import os
import time
import structlog
from dataclasses import dataclass, field
from typing import AsyncGenerator, Awaitable, Callable, List, Tuple

from google.generativeai import genai, types

# ----------------------------------------------------------------------
# MODUŁY WŁASNE
# ----------------------------------------------------------------------
# ``ai_common`` musi być w PYTHONPATH (zawiera m.in. build_prompt,
# build_tool_specs, web_search, QUIET_CONFIRMATIONS itp.).
from ai_common import (
    ASSISTANT_LANGUAGE,
    QUIET_CONFIRMATIONS,
    build_tool_specs,
    debug_log,
    web_search,
)

# ----------------------------------------------------------------------
# KONFIGURACJA I STAŁE
# ----------------------------------------------------------------------
log = structlog.get_logger(__name__)

def _env(name: str, default: str) -> str:
    """Wczytuje zmienną środowiskową (obsługuje 'null' zwracane przez bashio)."""
    val = (os.getenv(name) or "").strip()
    return default if not val or val.lower() == "null" else val


GEMINI_MODEL     = _env("GEMINI_MODEL", "gemini-3.8-live")
GEMINI_VOICE     = os.getenv("GEMINI_VOICE", "Vega")
PCM_RATE         = 16000                     # musi pasować do enkodera w kliencie
SILENCE_TIMEOUT  = float(os.getenv("GEMINI_SILENCE_DURATION_MS", "15000")) / 1000.0

# ----------------------------------------------------------------------
# TIMEOUTY (sekundy) – używane w pętli odbioru
# ----------------------------------------------------------------------
class IdleTimeouts:
    AFTER_FUNCTION = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AFTER_FUNCTION", "1.5"))
    AFTER_AUDIO    = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AFTER_AUDIO",   "1.2"))
    GENERAL        = float(os.getenv("RECEIVE_IDLE_TIMEOUT_GENERAL",       "8.0"))
    AWAITING_REPLY = float(os.getenv("RECEIVE_IDLE_TIMEOUT_AWAITING_REPLY","5.0"))


# ----------------------------------------------------------------------
# KLASA STANU SESJI
# ----------------------------------------------------------------------
@dataclass
class SessionState:
    """Wszystkie pola, które zmieniają się podczas działania jednej sesji."""
    input_transcript: List[str] = field(default_factory=list)   # co model usłyszał (transkrypcja)
    output_transcript: List[str] = field(default_factory=list)  # co model powiedział (tekst)
    audio_out_chunks: List[bytes] = field(default_factory=list) # surowe audio (PCM/Opus)

    pending_fc: List[str] = field(default_factory=list)   # np. "search_web(query='…')"
    tools_running: int = 0
    tools_done: asyncio.Event = field(default_factory=asyncio.Event)

    turn_open: bool = False
    reply_owed: bool = False
    responding_signaled: bool = False

    last_response_ts: float = 0.0
    last_audio_ts: float = 0.0

    send_done: bool = False

    # Event który jest ustawiany przy każdej aktywności (audio, turn, tool)
    activity_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Flaga zamknięcia po wykryciu ciszy
    shutdown_requested: bool = False

    # Id sesji – przydatne w logach
    session_id: str = field(default_factory=lambda: os.urandom(6).hex())


# ----------------------------------------------------------------------
#  REZULTAT JEDNEJ TURY / SESJI
# ----------------------------------------------------------------------
@dataclass
class GeminiTurnResult:
    """Wartość zwracana przez ``GeminiSession.stream_audio``."""
    spoken_text: str                  # kompletna treść wypowiedzi modelu
    tool_calls: str                   # lista wywołań narzędzi w formie tekstowej
    audio_chunks: List[bytes]           # audio do odtworzenia po stronie UI


# ----------------------------------------------------------------------
#  FUNKCJE POMOCNICZE
# ----------------------------------------------------------------------
def _build_tool_decls(room_keys: Tuple[str, ...], vacuum_enabled: bool) -> List[types.Tool]:
    """Konwertuje specyfikację z ``ai_common.build_tool_specs`` na deklaracje Gemini."""
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
    """Wybiera najbardziej adekwatny timeout w zależności od bieżącego stanu."""
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
    Czeka ``SILENCE_TIMEOUT`` sekund na brak aktywności.
    Gdy timeout wygaśnie → ustawiamy ``state.shutdown_requested = True``.
    """
    while not state.shutdown_requested:
        try:
            await asyncio.wait_for(state.activity_event.wait(),
                                   timeout=SILENCE_TIMEOUT)
            # Aktywność – wyzeruj flagę i kontynuuj oczekiwanie
            state.activity_event.clear()
        except asyncio.TimeoutError:
            log.info("gemini.silence_timeout",
                     timeout=SILENCE_TIMEOUT,
                     session=state.session_id)
            state.shutdown_requested = True
            break


def _is_quiet_tool(name: str, result: dict) -> bool:
    """Czy narzędzie jest „ciche” (potwierdzenie wyłącznie dźwiękiem)."""
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
    return result.get("count", 1) != 0


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


# ----------------------------------------------------------------------
#  KLASA GEMINISESSION
# ----------------------------------------------------------------------
class GeminiSession:
    """
    Główna klasa zarządzająca jedną sesją Gemini Live.
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
        self.session_id = os.urandom(6).hex()      # przydatne w logach

    # ------------------------------------------------------------------
    # 1️⃣ Budowanie promptu i konfiguracji
    # ------------------------------------------------------------------
    def _build_prompt(self) -> str:
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
            "informacje z internetu, powiedz najpierw bardzo krótkie zdanie, np. "
            "„Już sprawdzam.”, a dopiero potem wywołaj narzędzie."
        )
        return prompt + async_tools_prompt

    def _live_config(self, room_keys: Tuple[str, ...]) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice
                    )
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
    # 2️⃣ Wysyłanie audio (ręczne VAD)
    # ------------------------------------------------------------------
    async def _send_audio(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        sess: types.LiveConnect,
        state: SessionState,
    ) -> None:
        """Przesyła PCM → Gemini. Pierwszy chunk uruchamia ``activity_start``."""
        chunk_no = 0
        async for chunk in audio_chunks:
            chunk_no += 1
            if chunk_no == 1:
                log.debug("gemini.send.start", session=self.session_id)
                await sess.send_realtime_input(activity_start=types.ActivityStart())
            await sess.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={PCM_RATE}",
                )
            )
            # Każdy przychodzący fragment PCM jest aktywnością
            state.activity_event.set()

        # Koniec strumienia audio – ręczne VAD
        if chunk_no:
            await sess.send_realtime_input(activity_end=types.ActivityEnd())
        else:
            await sess.send_realtime_input(audio_stream_end=True)

        state.send_done = True
        log.debug("gemini.send.end", chunks=chunk_no, session=self.session_id)

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
        Uruchamia wszystkie funkcje z jednego ``tool_call`` równolegle,
        po czym odsyła ich wyniki do Gemini.
        """
        async def _exec_one(fc: types.FunctionCall) -> dict:
            args = dict(fc.args)
            try:
                if fc.name == "search_web":
                    # Najpierw spróbuj użyć Gemini + Google Search, po 429 fallback do DuckDuckGo
                    return await self._search_with_fallback(args.get("query", ""))
                # Inne funkcje – wywołanie dostarczonego callbacka
                return await self.on_function_call(fc.name, args)
            except Exception as exc:   # noqa: BLE001
                log.error("tool.exception",
                          name=fc.name,
                          error=str(exc),
                          session=self.session_id)
                return {"status": "error", "message": str(exc)}

        results = await asyncio.gather(*[_exec_one(fc) for fc in calls])

        quiet = QUIET_CONFIRMATIONS and all(
            _is_quiet_tool(fc.name, res) for fc, res in zip(calls, results)
        )
        extra = {"scheduling": types.FunctionResponseScheduling.SILENT} if quiet else {}

        await sess.send_tool_response(
            function_responses=[
                types.FunctionResponse(id=fc.id,
                                      name=fc.name,
                                      response=res,
                                      **extra)
                for fc, res in zip(calls, results)
            ]
        )
        log.debug("gemini.tools.sent",
                  quiet=quiet,
                  elapsed_ms=int((time.monotonic() - self._t0) * 1000),
                  session=self.session_id)

        if quiet:
            # Odgrywamy jedynie dźwięk potwierdzenia
            state.audio_out_chunks.append(CONFIRM_CHIME)
            await self._audio_sink(CONFIRM_CHIME)
        else:
            state.reply_owed = True
            state.last_response_ts = time.monotonic()

        state.tools_running -= 1
        state.tools_done.set()

    # ------------------------------------------------------------------
    # 4️⃣ Odbiór wiadomości od Gemini
    # ------------------------------------------------------------------
    async def _receive_loop(
        self,
        sess: types.LiveConnect,
        state: SessionState,
        audio_sink: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """
        Nasłuchuje strumienia Gemini, aktualizuje stan i uruchamia narzędzia.
        Każde przyjęte zdarzenie (audio, turn, tool) wywołuje
        ``state.activity_event.set()`` – w ten sposób watchdog odświeża timeout.
        """
        self._audio_sink = audio_sink

        next_msg: asyncio.Future | None = None
        msgs = sess.receive().__aiter__()

        while not state.shutdown_requested:
            timeout = _choose_idle_timeout(state)

            if next_msg is None:
                next_msg = asyncio.ensure_future(msgs.__anext__())
            tools_waiter = asyncio.ensure_future(state.tools_done.wait())

            done, _ = await asyncio.wait(
                {next_msg, tools_waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            tools_waiter.cancel()

            # ------------------- wszystkie narzędzia skończyły -------------------
            if tools_waiter in done:
                state.tools_done.clear()
                if state.tools_running == 0 and not state.reply_owed and not state.turn_open:
                    log.debug("gemini.quiet_tool_end",
                              session=self.session_id)
                    break
                continue

            # ------------------- timeout -------------------
            if next_msg not in done:
                log.debug("gemini.idle_timeout",
                          timeout=timeout,
                          state=state,
                          session=self.session_id)
                break

            # ------------------- odebrano wiadomość -------------------
            try:
                message = next_msg.result()
            except StopAsyncIteration:
                # Iterator zakończył się po „turn_complete”.  Nie przerywamy,
                # po prostu wracamy do nasłuchiwania kolejnych zdarzeń.
                next_msg = None
                if state.tools_running or state.reply_owed:
                    msgs = sess.receive().__aiter__()
                    continue
                break
            finally:
                next_msg = None

            # Każda wiadomość = aktywność → resetujemy watchdog.
            state.activity_event.set()

            # ----------------- diagnostyka -----------------
            fields = [
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
                if getattr(message, n, None) is not None
            ]
            if sc := message.server_content:
                fields += [
                    n
                    for n in ("generation_complete", "turn_complete", "interrupted")
                    if getattr(sc, n, None)
                ]
            log.debug("gemini.msg",
                      fields=fields,
                      session=self.session_id)

            # ----------------- SERVER CONTENT -----------------
            if sc:
                if sc.input_transcription and sc.input_transcription.text:
                    state.input_transcript.append(sc.input_transcription.text)
                if sc.output_transcription and sc.output_transcription.text:
                    state.output_transcript.append(sc.output_transcription.text)

                # ---- model turn (audio + ewentualny tekst) ----
                if sc.model_turn:
                    state.turn_open = True
                    if not state.responding_signaled:
                        state.responding_signaled = True
                        if self.on_responding:
                            self.on_responding()
                    for part in sc.model_turn.parts:
                        if part.inline_data:
                            state.audio_out_chunks.append(part.inline_data.data)
                            state.last_audio_ts = time.monotonic()
                            await audio_sink(part.inline_data.data)
                        if part.text:
                            state.output_transcript.append(part.text)

                # ---- zakończenie tury ----
                if sc.turn_complete or sc.generation_complete:
                    state.turn_open = False
                    # Jeśli odtwarzane audio już dotarło po wywołaniu narzędzia,
                    # uznajemy, że odpowiedź została już wypowiedziana.
                    if state.reply_owed and state.last_audio_ts > state.last_response_ts:
                        state.reply_owed = False
                    # Jeśli jeszcze czekamy na wynik narzędzia – pozostajemy w pętli.
                    if state.tools_running or state.reply_owed:
                        log.debug("gemini.turn_complete_waiting",
                                  tools=state.tools_running,
                                  reply_owed=state.reply_owed,
                                  session=self.session_id)
                        continue
                    # Normalny koniec tury – wracamy do nasłuchiwania.
                    continue

            # ----------------- TOOL CALL -----------------
            if tc := message.tool_call:
                state.turn_open = True
                if not state.responding_signaled:
                    state.responding_signaled = True
                    if self.on_responding:
                        self.on_responding()
                for fc in tc.function_calls:
                    state.pending_fc.append(f"{fc.name}({dict(fc.args)})")
                state.tools_running += 1
                # Uruchamiamy batch w tle – nie blokujemy odbioru kolejnych pkt audio.
                asyncio.create_task(self._run_tool_batch(tc.function_calls,
                                                         sess,
                                                         state))

        # -------------------------------------------------------
        # Po wyjściu z pętli: czekamy (max 10 s) na narzędzia,
        # które wciąż mogą działać.
        # -------------------------------------------------------
        if state.tools_running:
            log.debug("gemini.waiting_for_tools",
                      remaining=state.tools_running,
                      session=self.session_id)
            try:
                await asyncio.wait_for(state.tools_done.wait(), timeout=10)
            except asyncio.TimeoutError:
                log.warning("gemini.tools_not_finished_after_timeout",
                            remaining=state.tools_running,
                            session=self.session_id)

    # ------------------------------------------------------------------
    # 5️⃣ Publiczna metoda – uruchamia jedną sesję (audio ↔ Gemini)
    # ------------------------------------------------------------------
    async def stream_audio(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        on_audio_out: Callable[[bytes], Awaitable[None]],
    ) -> GeminiTurnResult:
        """
        - Strumieniuje podany PCM → Gemini Live.
        - Obsługuje tool‑calls (z fallbackiem przy 429).
        - Pozostawia sesję otwartą, dopóki nie nastąpi 15 s ciszy.
        - Zwraca ``GeminiTurnResult``.
        """
        self._t0 = time.monotonic()
        state = SessionState()
        state.session_id = self.session_id

        config = self._live_config(tuple(self.room_lights.keys()))

        async with self.client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config,
        ) as sess:
            # -------- watchdog ciszy ----------
            watcher = asyncio.create_task(_silence_watcher(state))

            # -------- dwa równoległe zadania: send / receive ----------
            send_task = asyncio.create_task(self._send_audio(audio_chunks, sess, state))
            recv_task = asyncio.create_task(self._receive_loop(sess, state, on_audio_out))

            # -------- czekamy na dowolny sygnał zakończenia (watcher, błąd, zamknięcie) ----------
            while not state.shutdown_requested:
                done, _ = await asyncio.wait(
                    {send_task, recv_task, watcher},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if watcher in done:
                    state.shutdown_requested = True
                    break
                if send_task in done or recv_task in done:
                    # Coś się nie udało – zamykamy sesję.
                    state.shutdown_requested = True
                    break

            # -------- Sprzątanie ----------
            await sess.close()
            watcher.cancel()

        # -------- przygotowanie wyniku ----------
        spoken = "".join(state.output_transcript).strip()
        tools  = " ".join(state.pending_fc).strip()
        log.info("gemini.session_finished",
                 duration_ms=int((time.monotonic() - self._t0) * 1000),
                 spoken_len=len(spoken),
                 tool_calls=len(state.pending_fc),
                 session=self.session_id)

        return GeminiTurnResult(
            spoken_text=spoken,
            tool_calls=tools,
            audio_chunks=state.audio_out_chunks,
        )

    # ------------------------------------------------------------------
    # 6️⃣ Wyszukiwanie z fallbackiem (Google → DuckDuckGo)
    # ------------------------------------------------------------------
    async def _search_with_fallback(self, query: str) -> dict:
        """
        Najpierw próbuje użyć ``web_search`` (Google + Gemini).
        Jeśli zwróci błąd 429, przełącza się na DuckDuckGo.
        Zwraca zawsze słownik ``{'status': ..., 'content': ...}``.
        """
        try:
            return await web_search(query, self.client)      # Twoja funkcja w ai_common
        except Exception as exc:
            # Sprawdzamy, czy to limit Gemini
            if isinstance(exc, genai.exceptions.RateLimitError) or (
                isinstance(exc, genai.exceptions.GoogleAPIError) and
                getattr(exc, "status_code", None) == 429
            ):
                log.warning("gemini.search.quota_exhausted",
                            query=query,
                            session=self.session_id)
                # ----- fallback to DuckDuckGo -----------------
                try:
                    return await self._duckduckgo_search(query)
                except Exception as e2:
                    log.error("gemini.search.duckduckgo_failed",
                              error=str(e2),
                              session=self.session_id)
                    return {"status": "error",
                            "message": "Nie udało się wykonać wyszukiwania – quota wyczerpana."}
            else:
                # Inny nieoczekiwany błąd – propagujemy dalej
                raise

    async def _duckduckgo_search(self, query: str) -> dict:
        """Minimalny, szybki fallback – zapytanie HTTP do DuckDuckGo (JSON)."""
        import httpx
        url = "https://duckduckgo.com/jslite"
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(url, json={"q": query})
            data = resp.json()
        # Zwracamy pierwszy wynik w prostym formacie:
        if data.get("results"):
            first = data["results"][0]
            return {
                "status": "ok",
                "title": first.get("title", ""),
                "link": first.get("url", ""),
                "snippet": first.get("description", ""),
            }
        return {"status": "error", "message": "Brak wyników w DuckDuckGo"}

# ----------------------------------------------------------------------
#  PRZYKŁADOWE UŻYCIE (do uruchomienia bez front‑endu)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Demonstruje działanie w trybie “dry‑run”.  Nie wymaga prawdziwego mikrafonu –
    generuje sztuczne PCM‑chunki i wypisuje rezultat na konsolę.
    """
    async def dummy_function(name: str, args: dict) -> dict:
        print(f"[dummy] {name} -> {args}")
        await asyncio.sleep(0.1)
        return {"status": "ok", "message": f"Executed {name}"}

    async def fake_audio_source() -> AsyncGenerator[bytes, None]:
        """5 losowych chunków (po 0,2 s każdy)."""
        for _ in range(5):
            # 0.2 s czystego szumu
            yield os.urandom(int(0.2 * PCM_RATE * 2))   # 2 bajty na próbkę
            await asyncio.sleep(0.2)

    async def audio_sink(chunk: bytes) -> None:
        print(f"[audio_sink] received {len(chunk)} B")

    async def run_demo():
        client = genai.GenerativeModel(GEMINI_MODEL).client
        sess = GeminiSession(
            client=client,
            entity_list="",
            room_lights={},
            ha_context="",
            history=[],
            on_function_call=dummy_function,
        )
        result = await sess.stream_audio(fake_audio_source(), audio_sink)
        print("\n=== RESULT ===")
        print("Spoken :", result.spoken_text)
        print("Tools  :", result.tool_calls)
        print("Audio  :", len(result.audio_chunks), "chunks")
    asyncio.run(run_demo())
