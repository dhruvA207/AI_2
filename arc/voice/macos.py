"""Apple's on-device speech *recognition*.

Chosen over Whisper for two reasons that were measured rather than assumed:

*Licence.* ``mlx-whisper`` is MIT itself but drags in ``torch`` (Apache-2.0 + BSD-2 +
BSD-3 + BSL-1.0 + MIT), ``scipy``/``numba``/``numpy`` (BSD) and ``tqdm`` (MPL-2.0),
which breaks the Apache-2.0/MIT rule in BRIEF §0.1 — the same way ``trafilatura ->
courlan -> tld`` did. ``pyobjc-framework-Speech`` is MIT and 9 KB.

*Latency.* Apple's recogniser streams partial results while you speak, so the
transcript lands at end-of-speech. Whisper is batch: wait for silence, *then* infer.
At ~14 tok/s the model is the bottleneck anyway, which is why the session speaks
sentence by sentence instead of trying to shave milliseconds here.

Three things in this file are load-bearing and were each found the hard way:

1. **Usage descriptions are injected into the main bundle at import.** An unbundled
   Python has no Info.plist, so macOS silently refuses to even *prompt* for speech
   recognition and the status stays ``notDetermined`` forever — no error, no dialog.
2. **Recognition is pinned to en-US with ``requiresOnDeviceRecognition``.** Verified on
   this machine: en-US is on-device, en-GB and en-IN are not ("No Assistant asset for
   language") and would stream audio to Apple. The flag makes that fail loudly.
3. **Amplitude is computed from the PCM tap in pure Python.** numpy would be the
   obvious tool and is BSD-3, so it is not available; sub-sampling 64 points from each
   buffer is accurate enough for a visual and costs nothing.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from arc.errors import ArcError
from arc.log import get_logger
from arc.voice.base import (
    LevelHandler,
    ResultHandler,
    SpeechRecognizer,
    Transcript,
)

_log = get_logger(__name__)

#: The only locale with an on-device asset on this machine. See the module docstring.
ON_DEVICE_LOCALE = "en-US"

#: How fast the tracked noise floor may creep upward, per audio buffer (~90/s). Slow
#: enough that speech cannot drag the floor up with it, fast enough to follow a fan
#: switching on within a couple of seconds.
_FLOOR_RISE = 0.0015

#: Nothing below this counts as speech however quiet the room is, so a silent room
#: with a floor near zero does not turn every rustle into an utterance.
_ABSOLUTE_FLOOR = 0.06


def _inject_usage_descriptions() -> None:
    """Give the process an Info.plist entry so macOS will prompt for the microphone.

    Without this the TCC prompt never appears and authorisation stays ``notDetermined``
    — the failure is completely silent, which is why this runs before any framework
    import rather than being left to the caller.
    """
    from Foundation import NSBundle

    info = NSBundle.mainBundle().infoDictionary()
    if info is None:  # pragma: no cover - defensive
        return
    info.setdefault(
        "NSSpeechRecognitionUsageDescription",
        "ARC transcribes what you say, on this machine, so you can talk to it.",
    )
    info.setdefault(
        "NSMicrophoneUsageDescription",
        "ARC listens only while you hold the microphone open with Command-S.",
    )


def available() -> bool:
    """Whether the Apple speech frameworks can be imported."""
    try:
        _inject_usage_descriptions()
        import AVFoundation  # noqa: F401
        import Speech  # noqa: F401
    except Exception:
        return False
    return True


def authorization() -> dict[str, str]:
    """Report microphone and speech-recognition grants, for ``arc doctor``."""
    names = {0: "not determined", 1: "denied", 2: "restricted", 3: "granted"}
    try:
        _inject_usage_descriptions()
        import AVFoundation as AV
        import Speech
    except Exception as exc:
        return {"speech": f"unavailable ({exc})", "microphone": "unavailable"}

    return {
        "speech": names.get(Speech.SFSpeechRecognizer.authorizationStatus(), "unknown"),
        "microphone": names.get(
            AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio),
            "unknown",
        ),
    }


def request_authorization(timeout: float = 30.0) -> bool:
    """Trigger the TCC prompt and wait for an answer. Returns True when granted."""
    _inject_usage_descriptions()
    import Speech
    from Foundation import NSDate, NSRunLoop

    done = threading.Event()
    box: dict[str, int] = {}

    def handler(status: int) -> None:
        box["status"] = status
        done.set()

    Speech.SFSpeechRecognizer.requestAuthorization_(handler)

    # The callback is delivered through the run loop, so it has to be pumped rather
    # than merely waited on.
    loop = NSRunLoop.currentRunLoop()
    waited = 0.0
    while waited < timeout and not done.is_set():
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
        waited += 0.2
    return box.get("status") == 3


class AppleRecognizer(SpeechRecognizer):
    """On-device speech recognition via ``SFSpeechRecognizer``."""

    def __init__(
        self,
        locale: str = ON_DEVICE_LOCALE,
        *,
        require_on_device: bool = True,
        silence_ms: int = 900,
        speech_margin: float = 0.12,
    ) -> None:
        _inject_usage_descriptions()
        try:
            import AVFoundation as AV
            import Speech
            from Foundation import NSLocale
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ArcError(
                "voice support needs the Apple speech bindings: pip install 'arc[voice]'"
            ) from exc

        self._AV = AV
        self._Speech = Speech
        self._locale = locale
        self._require_on_device = require_on_device

        recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(
            NSLocale.alloc().initWithLocaleIdentifier_(locale)
        )
        if recognizer is None:
            raise ArcError(f"no speech recogniser for locale {locale!r}")

        self._on_device = bool(recognizer.supportsOnDeviceRecognition())
        if require_on_device and not self._on_device:
            raise ArcError(
                f"locale {locale!r} has no on-device speech asset, so recognition would "
                f"send your audio to Apple's servers. ARC is local-first, so this is "
                f"refused. Use {ON_DEVICE_LOCALE!r}, or install the dictation asset for "
                f"{locale!r} in System Settings > Keyboard > Dictation."
            )

        self._recognizer = recognizer
        self._silence = max(0.2, silence_ms / 1000.0)
        self._margin = speech_margin
        #: Tracked ambient level. Starts high so the first buffers do not read as
        #: speech before the floor has settled.
        self._floor = 1.0
        self._lock = threading.Lock()
        self._engine: Any = None
        self._request: Any = None
        self._task: Any = None
        self._listening = False
        self._on_result: ResultHandler | None = None
        #: Latest partial, and when speech was last heard. Together these are the
        #: endpoint detector — see ``_maybe_endpoint``.
        self._pending = ""
        self._last_voice = 0.0
        #: Set while ARC is speaking. The tap keeps running (so levels still move) but
        #: no audio reaches the recogniser, which is what stops it hearing itself.
        self._suspended = False

    @property
    def name(self) -> str:
        return f"apple-speech:{self._locale}"

    @property
    def on_device(self) -> bool:
        return self._on_device

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start(self, on_result: ResultHandler, on_level: LevelHandler) -> None:
        """Open the microphone and begin recognising.

        **The host must be pumping the main run loop** — see :func:`pump`. Audio tap
        blocks are invoked on the audio thread, so levels arrive regardless, but
        ``SFSpeechRecognitionTask`` delivers its results on the *main queue*. Without a
        main-loop pump you get a microphone that is provably capturing audio (levels
        move, the recording dot lights) and a recogniser that never calls back once.
        That reads as "recognition is broken" when nothing is broken at all; it was
        measured here as 100 level callbacks and 0 transcripts.
        """
        with self._lock:
            if self._listening:
                return

            AV = self._AV
            Speech = self._Speech

            request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
            request.setShouldReportPartialResults_(True)
            request.setRequiresOnDeviceRecognition_(self._require_on_device)

            engine = AV.AVAudioEngine.alloc().init()
            node = engine.inputNode()

            # Prepare *before* reading the format. The engine can otherwise hand back a
            # stale one — observed reporting 24000 Hz while the device was actually at
            # 48000, right after Gemini's 24 kHz playback had reconfigured the shared
            # audio device. The tap then captures at the wrong rate, and the audio is
            # loud but unintelligible: 130 buffers at peak 0.53 and the recogniser
            # replying "No speech detected". Nothing errors, so it looks like the
            # microphone works and recognition is broken.
            engine.prepare()
            fmt = node.outputFormatForBus_(0)
            if fmt is None or fmt.sampleRate() <= 0:
                fmt = node.inputFormatForBus_(0)
            if fmt is None or fmt.sampleRate() <= 0:
                raise ArcError(
                    "the audio input reported no usable format. Another app may hold the "
                    "microphone; check System Settings > Sound > Input."
                )
            _log.info(
                "audio input", extra={"rate": fmt.sampleRate(), "channels": fmt.channelCount()}
            )

            def tap(buffer: Any, _when: Any) -> None:
                # self._request rather than the captured `request`: endpointing swaps in
                # a fresh one per utterance, and the tap must follow it.
                current = self._request
                if current is not None and not self._suspended:
                    current.appendAudioPCMBuffer_(buffer)
                level = _rms(buffer)
                if level is not None:
                    if not self._suspended:
                        self._maybe_endpoint(level)
                    try:
                        on_level(level)
                    except Exception:  # pragma: no cover - a UI callback must not kill audio
                        _log.exception("level handler failed")

            node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap)

            def handler(result: Any, error: Any) -> None:
                if result is not None:
                    text = str(result.bestTranscription().formattedString())
                    self._pending = text
                    try:
                        # Always reported as partial. In buffer mode Apple only sets
                        # isFinal when the audio *ends*, so while the mic stays open a
                        # final never arrives — `_maybe_endpoint` is what decides an
                        # utterance is over.
                        on_result(Transcript(text=text, is_final=False))
                    except Exception:  # pragma: no cover
                        _log.exception("result handler failed")
                if error is not None:
                    # "No speech detected" is the normal outcome of opening the mic and
                    # saying nothing, so it is not worth surfacing as a failure.
                    message = str(error.localizedDescription())
                    if "No speech" not in message:
                        _log.info("recognition ended", extra={"reason": message})

            task = self._recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)

            ok, err = engine.startAndReturnError_(None)
            if not ok:
                node.removeTapOnBus_(0)
                raise ArcError(f"could not start audio engine: {err}")

            self._engine = engine
            self._request = request
            self._task = task
            self._on_result = on_result
            self._pending = ""
            self._last_voice = time.monotonic()
            self._floor = 1.0
            self._listening = True
            _log.info("listening", extra={"locale": self._locale, "on_device": self._on_device})

    def set_suspended(self, suspended: bool) -> None:
        """Gate audio into the recogniser. See ``SpeechRecognizer.set_suspended``."""
        if suspended == self._suspended:
            return
        self._suspended = suspended
        if suspended:
            # Drop whatever was mid-utterance and start clean on resume, so the first
            # words after ARC finishes are not glued onto a half-heard fragment.
            self._pending = ""
        else:
            self._last_voice = time.monotonic()
            self._restart_request()

    def _maybe_endpoint(self, level: float) -> None:
        """Decide an utterance has ended, and start a fresh one.

        Necessary because ``SFSpeechAudioBufferRecognitionRequest`` only marks a result
        final when the audio stream ends. While the microphone stays open you get an
        unbroken sequence of partials and never a final, so nothing downstream ever
        fires — observed as a perfect transcript sitting on screen with no reply.

        The threshold is **relative to a tracked noise floor**, not absolute. A fixed
        one does not survive contact with a real room: measured here, 94% of samples sat
        above 0.12 with a median of 0.306, so ``_last_voice`` was refreshed on every
        buffer, silence was never detected, and a flawless transcript sat on screen
        while nothing fired. The floor drops instantly to any quieter sample and creeps
        upward slowly, so it settles on ambient within a second or two and follows the
        room rather than a guess made at development time.

        The restart matters too — Apple accumulates across one request, so without it
        the next utterance arrives with the previous one still glued to the front.
        """
        now = time.monotonic()

        # Track the quiet level: fall to any new minimum at once, rise slowly.
        self._floor = min(level, self._floor + _FLOOR_RISE)
        speaking = level > self._floor + self._margin and level > _ABSOLUTE_FLOOR

        if speaking:
            self._last_voice = now
            return

        text = self._pending.strip()
        if not text or (now - self._last_voice) < self._silence:
            return

        handler = self._on_result
        self._pending = ""
        self._last_voice = now

        if handler is not None:
            try:
                handler(Transcript(text=text, is_final=True))
            except Exception:  # pragma: no cover
                _log.exception("result handler failed")
        self._restart_request()

    def _restart_request(self) -> None:
        """Swap in a clean request and task so the next utterance starts empty."""
        Speech = self._Speech
        old_task, old_request = self._task, self._request
        try:
            request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
            request.setShouldReportPartialResults_(True)
            request.setRequiresOnDeviceRecognition_(self._require_on_device)

            def handler(result: Any, error: Any) -> None:
                if result is not None:
                    text = str(result.bestTranscription().formattedString())
                    self._pending = text
                    if self._on_result is not None:
                        try:
                            self._on_result(Transcript(text=text, is_final=False))
                        except Exception:  # pragma: no cover
                            _log.exception("result handler failed")

            task = self._recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)
            # Publish the new pair before tearing down the old, so the audio tap never
            # sees a null request and drops buffers mid-sentence.
            self._request = request
            self._task = task
            if old_request is not None:
                old_request.endAudio()
            if old_task is not None:
                old_task.cancel()
        except Exception:  # pragma: no cover - keep listening even if the swap fails
            _log.exception("could not restart recognition request")

    def stop(self) -> None:
        with self._lock:
            if not self._listening:
                return
            self._on_result = None
            self._pending = ""
            try:
                self._engine.stop()
                self._engine.inputNode().removeTapOnBus_(0)
                self._request.endAudio()
                if self._task is not None:
                    self._task.cancel()
            except Exception:  # pragma: no cover - teardown must not raise
                _log.exception("error stopping recogniser")
            finally:
                self._engine = None
                self._request = None
                self._task = None
                self._listening = False


def pump(stop: threading.Event, interval: float = 0.1) -> None:
    """Run the main run loop until ``stop`` is set.

    Call this **from the main thread**. ``SFSpeechRecognitionTask`` delivers its
    results on the main queue, so if nothing drains that queue the recogniser is
    simply never heard from — no error, no timeout, just silence while the microphone
    happily keeps capturing. Every host that wants transcripts has to give the main
    thread over to this: ``arc voice listen`` blocks on it, and ``arc serve`` moves the
    HTTP server to a background thread so the main one is free to pump.
    """
    from Foundation import NSDate, NSRunLoop

    loop = NSRunLoop.currentRunLoop()
    while not stop.is_set():
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(interval))


def _rms(buffer: Any) -> float | None:
    """Root-mean-square amplitude of one PCM buffer, as 0..1.

    Sub-samples ~64 points rather than reading every frame: this runs on the audio
    thread for every buffer, and the result only drives a visual.
    """
    try:
        frames = int(buffer.frameLength())
        channels = buffer.floatChannelData()
        if not channels or frames <= 0:
            return None
        data = channels[0]
        step = max(1, frames // 64)
        total = 0.0
        count = 0
        for i in range(0, frames, step):
            sample = data[i]
            total += sample * sample
            count += 1
        if not count:
            return None
        # Speech sits well below full scale, so a raw RMS reads as almost nothing.
        # Map roughly -45..-5 dBFS onto 0..1, which is where conversational level lives.
        rms = math.sqrt(total / count)
        db = 20 * math.log10(rms + 1e-9)
        return max(0.0, min(1.0, (db + 45.0) / 40.0))
    except Exception:  # pragma: no cover - never let the audio thread die
        return None
