"""Telling people apart by their VOICE (speaker identification, on the laptop).

Each utterance is turned into a voiceprint (a fixed-length vector). The VoiceBank
compares it with the voices heard so far: close enough to a known voice -> same
person, otherwise a new voice ("V1", "V2", ...). Voiceprints are only kept in
memory for the current interaction and never leave the laptop.

Two embedders:
  * Resemblyzer (neural, much more accurate)  pip install resemblyzer   (pulls in torch)
  * MFCC statistics (numpy/scipy only)        built in, used when Resemblyzer is missing.
    Good enough for 2-4 clearly different voices in a quiet room; similar voices
    (e.g. two young men) will be confused more often.

Voices are linked to the PEOPLE the camera sees by `SpeakerLinker`:
  * only one person present                  -> the voice is theirs
  * the robot just invited person X          -> the reply's voice is X's
  * all but one present person already have a voice -> a new voice is the remaining one
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("nuru.voices")
SR = 16000


# ----------------------------------------------------------------- embedders
def _mel_filterbank(n_mels=40, n_fft=512, sr=SR, fmin=60.0, fmax=7600.0):
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        if b > a:
            fb[i - 1, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i - 1, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


_FB = _mel_filterbank()


def mfcc_embedding(audio: np.ndarray, sr: int = SR, n_mfcc: int = 20) -> np.ndarray | None:
    """Mean + std of MFCCs over voiced frames, then normalised. ~40-dim vector."""
    from scipy.fft import dct
    x = np.asarray(audio, np.float32).reshape(-1)
    if len(x) < sr * 0.4:
        return None
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])                     # pre-emphasis
    flen, hop = 400, 160                                             # 25 ms / 10 ms
    n = 1 + (len(x) - flen) // hop
    frames = np.lib.stride_tricks.as_strided(x, (n, flen), (x.strides[0] * hop, x.strides[0])) * np.hamming(flen)
    energy = (frames ** 2).mean(axis=1)
    voiced = frames[energy > np.percentile(energy, 40)]              # skip pauses between words
    if len(voiced) < 20:
        return None
    spec = np.abs(np.fft.rfft(voiced, 512)) ** 2
    mel = np.log(spec @ _FB.T + 1e-10)
    cep = dct(mel, type=2, axis=1, norm="ortho")[:, 1:n_mfcc + 1]   # drop c0 (loudness)
    v = np.concatenate([cep.mean(axis=0), cep.std(axis=0)])
    return v / (np.linalg.norm(v) + 1e-9)


class Embedder:
    """Resemblyzer if installed, otherwise MFCC statistics."""

    def __init__(self, prefer_neural: bool = True):
        self.kind = "mfcc"
        self._enc = None
        if prefer_neural:
            try:
                from resemblyzer import VoiceEncoder
                self._enc = VoiceEncoder(verbose=False)
                self.kind = "resemblyzer"
            except Exception as e:   # not installed / no torch: fall back quietly
                log.info("Resemblyzer not available (%s) - using MFCC voiceprints", e)

    @property
    def threshold(self) -> float:
        """Cosine similarity above which two voiceprints count as the same person."""
        return 0.75 if self.kind == "resemblyzer" else 0.93

    def __call__(self, audio: np.ndarray, sr: int = SR) -> np.ndarray | None:
        if self._enc is not None:
            if len(audio) < sr * 0.5:
                return None
            from resemblyzer import preprocess_wav
            return self._enc.embed_utterance(preprocess_wav(np.asarray(audio, np.float32).reshape(-1), sr))
        return mfcc_embedding(audio, sr)


# ----------------------------------------------------------------- the bank
class VoiceBank:
    """Online clustering of voiceprints into voices V1, V2, ..."""

    def __init__(self, threshold: float, max_voices: int = 8):
        self.threshold = threshold
        self.max_voices = max_voices
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}

    def reset(self):
        self.centroids.clear()
        self.counts.clear()

    def identify(self, emb: np.ndarray | None) -> tuple[str | None, float]:
        """-> (voice id, similarity). Unknown voice -> a new id. None embedding -> (None, 0)."""
        if emb is None:
            return None, 0.0
        best, sim = None, -1.0
        for vid, c in self.centroids.items():
            s = float(np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c) + 1e-9))
            if s > sim:
                best, sim = vid, s
        if best is not None and (sim >= self.threshold or len(self.centroids) >= self.max_voices):
            n = self.counts[best]
            self.centroids[best] = (self.centroids[best] * n + emb) / (n + 1)   # running mean
            self.counts[best] = n + 1
            return best, sim
        vid = f"V{len(self.centroids) + 1}"
        self.centroids[vid], self.counts[vid] = np.asarray(emb, float), 1
        return vid, sim


class SpeakerLinker:
    """Which camera person (face id) owns which voice."""

    def __init__(self):
        self.voice_to_person: dict[str, str] = {}

    def reset(self):
        self.voice_to_person.clear()

    def person_for(self, voice: str | None) -> str | None:
        return self.voice_to_person.get(voice) if voice else None

    def link(self, voice: str | None, present: list[str], invited: str | None = None) -> tuple[str | None, str]:
        """Attribute an utterance with this voice to a present person.
        -> (person id or None, how it was decided)."""
        if not voice:
            return None, "no voiceprint"
        known = self.voice_to_person.get(voice)
        if known in present:
            return known, "known voice"
        if len(present) == 1:
            self.voice_to_person[voice] = present[0]
            return present[0], "only person present"
        if invited in present and invited not in self.voice_to_person.values():
            self.voice_to_person[voice] = invited
            return invited, "answered an invitation"
        unlinked = [p for p in present if p not in self.voice_to_person.values()]
        if len(unlinked) == 1:
            self.voice_to_person[voice] = unlinked[0]
            return unlinked[0], "only person without a voice yet"
        return None, "not yet linked to a face"
