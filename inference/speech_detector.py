from collections import deque

import numpy as np

from common.audio import rms_energy


class SpeechDetector:
    def __init__(
        self,
        energy_threshold=0.01,
        start_chunks=2,
        end_silence_chunks=6,
        min_speech_chunks=3,
        max_chunks=80,
        pre_roll_chunks=2,
    ):
        self.energy_threshold = energy_threshold
        self.start_chunks = start_chunks
        self.end_silence_chunks = end_silence_chunks
        self.min_speech_chunks = min_speech_chunks
        self.max_chunks = max_chunks
        self.pre_roll_chunks = pre_roll_chunks

        self.reset()

    def reset(self):
        self.in_speech = False
        self.speech_chunks = []
        self.pre_roll = deque(maxlen=self.pre_roll_chunks + self.start_chunks)
        self.loud_count = 0
        self.silent_count = 0

    def process_chunk(self, chunk):
        # get chunk
        # calculate energy . loud or not
        # if speech already found or not.
        # if not found so far --> add to preroll , keep track of loud count
        ## loud count > start_chunks (threshold, min chunks needed to confirm speech)
        ## means found speech , add preroll to speech
        ## next time speech is found , directly add chunk
        ## now total chunks processed size??
        ## check silent count as well
        ## if silent count ==> threshold done ..
        ## or size reached of total chunks
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)

        loud = rms_energy(chunk) >= self.energy_threshold

        if not self.in_speech:
            self.pre_roll.append(chunk)

            self.loud_count = self.loud_count + 1 if loud else 0

            if self.loud_count >= self.start_chunks:
                self.in_speech = True
                self.speech_chunks = list(self.pre_roll)
                self.silent_count = 0

            return None

        self.speech_chunks.append(chunk)

        self.silent_count = 0 if loud else self.silent_count + 1

        if len(self.speech_chunks) >= self.max_chunks or self.silent_count >= self.end_silence_chunks:
            result = np.concatenate(self.speech_chunks)

            valid = len(self.speech_chunks) >= self.min_speech_chunks

            self.reset()

            return result if valid else None

        return None
