# Speaking Piano

Play a piano by speaking the note you want to hear.

Speaking Piano is an experiment in building a small, local speech-driven piano system. Instead of pressing keys on a virtual 88-key piano, the user speaks a note such as:

```text
C four
```

The system listens to the microphone, recognizes the individual components of the note, builds `C4`, and sends it to a piano synthesizer.

```text
Speak "C four"
       ↓
Microphone
       ↓
Speech detection
       ↓
Audio segmentation
       ↓
Letter classifier → C
Digit classifier  → 4
       ↓
Build note → C4
       ↓
Piano player
       ↓
Speaker
```

The interesting part is that the project does not depend on a general speech-to-text model to recognize the piano notes. The note recognition system is built using custom CNN classifiers trained specifically for this problem.

---

## The original idea

The project started with a simple goal: create a piano that could run on a laptop.

A normal piano has 88 keys. On a physical piano, that is not a problem. On a laptop keyboard or a small virtual interface, however, playing notes becomes awkward. Mapping all piano keys to keyboard shortcuts is possible, but remembering and pressing them is not particularly enjoyable.

So the question became:

> What if the piano could be played by speaking?

The first idea was straightforward:

```text
Speak → Speech recognition model → Text → Piano note
```

For example:

```text
"C four"
    ↓
C4
    ↓
Play C4
```

That works in principle, but single-letter recognition turned out to be a problem.

General speech recognition systems are designed to understand words and sentences. A spoken sequence such as:

```text
A
B
C
D
E
F
G
```

is a much smaller and more specialized problem. Individual letters are short, acoustically similar, and can easily be confused depending on pronunciation, microphone quality, background noise, and the surrounding silence.

For this project, recognizing `A`, `B`, or `C` correctly was more important than recognizing arbitrary English sentences.

That led to a different approach.

---

# The approach

Instead of asking a general speech model to solve a very specific classification problem, the project uses custom audio classifiers.

The piano note is split into smaller pieces.

For example:

```text
C4
```

is recognized as:

```text
"C" + "4"
```

An accidental can also be added:

```text
C + sharp + 4
```

to produce:

```text
C#4
```

The recognition flow therefore became:

```text
Letter → Octave → Optional accidental → Piano note
```

For example:

```text
User says:

C
    ↓
Letter classifier
    ↓
C

User says:

four
    ↓
Digit classifier
    ↓
4

Combine:

C + 4
    ↓
C4
```

This makes the problem much smaller and more controlled.

Instead of one model trying to understand all possible piano note combinations, separate classifiers handle a limited number of classes.

### Letter classifier

```text
A
B
C
D
E
F
G
```

### Digit classifier

```text
0
1
2
3
4
5
6
7
8
```

### Accidental classifier

```text
sharp
flat
```

The resulting pieces are combined into a canonical piano note.

---

# Building the dataset

The first requirement was audio data.

To train a classifier for spoken letters, the audio first needs to be split into individual samples.

Raw recordings are inconvenient for this. A recording may contain multiple utterances:

```text
A ... B ... C ... D ...
```

The training pipeline needs separate files:

```text
A.wav
B.wav
C.wav
D.wav
```

That led to a separate utility project:

[speech-slicer](https://github.com/jdecodes/speech-slicer?utm_source=chatgpt.com)

The speech slicer takes an audio recording and separates it into individual speech segments using silence and audio energy.

Conceptually:

```text
Long recording

[A] ... [B] ... [C] ... [D]

        ↓

Speech slicer

        ↓

A.wav
B.wav
C.wav
D.wav
```

This made it easier to record multiple examples and prepare them for the classifier.

The dataset consists of limited recordings of the required spoken classes rather than a large general-purpose speech corpus.

The goal was not to build a universal speech recognition system. The goal was to recognize a small vocabulary well enough to control a piano.

---

# Training the classifiers

Separate CNN models were trained for the different categories.

The project contains models for:

```text
Letters: A-G
Digits: 0-8
Accidentals: sharp / flat
```

Each audio sample goes through preprocessing before it reaches the model.

The overall training pipeline looks like this:

```text
Audio file
    ↓
Load audio
    ↓
Convert to expected sample rate
    ↓
Fix audio length
    ↓
Extract audio features
    ↓
CNN
    ↓
Class probabilities
    ↓
Predicted class
```

For example:

```text
Audio

~~~~~ spoken "C" ~~~~~

        ↓

Audio features

┌─────────────────────┐
│ frequency over time │
│                     │
│ ███████             │
│   █████████         │
│       ███████       │
└─────────────────────┘

        ↓

CNN

        ↓

C → 0.97
D → 0.01
E → 0.01
B → 0.01
```

The highest-confidence prediction becomes the recognized label.

---

# The live recognition problem

Training a model is only one part of the system.

During live recognition, the microphone continuously produces audio. The classifier cannot simply receive the entire microphone stream.

The system first needs to determine:

1. When speech starts.
2. Which audio belongs to the utterance.
3. When speech ends.
4. When to send the collected audio to the classifier.

This is handled by `SpeechDetector`.

The microphone produces fixed-size audio chunks:

```text
chunk 1
chunk 2
chunk 3
chunk 4
...
```

For each chunk, the detector calculates its RMS energy.

```text
RMS Energy
     │
     │          ███████████
     │       ███████████████
─────┼──────────────────────── threshold
     │
     └──────────────────────────── Time
```

If the energy is above the configured threshold, the chunk is considered loud.

The detector has two main states.

## Waiting for speech

Initially, the detector is waiting.

```text
WAITING
```

Each incoming chunk is checked:

```text
chunk
   ↓
calculate energy
   ↓
loud or silent?
```

The detector requires several consecutive loud chunks before confirming that speech has actually started.

This prevents a single noise spike from being treated as speech.

```text
noise       loud  loud  loud
  │           │     │     │
  ▼           ▼     ▼     ▼

silent ──── loud ─ loud ─ loud

                  ↓
           enough loud chunks

                  ↓

           SPEECH DETECTED
```

---

# The beginning of the audio problem

One of the issues discovered during testing was that some spoken letters appeared to be missing their beginning.

This was especially noticeable for letters whose initial sound had lower energy.

For example, the beginning of a spoken `C` might be quieter than the main part of the sound.

```text
Actual audio:

quiet beginning       louder speech

    ░░░░░              ███████████
  ░░░░░░░░          ███████████████
────────────────────────────────────
```

The speech detector waits for a number of consecutive loud chunks before confirming speech.

Suppose:

```text
start_chunks = 3
```

The detector may see:

```text
quiet     quiet     loud     loud     loud
  0         0         1        2        3
                                      ↑
                              speech confirmed
```

By the time speech is confirmed, the first quiet chunks may already have passed.

If those chunks are discarded, the classifier receives a truncated version of the letter.

That is a problem because the model may have been trained on complete utterances.

---

# Pre-roll buffering

To avoid losing the beginning of an utterance, the detector keeps a rolling buffer of recent audio chunks.

```text
Incoming audio

chunk 1 → buffer
chunk 2 → buffer
chunk 3 → buffer
chunk 4 → buffer
...
```

When speech is finally confirmed, the recent buffered chunks are included with the speech audio.

```text
Rolling pre-roll buffer
        │
        ▼

[quiet beginning][loud speech]
        │              │
        └──────┬───────┘
               ↓
        Complete utterance
```

The pre-roll buffer was increased to eight chunks.

The maximum speech window is eighty chunks, so the pre-roll buffer represents approximately ten percent of the maximum window.

```text
|-- pre-roll --|---------------- speech window ----------------|

     8 chunks                     up to 80 chunks
```

After increasing the pre-roll buffer, recognition accuracy improved significantly during testing.

The important point was not simply that the classifier needed to be changed. The audio reaching the classifier had changed.

The pipeline had to preserve the actual beginning of the spoken sound.

---

# The speech detector

The live detector works approximately like this:

```text
Get audio chunk
       ↓
Calculate RMS energy
       ↓
Is speech already active?
       │
       ├── No
       │     ↓
       │  Add chunk to pre-roll buffer
       │     ↓
       │  Update consecutive loud count
       │     ↓
       │  loud count >= start_chunks?
       │     │
       │     ├── No → keep waiting
       │     │
       │     └── Yes
       │            ↓
       │       Speech confirmed
       │            ↓
       │       Copy pre-roll into speech buffer
       │
       └── Yes
             ↓
          Add chunk to speech buffer
             ↓
          Update silent count
             ↓
          Enough silence?
             │
             ├── No
             │
             └── Yes
                   ↓
              Speech finished
                   ↓
              Combine chunks
                   ↓
              Send audio to classifier
```

Speech can also finish when the maximum configured number of chunks is reached.

This prevents the detector from recording indefinitely.

---

# Sequential note recognition

The system does not try to recognize the entire note phrase at once.

Instead, it runs a small recognition session.

```text
Speak a letter: A-G
        ↓
Detect speech
        ↓
Classify audio
        ↓
Accepted?
        ↓
Return letter

        ↓

Speak an octave: 0-8
        ↓
Detect speech
        ↓
Classify audio
        ↓
Accepted?
        ↓
Return digit

        ↓

Build final note
```

For example:

```text
Speak: C
        ↓
Letter model
        ↓
C

Speak: four
        ↓
Digit model
        ↓
4

        ↓

C4
```

The final result is passed to a callback:

```python
self.play_callback(note)
```

This creates a clean boundary between recognition and playback.

The recognition system does not need to know how piano audio is generated.

It only produces:

```text
C4
D#5
A3
```

What happens to the note after that is the responsibility of the piano player.

---

# Debugging with WAV dumps

Live audio debugging is difficult if the original audio disappears immediately after classification.

To make debugging easier, recognized utterances can be saved as WAV files.

A debug file contains information about the prediction:

```text
000001_C_0.98_accepted.wav
000002_B_0.42_rejected.wav
000003_4_0.91_accepted.wav
```

The filename includes:

```text
counter
prediction
confidence
accepted/rejected status
```

This makes it possible to inspect what the classifier actually received.

For example:

```text
Classifier says:

C → 0.42 → rejected

        ↓

Open:

000002_C_0.42_rejected.wav

        ↓

Listen to actual input
```

The WAV dumps were particularly useful for discovering the truncated speech issue.

Without them, a low-confidence result could look like a model problem.

After listening to the actual audio, it became possible to see that the model was sometimes receiving incomplete speech.

---

# Integrating the piano player

Once the system produces a canonical note such as:

```text
C4
```

it is sent to `PianoPlayer`.

The connection is made through the callback:

```text
SequentialNoteSession
        │
        │ play_callback("C4")
        ▼
PianoPlayer.play_note("C4")
        │
        ▼
Note Queue
        │
        ▼
Audio Worker Thread
        │
        ▼
Speaker
```

The piano player runs independently from the recognition loop.

When a note is received:

```python
piano_player.play_note("C4")
```

the note is placed into a thread-safe queue.

```text
Recognition thread
       │
       │ C4
       ▼
   Note Queue
       │
       ▼
Audio thread
```

The audio worker reads notes from the queue and plays them.

This allows recognition and audio playback to operate independently.

---

# Piano sound generation

The project currently generates a piano-like sound using additive synthesis.

The frequency of a note is calculated from its MIDI number.

For example:

```text
A4 = MIDI 69 = 440 Hz
```

Other notes are calculated using the equal-temperament relationship:

```text
frequency = 440 × 2 ^ ((midi - 69) / 12)
```

For example:

```text
C4
    ↓
MIDI note
    ↓
Frequency
    ↓
Generate waveform
```

The generated signal combines multiple harmonics:

```text
fundamental
2nd harmonic
3rd harmonic
4th harmonic
...
```

Conceptually:

```text
Fundamental:

~~~~~~~

2nd harmonic:

~~~~~~~~~~~~~~

3rd harmonic:

~~~~~~~~~~~~~~~~~~~~~

        ↓

Combine

        ↓

Piano-like waveform
```

An attack and exponential decay envelope are then applied to make the sound behave more like a struck piano note.

```text
Volume

1.0 ──┐
      │\
      │ \
      │  \
      │   \
0.0 ──┴────\──────── Time
     attack   decay
```

Generated notes are cached.

The first time a note is requested:

```text
C4
 ↓
Generate audio
 ↓
Store in cache
```

The next time:

```text
C4
 ↓
Load from cache
 ↓
Play immediately
```

---

# Architecture

The current architecture is:

```text
┌──────────────────────┐
│      Microphone      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    AudioRecorder     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   SpeechDetector     │
│                      │
│ • RMS energy         │
│ • loud detection     │
│ • pre-roll buffer    │
│ • silence detection  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   AudioClassifier    │
│                      │
│ • Letters CNN        │
│ • Digits CNN         │
│ • Accidentals CNN    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ SequentialNoteSession│
│                      │
│ Letter + Octave      │
│ + Optional Accidental│
└──────────┬───────────┘
           │
           │ "C4"
           ▼
┌──────────────────────┐
│     PianoPlayer      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Note Queue      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Audio Worker Thread │
│                      │
│ • Generate/load note │
│ • Mix active notes   │
│ • Prevent clipping   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│       Speaker        │
└──────────────────────┘
```

---

# Project structure

The project is organized around the main stages of the pipeline.

```text
piano-cnn/
│
├── audio/
│   ├── recorder.py
│   └── ...
│
├── common/
│   ├── audio.py
│   └── ...
│
├── data/
│   ├── letters/
│   ├── digits/
│   └── accidentals/
│
├── generate_models/
│   ├── train_letters.py
│   ├── train_digits.py
│   └── ...
│
├── inference/
│   └── speech_detector.py
│
├── metadata/
│   ├── letters_metadata.json
│   ├── digits_metadata.json
│   └── ...
│
├── models/
│   ├── letters_cnn.pt
│   ├── digits_cnn.pt
│   └── accidentals_cnn.pt
│
├── recognition/
│   └── audio_classifier.py
│
├── training/
│   └── ...
│
├── classifier.py
├── piano_player.py
├── test_seq.py 
└── requirements.txt
```

---

# Running the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the speaking piano:

```bash
python .\test_seq.py 
```

The application keeps the microphone active and guides the recognition sequence:

```text
Sequential note recognizer started.
Microphone is active.

Speak a letter: A to G
Detected letter: C

Speak the octave: 0 to 8
Detected octave: 4

FINAL NOTE: C4

Playing: C4
```

---

# Example session

```text
Sequential note recognizer started.

Speak a letter: A to G

letter: C (0.982) OK

Detected letter: C

Speak the octave: 0 to 8

octave: 4 (0.943) OK

Detected octave: 4

FINAL NOTE: C4

Queued piano note: C4

Generating piano note: C4

Playing: C4
```

The same flow can be repeated for another note:

```text
Speak a letter: A to G
→ G

Speak the octave: 0 to 8
→ 5

FINAL NOTE: G5
```

---

# Current result

The final system can:

* Record audio continuously from the microphone.
* Detect when the user starts speaking.
* Preserve audio before speech confirmation using a pre-roll buffer.
* Detect when speech ends using consecutive silent chunks.
* Classify spoken letters from `A` to `G`.
* Classify spoken octaves from `0` to `8`.
* Build a valid piano note such as `C4`.
* Validate notes against the standard piano range `A0-C8`.
* Send recognized notes through a callback.
* Queue notes for playback.
* Generate and cache piano-like audio.
* Play the generated note through the speaker.
* Save recognized audio as WAV files for debugging.
* Record prediction confidence and acceptance status in debug filenames.
