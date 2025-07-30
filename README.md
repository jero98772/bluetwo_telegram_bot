# bluetwo_telegram_bot

A Telegram bot that converts text into DTMF (Dual-Tone Multi-Frequency) audio tones and decodes DTMF audio back into text. Inspired by how telephone keypads encode information, this project demonstrates signal processing, audio synthesis, and basic NLP over a messaging platform.

---

## 🚀 Features

* 🔤 **Text to DTMF Audio**: Sends DTMF tones that represent your text.
* 🎧 **Audio to Text**: Decodes clean DTMF audio signals into readable text.
* 🛠️ **Encoding Scheme**: Mimics T9-style multi-press key input (e.g. `'c' -> '222'`).
* 🧪 `/test` command: Verifies the encoding-decoding pipeline.
* 📦 All messages are handled within Telegram via a bot.
* 🎵 WAV audio files generated using NumPy and `scipy`.

---

## 📸 Demo

![](https://raw.githubusercontent.com/jero98772/bluetwo_telegram_bot/refs/heads/main/media/1.jpeg)
### Text to DTMF

```
Input: hello
Encoded: #44#33#555#555#666#
```

<audio controls>
  <source src="https://raw.githubusercontent.com/jero98772/bluetwo_telegram_bot/refs/heads/main/media/1.acc" type="audio/wav">
  Your browser does not support the audio element.
</audio>

### DTMF to Text

```
Decoded: Hola mundo
```

---

## ⚙️ Requirements

* Python 3.8+
* [`ffmpeg`](https://ffmpeg.org/download.html) installed and in system PATH
* Python packages:

  ```bash
  pip install numpy scipy pyTelegramBotAPI
  ```

---

## 📦 Installation

```bash
git clone https://github.com/jero98772/bluetwo_telegram_bot.git
cd bluetwo_telegram_bot
python main.py
```

Update the bot token at the top of `main.py`:

```python
BOT_TOKEN = "your_bot_token_here"
```

---

## 🧠 How It Works

1. **Text Input**
   The bot encodes text into multi-press digit sequences (e.g., `a -> 2`, `c -> 222`).

2. **DTMF Audio**
   Each symbol is mapped to a DTMF tone pair (e.g., `2 -> 697Hz + 1336Hz`) using sine waves.

3. **Decoding Audio**
   Audio is segmented by energy levels, then FFT is applied to detect tone pairs. These are mapped back to keys and decoded into original text.

---

## 🧪 Usage

### Telegram Commands

* `/start` or `/help`: See usage instructions.
* `/test`: Run an internal test with sample input `"hello"`.

### Interactions

* 📝 Send a **text message**: receive a `.wav` file with the encoded DTMF tones.
* 🎧 Send a **voice or audio message**: bot will decode the DTMF tones into text.

---


