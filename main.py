import numpy as np
from scipy.io.wavfile import write, read
from scipy.io import wavfile
import io
import os
import tempfile
import telebot
from telebot import types
import subprocess

# Your bot token here
BOT_TOKEN = "your token here line 11"
bot = telebot.TeleBot(BOT_TOKEN)

letras = "abcdefghijklmnopqrstuvwxyz"
FS = 22050
numcode = ['22', '222', '2222', '33', '333', '3333', '44', '444', '4444', '55', '555', '5555', '66', '666', '6666', '77', '777', '7777', '77777', '88', '888', '8888', '99', '999', '9999', '99999']

def encpalabranum(palabra, sep="#", space="*"):
    """Encode text to DTMF number sequence"""
    palabraenc = ""
    for i in palabra:
        if i.isdigit():
            palabraenc += str(int(i))
            palabraenc += sep
        else:
            for ii in range(len(numcode)):
                if i == letras[ii]:
                    palabraenc += numcode[ii]
                    palabraenc += sep
                    break
            if i == " ":
                palabraenc += space
                palabraenc += sep
    return sep + palabraenc

def decpalabranum(palbraenc, sep="#", space="*"):
    """Decode DTMF number sequence to text"""
    palabra = ""
    caracter = ""
    
    # Remove leading separator if present
    if palbraenc.startswith(sep):
        palbraenc = palbraenc[1:]
    
    # Add trailing separator if not present
    if not palbraenc.endswith(sep):
        palbraenc += sep
    
    for i in palbraenc:
        if i == sep:
            if caracter == space:
                palabra += " "
            elif caracter in numcode:
                palabra += letras[numcode.index(caracter)]
            elif caracter.isdigit() and len(caracter) == 1:
                palabra += caracter
            caracter = ""
        else:
            caracter += i
    return palabra

def dtmf_dial(number):
    """Generate DTMF audio for given number sequence"""
    DTMF = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477),        
    }
    
    # Increased timing for better recognition
    MARK = 0.15   # Tone duration
    SPACE = 0.08  # Silence between tones
    
    n = np.arange(0, int(MARK * FS))
    x = np.array([])
    
    for d in number:
        if d in DTMF:
            # Generate dual-tone signal with proper amplitude
            s1 = 0.5 * np.sin(2*np.pi * DTMF[d][0] / FS * n)
            s2 = 0.5 * np.sin(2*np.pi * DTMF[d][1] / FS * n)
            s = s1 + s2
            
            # Add the tone and silence
            x = np.concatenate((x, s, np.zeros(int(SPACE * FS))))
    
    # Normalize audio to prevent clipping
    if len(x) > 0:
        x = x / np.max(np.abs(x)) * 0.8
    
    return x

def dtmf_split(x, win=480, th=0.01):
    """Split audio into tone segments with improved detection"""
    edges = []
    
    # Ensure we have enough samples
    if len(x) < win:
        return edges
    
    # Reshape into windows and calculate energy
    num_windows = int(len(x) / win)
    w = np.reshape(x[:num_windows * win], (-1, win))
    we = np.sum(w * w, axis=1)
    
    # Normalize energy threshold
    max_energy = np.max(we)
    if max_energy > 0:
        th = th * max_energy
    
    L = len(we)
    ix = 0
    
    while ix < L:
        # Skip silence
        while ix < L and we[ix] < th:
            ix += 1
        if ix >= L:
            break
        
        # Find end of tone
        iy = ix
        while iy < L and we[iy] > th:
            iy += 1
        
        # Only add if tone is long enough (at least 50ms)
        tone_length = (iy - ix) * win
        if tone_length >= int(0.05 * FS):  # 50ms minimum
            edges.append((ix * win, iy * win))
        
        ix = iy
    
    return edges

def dtmf_decode(x, edges=None):
    """Decode DTMF audio to number sequence with improved accuracy"""
    # DTMF frequencies
    LO_FREQS = np.array([697.0, 770.0, 852.0, 941.0])
    HI_FREQS = np.array([1209.0, 1336.0, 1477.0])
    
    KEYS = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9'], ['*', '0', '#']]
    
    # Tighter frequency ranges
    LO_RANGE = (690.0, 950.0)
    HI_RANGE = (1200.0, 1490.0)
    
    number = []
    
    if edges is None:
        edges = dtmf_split(x)
    
    print(f"Found {len(edges)} tone segments")  # Debug
    
    for i, g in enumerate(edges):
        # Get tone segment
        segment = x[g[0]:g[1]]
        if len(segment) < 100:  # Skip very short segments
            continue
        
        # Apply window to reduce spectral leakage
        window = np.hanning(len(segment))
        segment = segment * window
        
        # Compute FFT with zero padding for better frequency resolution
        N = max(2048, len(segment))
        X = np.abs(np.fft.fft(segment, N))
        
        # Frequency resolution
        res = float(FS) / N
        
        # Find peaks in low and high frequency ranges
        a_lo = int(LO_RANGE[0] / res)
        b_lo = int(LO_RANGE[1] / res)
        lo_peak = a_lo + np.argmax(X[a_lo:b_lo])
        lo_freq = lo_peak * res
        
        a_hi = int(HI_RANGE[0] / res)
        b_hi = int(HI_RANGE[1] / res)
        hi_peak = a_hi + np.argmax(X[a_hi:b_hi])
        hi_freq = hi_peak * res
        
        # Check if peaks are strong enough
        lo_power = X[lo_peak]
        hi_power = X[hi_peak]
        noise_floor = np.mean(X)
        
        if lo_power < 5 * noise_floor or hi_power < 5 * noise_floor:
            print(f"Segment {i}: Weak signal, skipping")
            continue
        
        # Match to DTMF frequencies with tolerance
        row = np.argmin(np.abs(LO_FREQS - lo_freq))
        col = np.argmin(np.abs(HI_FREQS - hi_freq))
        
        # Verify the match is close enough (within 30 Hz)
        if (abs(LO_FREQS[row] - lo_freq) < 30 and 
            abs(HI_FREQS[col] - hi_freq) < 30):
            detected_key = KEYS[row][col]
            number.append(detected_key)
            print(f"Segment {i}: {lo_freq:.1f}Hz + {hi_freq:.1f}Hz = '{detected_key}'")
        else:
            print(f"Segment {i}: No match for {lo_freq:.1f}Hz + {hi_freq:.1f}Hz")
    
    return number

def convert_audio_to_wav(input_file, output_file):
    """Convert audio file to WAV using ffmpeg"""
    cmd = [
        'ffmpeg', '-i', input_file, 
        '-ar', str(FS),  # Set sample rate
        '-ac', '1',      # Convert to mono
        '-af', 'volume=1.0',  # Normalize volume
        '-y',            # Overwrite output file
        output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def text_to_audio(text):
    """Convert text to DTMF audio file"""
    # Clean and limit input text
    text = text.strip()[:50]  # Limit length to prevent very long audio
    
    encoded_text = encpalabranum(text.lower())
    print(f"Text: '{text}' -> Encoded: '{encoded_text}'")
    
    audio_data = dtmf_dial(encoded_text)
    
    if len(audio_data) == 0:
        raise ValueError("No audio generated")
    
    # Save as 16-bit WAV
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio_16bit = (audio_data * 32767).astype(np.int16)
    write(temp_file.name, FS, audio_16bit)
    temp_file.close()
    
    print(f"Audio saved: {len(audio_data)} samples, {len(audio_data)/FS:.2f}s")
    return temp_file.name

def audio_to_text(audio_file_path):
    """Convert DTMF audio file to text"""
    try:
        # Read audio file
        rate, audio_data = wavfile.read(audio_file_path)
        print(f"Audio loaded: {rate}Hz, {len(audio_data)} samples, {audio_data.dtype}")
        
        # Convert to float and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Handle stereo by taking first channel
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        print(f"Audio stats: min={np.min(audio_data):.3f}, max={np.max(audio_data):.3f}")
        
        # Decode DTMF
        decoded_numbers = dtmf_decode(audio_data)
        print(f"Decoded DTMF: {decoded_numbers}")
        
        if not decoded_numbers:
            return ""
        
        # Convert to text
        number_string = ''.join(decoded_numbers)
        decoded_text = decpalabranum(number_string)
        print(f"Final text: '{decoded_text}'")
        
        return decoded_text
        
    except Exception as e:
        print(f"Error in audio_to_text: {e}")
        return ""

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🎵 Welcome to DTMF Audio Bot! 🎵

This bot can:
📝 Convert your text messages to DTMF audio
🎧 Convert DTMF audio back to text

How to use:
• Send me any text message and I'll convert it to audio
• Send me an audio file and I'll decode it back to text

Features:
• Supports letters, numbers, and spaces
• Maximum 50 characters per message
• High-quality DTMF encoding/decoding

Just type something or send an audio file to get started!
"""
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['test'])
def test_encoding(message):
    """Test command to verify encoding/decoding works"""
    test_text = "hello"
    bot.reply_to(message, f"🧪 Testing with: '{test_text}'")
    
    try:
        # Test encoding
        encoded = encpalabranum(test_text)
        bot.reply_to(message, f"Encoded: {encoded}")
        
        # Test decoding
        decoded = decpalabranum(encoded)
        bot.reply_to(message, f"Decoded: '{decoded}'")
        
        if decoded == test_text:
            bot.reply_to(message, "✅ Encoding/decoding test passed!")
        else:
            bot.reply_to(message, "❌ Encoding/decoding test failed!")
            
    except Exception as e:
        bot.reply_to(message, f"❌ Test error: {e}")

@bot.message_handler(content_types=['text'])
def handle_text(message):
    text = message.text
    
    # Skip commands
    if text.startswith('/'):
        return
    
    # Validate input
    if len(text.strip()) == 0:
        bot.reply_to(message, "Please send a non-empty message.")
        return
    
    if len(text) > 50:
        bot.reply_to(message, "Message too long! Please limit to 50 characters.")
        return
    
    bot.reply_to(message, f"🔄 Converting text to audio: '{text}'")
    
    try:
        # Generate audio from text
        audio_file_path = text_to_audio(text)
        
        # Send audio file
        with open(audio_file_path, 'rb') as audio_file:
            bot.send_audio(
                message.chat.id, 
                audio_file,
                caption=f"🎵 DTMF audio for: '{text}'",
                title="DTMF Audio"
            )
        
        # Clean up temporary file
        os.unlink(audio_file_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Error generating audio: {e}")

@bot.message_handler(content_types=['audio', 'voice'])
def handle_audio(message):
    bot.reply_to(message, "🎧 Processing your audio...")
    
    try:
        # Get file info
        if message.content_type == 'voice':
            file_info = bot.get_file(message.voice.file_id)
        else:
            file_info = bot.get_file(message.audio.file_id)
        
        # Download the file
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Save to temporary file
        file_extension = os.path.splitext(file_info.file_path)[1] or '.ogg'
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_input.write(downloaded_file)
        temp_input.close()
        
        # Convert to WAV using ffmpeg
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav.close()
        
        if convert_audio_to_wav(temp_input.name, temp_wav.name):
            print(f"Processing audio file: {temp_wav.name}")
            
            # Decode audio to text
            decoded_text = audio_to_text(temp_wav.name)
            
            if decoded_text.strip():
                bot.reply_to(message, f"📝 Decoded text: '{decoded_text}'")
            else:
                bot.reply_to(message, "❌ Could not decode the audio. Make sure it contains clear DTMF tones.")
        else:
            bot.reply_to(message, "❌ Could not process audio file. Make sure ffmpeg is installed.")
    
    except Exception as e:
        bot.reply_to(message, f"❌ Error processing audio: {e}")
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_input.name)
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
        except:
            pass

@bot.message_handler(content_types=['document'])
def handle_document(message):
    # Check if it's an audio file
    if message.document.mime_type and message.document.mime_type.startswith('audio/'):
        bot.reply_to(message, "🎧 Processing your audio file...")
        
        try:
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            
            file_extension = os.path.splitext(message.document.file_name)[1] or '.wav'
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_input.write(downloaded_file)
            temp_input.close()
            
            # Convert to proper WAV format
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav.close()
            
            if convert_audio_to_wav(temp_input.name, temp_wav.name):
                decoded_text = audio_to_text(temp_wav.name)
                
                if decoded_text.strip():
                    bot.reply_to(message, f"📝 Decoded text: '{decoded_text}'")
                else:
                    bot.reply_to(message, "❌ Could not decode the audio. Make sure it contains clear DTMF tones.")
            else:
                bot.reply_to(message, "❌ Could not process audio file. Make sure ffmpeg is installed.")
        
        except Exception as e:
            bot.reply_to(message, f"❌ Error processing document: {e}")
        
        finally:
            try:
                os.unlink(temp_input.name)
                if os.path.exists(temp_wav.name):
                    os.unlink(temp_wav.name)
            except:
                pass
    else:
        bot.reply_to(message, "Please send an audio file or text message.")

if __name__ == '__main__':
    print("🤖 DTMF Bot is starting...")
    print("Make sure to set your BOT_TOKEN!")
    print("Install required packages: pip install pyTelegramBotAPI numpy scipy")
    print("Make sure ffmpeg is installed on your system")
    print("\nTo test encoding/decoding, use /test command")
    bot.infinity_polling()