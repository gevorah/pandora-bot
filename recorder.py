import pyaudio
import wave

FORMAT = pyaudio.paInt16  # Format of audio samples (16-bit signed integers)
CHANNELS = 2              # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100              # Sample rate (samples per second)
CHUNK = 1024              # Number of frames per buffer
SECONDS = 6               # Duration of recording in seconds

def record(filename):
  p = pyaudio.PyAudio()

  stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  frames_per_buffer=CHUNK,
                  input=True)
    
  print('Recording...')

  frames = []
    
  # Store data in chunks
  for i in range(0, int(RATE / CHUNK * SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

  stream.stop_stream()
  stream.close()

  p.terminate()
    
  print('Recording finished.')
    
  # Save the recorded data as a WAV file
  wf = wave.open(filename, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()
