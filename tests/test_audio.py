import numpy as np
import wave
import subprocess
import imageio_ffmpeg

from music_genre.audio import N_MELS, SAMPLE_RATE, WINDOW_FRAMES, crop_window, load_audio, log_mel, prediction_windows, probe_duration


def test_spectrogram_and_windows_have_model_shape():
    seconds = 3
    time = np.arange(SAMPLE_RATE * seconds) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    spectrogram = log_mel(wave)
    windows = prediction_windows(spectrogram)
    assert spectrogram.shape[0] == N_MELS
    assert windows.shape == (10, N_MELS, WINDOW_FRAMES)
    assert np.isfinite(windows).all()
    assert crop_window(spectrogram, 0).shape == (N_MELS, WINDOW_FRAMES)


def test_wave_decode_and_duration(tmp_path):
    path = tmp_path / "tone.wav"
    time = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    samples = (np.sin(2 * np.pi * 440 * time) * 15000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())
    assert abs(probe_duration(path) - 4) < 0.01
    decoded = load_audio(path, start=1, max_duration=3)
    assert decoded.size == 3 * SAMPLE_RATE


def test_extracts_audio_from_video(tmp_path):
    path = tmp_path / "clip.mp4"
    result = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi",
        "-i", "color=c=black:s=64x64:d=4", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=4", "-shortest",
        "-c:v", "mpeg4", "-c:a", "aac", "-y", str(path),
    ], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    assert abs(probe_duration(path) - 4) < 0.1
    assert load_audio(path).size >= 3 * SAMPLE_RATE
