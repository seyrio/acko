#!/usr/bin/env python3
"""
Fix for Whisper dtype compatibility issues
Handles "expected m1 and m2 to have the same dtype, but got: float != double" error
"""

import whisper
import torch
import numpy as np
import pyaudio
import wave
import warnings
import tempfile
import os

class FixedWhisperRecorder:
    def __init__(self, model_name="base"):
        print(f"Loading Whisper model: {model_name}")
        
        # Set PyTorch to use float32 consistently
        torch.set_default_dtype(torch.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.model = whisper.load_model(model_name)
            
        # Ensure model uses float32
        self.model = self.model.float()
        if hasattr(self.model, 'encoder'):
            self.model.encoder = self.model.encoder.float()
        if hasattr(self.model, 'decoder'):
            self.model.decoder = self.model.decoder.float()
    
    def record_audio_fixed(self, duration=5, sample_rate=16000):
        """Record audio with proper dtype handling"""
        print(f"Recording for {duration} seconds at {sample_rate} Hz...")
        
        audio = pyaudio.PyAudio()
        
        # Try to find working input device
        device_index = None
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                try:
                    # Test device
                    test_stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    device_index = i
                    print(f"Using device: {info['name']}")
                    break
                except:
                    continue
        
        if device_index is None:
            # Try without specifying device
            print("Using default input device")
        
        # Record audio
        stream_config = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': sample_rate,
            'input': True,
            'frames_per_buffer': 1024
        }
        
        if device_index is not None:
            stream_config['input_device_index'] = device_index
        
        stream = audio.open(**stream_config)
        
        frames = []
        frames_to_record = int(sample_rate / 1024 * duration)
        
        print("Recording... Speak now!")
        max_level = 0
        
        for i in range(frames_to_record):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            
            # Monitor level
            chunk = np.frombuffer(data, dtype=np.int16)
            level = np.max(np.abs(chunk))
            max_level = max(max_level, level)
            
            if i % 10 == 0:  # Update every 10 chunks
                progress = (i / frames_to_record) * 100
                print(f"\rProgress: {progress:.0f}% | Max level: {max_level}", end="", flush=True)
        
        print(f"\nRecording complete! Max level: {max_level}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if max_level < 100:
            print(f"⚠️  Audio level is very low ({max_level}). Try speaking louder.")
        
        # Convert to numpy array with correct dtype
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 and normalize to [-1, 1] range
        audio_float32 = audio_array.astype(np.float32) / 32768.0
        
        return audio_float32
    
    def transcribe_with_dtype_fix(self, audio_array, language=None):
        """Transcribe with proper dtype handling"""
        if audio_array is None or len(audio_array) == 0:
            print("Empty audio array")
            return None
        
        try:
            print("Transcribing with dtype fixes...")
            
            # Ensure audio is float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Ensure audio is in correct range
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Transcription options
            options = {
                'fp16': False,  # Force float32
                'language': language if language else None,
                'task': 'transcribe'
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = self.model.transcribe(audio_array, **options)
            
            return result
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            
            # Try alternative method - save to file first
            print("Trying file-based transcription...")
            try:
                return self._transcribe_via_file(audio_array, language)
            except Exception as file_error:
                print(f"File-based transcription also failed: {file_error}")
                return None
    
    def _transcribe_via_file(self, audio_array, language=None):
        """Fallback: save to file and transcribe"""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Convert to int16 for WAV
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
            
            # Transcribe file
            options = {'fp16': False}
            if language:
                options['language'] = language
                
            result = self.model.transcribe(temp_path, **options)
            return result
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def record_and_transcribe_fixed(self, duration=5, language="en"):
        """Complete workflow with all fixes applied"""
        print("🎙️  Starting recording and transcription with dtype fixes")
        print("=" * 60)
        
        # Record audio
        audio_array = self.record_audio_fixed(duration)
        
        if audio_array is None:
            print("❌ Recording failed")
            return None
        
        # Transcribe
        result = self.transcribe_with_dtype_fix(audio_array, language)
        
        if result and result.get('text', '').strip():
            print(f"✅ Success: '{result['text']}'")
            return result
        else:
            print("❌ Transcription failed or returned empty text")
            return None


def fix_pytorch_whisper_compatibility():
    """Apply system-wide fixes for PyTorch/Whisper compatibility"""
    print("Applying PyTorch/Whisper compatibility fixes...")
    
    # Set default dtype to float32
    torch.set_default_dtype(torch.float32)
    
    # Set environment variables
    os.environ['TORCH_DEFAULT_DTYPE'] = 'float32'
    
    # Disable mixed precision if it's causing issues
    os.environ['WHISPER_NO_FP16'] = '1'
    
    print("✅ Compatibility fixes applied")


def install_compatible_versions():
    """Show installation commands for compatible versions"""
    print("🔧 INSTALLATION FIXES FOR DTYPE COMPATIBILITY")
    print("=" * 60)
    
    print("Try these installation commands:")
    print()
    
    print("Option 1 - Reinstall with specific versions:")
    print("pip uninstall torch torchvision torchaudio openai-whisper")
    print("pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2")
    print("pip install openai-whisper==20230314")
    print()
    
    print("Option 2 - Force CPU-only PyTorch:")
    print("pip uninstall torch torchvision torchaudio")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("pip install openai-whisper")
    print()
    
    print("Option 3 - Use conda (often more stable):")
    print("conda create -n whisper python=3.9")
    print("conda activate whisper")
    print("conda install pytorch torchvision torchaudio cpuonly -c pytorch")
    print("pip install openai-whisper")
    print()
    
    print("Option 4 - Development version:")
    print("pip install git+https://github.com/openai/whisper.git")


def test_dtype_compatibility():
    """Test if dtype issues are resolved"""
    print("Testing PyTorch/Whisper dtype compatibility...")
    
    try:
        # Test basic PyTorch operations
        a = torch.tensor([1.0, 2.0], dtype=torch.float32)
        b = torch.tensor([3.0, 4.0], dtype=torch.float32)
        c = torch.matmul(a, b)
        print(f"✅ PyTorch basic operations work: {c}")
        
        # Test Whisper model loading
        print("Loading Whisper model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = whisper.load_model("tiny")
        
        # Force float32
        model = model.float()
        
        # Test with dummy audio
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01  # 1 second of quiet noise
        result = model.transcribe(dummy_audio, fp16=False)
        
        print(f"✅ Whisper model test successful")
        print(f"Dummy result: '{result['text']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False


def main():
    """Main function with dtype fixes"""
    print("WHISPER DTYPE COMPATIBILITY FIXER")
    print("=" * 50)
    
    # Apply fixes
    fix_pytorch_whisper_compatibility()
    
    # Test compatibility
    if not test_dtype_compatibility():
        print("\n❌ Compatibility issues detected!")
        install_compatible_versions()
        return
    
    print("\n✅ Compatibility test passed!")
    
    # Run recording test
    try:
        recorder = FixedWhisperRecorder(model_name="tiny")
        
        input("\nPress Enter to test recording (5 seconds)...")
        result = recorder.record_and_transcribe_fixed(duration=5, language="en")
        
        if result:
            print(f"\n🎉 SUCCESS! Transcription: '{result['text']}'")
        else:
            print(f"\n❌ Test failed - check microphone settings")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running:")
        install_compatible_versions()


if __name__ == "__main__":
    main()