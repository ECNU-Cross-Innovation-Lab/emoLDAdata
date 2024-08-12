from pathlib import Path
import os
import subprocess
import moviepy.editor as mp
from pyannote.audio import Pipeline
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import torch,torchaudio
import os
import scipy.io.wavfile as wav



os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



# 定义文件夹路径
input_folder = Path('A:\ZQJ\CILlab\ScientificData/video_data')



def extract_audio_from_mp4(mp4_file, wav_file):
    """从MP4文件中提取WAV文件"""
    command = f"ffmpeg -i \"{mp4_file}\" -vn -acodec pcm_s16le -ar 44100 -ac 2 \"{wav_file}\""
    subprocess.run(command, shell=True)





if __name__=='__main__':
    # 初始化 pyannote.audio 说话人分离 pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token="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")
    pipeline.to(torch.device("cuda"))

    # token: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
    for item in input_folder.iterdir():
        print(item.name)
        # flag=False
        if item.is_dir() and (item.name.startswith('2023-L-D-A-ECNU-CS-z04') or item.name.startswith('2023-L-D-A-ECNU-CS-z05')) :
            # flag=True
            print(item.name)
            for file in item.iterdir():
                if file.suffix == '.mp4':
                    mp4_path = file
                    wav_filename = file.stem + '.wav'
                    wav_path = Path(item,wav_filename)
                    # sample_rate, audio_data = wav.read(wav_path)
                    # 提取音频
                    print(f"Extracting audio from {mp4_path} to {wav_path}...")

                    if not os.path.exists(wav_path):
                        extract_audio_from_mp4(mp4_path, wav_path)

                    # 读取音频
                    print(f"seprating audio from {wav_path}")
                    # 执行说话人分离
                    audio, sample_rate = torchaudio.load(wav_path)
                    # diarization = pipeline({"waveform": audio, "sample_rate": wav_path})

                    diarization = pipeline(wav_path, num_speakers=2)
                    # 语音分离
                    output_subfolder = Path(item, file.stem+"_seprate")
                    if not output_subfolder.exists():
                        output_subfolder.mkdir()
                    # print(diarization)
                    # 保存分离后的音频片段
                    speaker_a_audio = []
                    speaker_b_audio = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        start = int(turn.start * sample_rate)  # 转换为毫秒
                        end = int(turn.end * sample_rate)  # 转换为毫秒
                        # print(f"speaker:{speaker} from {turn.start} to {turn.end}")
                        if speaker == 'SPEAKER_00':
                            speaker_a_audio.append(audio[:,start:end])
                        elif speaker == 'SPEAKER_01':
                            speaker_b_audio.append(audio[:,start:end])
                    with open(Path(item,"audio.rttm"), "w") as rttm:
                        diarization.write_rttm(rttm)

                    # print(len(speaker_a_audio))
                    # print(speaker_a_audio[0])
                    # 合并音频片段
                    # print(speaker_a_audio)
                    # print(speaker_b_audio)
                    speaker_a_audio = np.concatenate(speaker_a_audio,axis=1)
                    speaker_b_audio = np.concatenate(speaker_b_audio,axis=1)
                    # print(speaker_a_audio.shape)
                    # # 保存为文件
                    # print("save_files")
                    # print(speaker_a_audio.shape)
                    # print(speaker_a_audio)
                    # print(np.ravel(speaker_a_audio).shape)
                    # print(np.ravel(speaker_a_audio))
                    # sf.write("baudio.wav", speaker_b_audio,  samplerate=sample_rate,format="WAV",subtype='PCM_16')

                    sf.write(Path(item,"speaker_a_audio.wav"), speaker_a_audio.transpose(), samplerate=sample_rate,format="WAV")
                    sf.write(Path(item,"speaker_b_audio.wav"), speaker_b_audio.transpose(),  samplerate=sample_rate,format="WAV")
                    print(f"Separating audio for {wav_path}...")
        # if flag:break



    print("All files processed.")
