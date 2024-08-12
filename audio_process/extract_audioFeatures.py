from pyannote.core import Segment

def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res:
        start = item.start
        end = item.end
        text = item.text.strip()
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = round(text_cache[0][0].start, 1)
    end = round(text_cache[-1][0].end, 1)
    return Segment(start, end), spk, sentence


PUNC_SENT_END = [',', '.', '?', '!', "，", "。", "？", "！"]


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


def write_to_txt(spk_sent, file):
    with open(file, 'w') as fp:
        for seg, spk, sentence in spk_sent:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
            fp.write(line)




import torch
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pathlib import Path


# 测试音频： https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_speaker_demo.wav
audio = "./test/asr/data/asr_speaker_demo.wav"
spk_rec_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="your huggingface token")
spk_rec_pipeline.to(torch.device("cuda"))

asr_result, info = asr_model.transcribe(audio, language="zh", beam_size=5)
diarization_result = spk_rec_pipeline(audio)

final_result = diarize_text(asr_result, diarization_result)
for segment, spk, sent in final_result:
    print("[%.2fs -> %.2fs] %s %s" % (segment.start, segment.end, sent, spk))

