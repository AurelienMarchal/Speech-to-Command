from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_wav(
    "/home/aurelienmarchal/Stage/Speech-to-Command/dataset/wavs/aurelien_marchal/put_yellow_in_front_of_blue_tig85983kv.wav")
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least 50 ms
    min_silence_len=50,

    # consider it silent if quieter than the file average dBFS - 4
    silence_thresh=sound_file.dBFS - 4
)


for i, chunk in enumerate(audio_chunks):

    if i == 0:
        combined = audio_chunks[0]
    
    else:
        combined += audio_chunks[i]

    out_file = ".//split_audio//chunk{0}.wav".format(i)
    print("exporting", out_file)
    chunk.export(out_file, format="wav")

combined.export(".//split_audio//combined.wav")