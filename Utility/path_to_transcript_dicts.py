import glob
import os
import random
from pathlib import Path


def limit_to_n(path_to_transcript_dict, n=40000):
    limited_dict = dict()
    if len(path_to_transcript_dict.keys()) > n:
        for key in random.sample(list(path_to_transcript_dict.keys()), n):
            limited_dict[key] = path_to_transcript_dict[key]
        return limited_dict
    else:
        return path_to_transcript_dict


def build_path_to_transcript_dict_mls_italian():
    lang = "italian"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(
        build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
    )


def build_path_to_transcript_dict_mls_french():
    lang = "french"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(
        build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
    )


def build_path_to_transcript_dict_mls_dutch():
    lang = "dutch"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(
        build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
    )


def build_path_to_transcript_dict_mls_polish():
    lang = "polish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(
        build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
    )


def build_path_to_transcript_dict_mls_spanish():
    lang = "spanish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(
        build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
    )


def build_path_to_transcript_dict_mls_portuguese():
    lang = "portuguese"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(
        build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
    )


def build_path_to_transcript_dict_multi_ling_librispeech_template(root):
    """
    https://arxiv.org/abs/2012.03411
    """
    path_to_transcript = dict()
    with open(os.path.join(root, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_folders = line.split("\t")[0].split("_")
            wav_path = os.path.join(
                root,
                "audio",
                wav_folders[0],
                wav_folders[1],
                line.split("\t")[0] + ".flac",
            )
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
            else:
                print(f"not found: {wav_path}")
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_mls_franconian_german():
    """Franconian"""
    lang = "german"
    root = f"/home/debehril/speech/dbwork/mul/spielwiese/SpeechCorpora/mls_2020/de/mls_{lang}_opus/train"
    return limit_to_n(
        build_path_to_transcript_dict_multi_ling_librispeech_speakerlist_template(
            root, speakers=["2043"]
        )
    )


def build_path_to_transcript_dict_multi_ling_librispeech_speakerlist_template(
    root, speakers: list
):
    """
    https://arxiv.org/abs/2012.03411
    """
    speakers = [str(speaker) for speaker in speakers]
    path_to_transcript = dict()
    with open(os.path.join(root, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            speaker = line.split("\t")[0].split("_")[0]
            if speaker not in speakers:
                continue
            norm_transcript = line.split("\t")[1]
            wav_folders = line.split("\t")[0].split("_")
            wav_path = os.path.join(
                root,
                "audio",
                wav_folders[0],
                wav_folders[1],
                line.split("\t")[0] + ".flac",
            )
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
            else:
                wav_path = os.path.join(
                    root,
                    "audio",
                    wav_folders[0],
                    wav_folders[1],
                    line.split("\t")[0] + ".opus",
                )
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
                else:
                    print(f"not found: {wav_path}")
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_karlsson():
    root = "/mount/resources/speech/corpora/HUI_German/Karlsson"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_eva():
    root = "/mount/resources/speech/corpora/HUI_German/Eva"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_bernd():
    root = "/mount/resources/speech/corpora/HUI_German/Bernd"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_friedrich():
    root = "/mount/resources/speech/corpora/HUI_German/Friedrich"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hokus():
    root = "/mount/resources/speech/corpora/HUI_German/Hokus"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hui_others():
    root = "/mount/resources/speech/corpora/HUI_German/others"
    pttd = dict()
    for speaker in os.listdir(root):
        pttd.update(
            build_path_to_transcript_dict_hui_template(root=f"{root}/{speaker}")
        )
    return pttd


def build_path_to_transcript_dict_hui_bernd_335():
    """Standard German"""
    root = "/mount/resources/speech/corpora/HUI_German/Bernd"
    book_dirs = ["wunderwelten"]
    return limit_to_n(
        build_path_to_transcript_dict_hui_template(
            root=root, specific_book_dirs=book_dirs
        )
    )


def build_path_to_transcript_dict_hui_karlsson_324():
    """Standard German"""
    root = "/mount/resources/speech/corpora/HUI_German/others/Julia_Niedermaier"
    book_dirs = [
        "smtliche_schriften_19111921_teil_1",
        "smtliche_schriften_19111921_teil_2",
    ]
    return limit_to_n(
        build_path_to_transcript_dict_hui_template(
            root=root, specific_book_dirs=book_dirs
        )
    )


def build_path_to_transcript_dict_hui_julia_320():
    """Austrian German"""
    root = "/mount/resources/speech/corpora/HUI_German/others/Julia_Niedermaier"
    book_dirs = [
        "deutsche_literaturgeschichte",
        "geheimnis_der_alten_mamsell",
        "saemtliche_schriften5",
    ]
    return limit_to_n(
        build_path_to_transcript_dict_hui_template(
            root=root, specific_book_dirs=book_dirs
        )
    )


def build_path_to_transcript_dict_hui_availle_321():
    """Austrian German"""
    root = "/mount/resources/speech/corpora/HUI_German/others/Availle"
    book_dirs = [
        "saemtliche_schriften3",
        "saemtliche_schriften4",
        "saemtliche_schriften5",
    ]
    return limit_to_n(
        build_path_to_transcript_dict_hui_template(
            root=root, specific_book_dirs=book_dirs
        )
    )


def build_path_to_transcript_dict_hui_ramona_deiniger_schnabel():
    """Standard German"""
    root = "/mount/resources/speech/corpora/HUI_German/Ramona_Deininger-Schnabel"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hui_alexandra_bogensperger(n=40000):
    """Standard German"""
    root = "/mount/resources/speech/corpora/HUI_German/others/Alexandra_Bogensperger"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root), n=n)


def build_path_to_transcript_dict_hui_julia_niedermaier():
    """Austrian"""
    root = "/speech/dbwork/mul/spielwiese3/debehril/accented_speech/corpora/accent/HUI-Audio-Corpus-German/others/Julia_Niedermaier"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hui_marham63():
    """Bavarian"""
    root = "/speech/dbwork/mul/spielwiese3/debehril/accented_speech/corpora/accent/HUI-Audio-Corpus-German/others/marham63"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hui_dom_bombadil():
    """Swabian"""
    root = "/speech/dbwork/mul/spielwiese3/debehril/accented_speech/corpora/accent/HUI-Audio-Corpus-German/others/DomBombadil"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hui_crln_yldz_ksr():
    """Kiezdeutsch"""
    root = "/speech/dbwork/mul/spielwiese3/debehril/accented_speech/corpora/accent/HUI-Audio-Corpus-German/others/Crln_Yldz_Ksr"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hui_template(root, specific_book_dirs=None):
    """
    https://arxiv.org/abs/2106.06309
    """
    path_to_transcript = dict()
    book_dirs = os.listdir(root)
    if specific_book_dirs:
        if isinstance(book_dirs, str):
            book_dirs = [book_dirs]
        book_dirs = [bd for bd in book_dirs if bd in specific_book_dirs]
    for el in book_dirs:
        if os.path.isdir(os.path.join(root, el)):
            with open(
                os.path.join(root, el, "metadata.csv"), "r", encoding="utf8"
            ) as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    norm_transcript = line.split("|")[1]
                    wav_path = os.path.join(
                        root, el, "wavs", line.split("|")[0] + ".wav"
                    )
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = norm_transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_elizabeth():
    root = "/mount/resources/speech/corpora/MAILabs_british_single_speaker_elizabeth"
    path_to_transcript = dict()
    for el in os.listdir(root):
        if os.path.isdir(os.path.join(root, el)):
            with open(
                os.path.join(root, el, "metadata.csv"), "r", encoding="utf8"
            ) as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    norm_transcript = line.split("|")[2]
                    wav_path = os.path.join(
                        root, el, "wavs", line.split("|")[0] + ".wav"
                    )
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = norm_transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_nancy():
    root = "/mount/resources/speech/corpora/NancyKrebs"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[1]
            wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_integration_test():
    root = "/mount/resources/speech/corpora/NancyKrebs"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n")[:500]:
        if line.strip() != "":
            norm_transcript = line.split("|")[1]
            wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_hokuspokus():
    path_to_transcript = dict()
    for transcript_file in os.listdir(
        "/mount/resources/speech/corpora/LibriVox.Hokuspokus/txt"
    ):
        if transcript_file.endswith(".txt"):
            with open(
                "/mount/resources/speech/corpora/LibriVox.Hokuspokus/txt/"
                + transcript_file,
                "r",
                encoding="utf8",
            ) as tf:
                transcript = tf.read()
            wav_path = (
                "/mount/resources/speech/corpora/LibriVox.Hokuspokus/wav/"
                + transcript_file.rstrip(".txt")
                + ".wav"
            )
            path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_fluxsing():
    root = "/mount/resources/speech/corpora/FluxSing"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[2]
            wav_path = os.path.join(root, line.split("|")[0])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_vctk():
    path_to_transcript = dict()
    for transcript_dir in os.listdir("/mount/resources/speech/corpora/VCTK/txt"):
        for transcript_file in os.listdir(
            f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}"
        ):
            if transcript_file.endswith(".txt"):
                with open(
                    f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}/"
                    + transcript_file,
                    "r",
                    encoding="utf8",
                ) as tf:
                    transcript = tf.read()
                wav_path = (
                    f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{transcript_dir}/"
                    + transcript_file.rstrip(".txt")
                    + "_mic2.flac"
                )
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_vctk_uk_p228():
    # age: 22, sex: F, accent: English, duration in s: 1422, region: Southern England
    path_to_transcript = dict()
    speaker = "p228"
    transcript_dir = f"/mount/resources/speech/corpora/VCTK/txt/{speaker}"
    for transcript_file in os.listdir(transcript_dir):
        if transcript_file.endswith(".txt"):
            with open(
                f"{transcript_dir}/" + transcript_file,
                "r",
                encoding="utf8",
            ) as tf:
                transcript = tf.read()
            wav_path = (
                f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{speaker}/"
                + transcript_file.rstrip(".txt")
                + "_mic1.flac"
            )
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_vctk_uk_p229():
    # age: 23, sex: F, accent: English, duration in s: 1167, region: Southern England
    path_to_transcript = dict()
    speaker = "p229"
    transcript_dir = f"/mount/resources/speech/corpora/VCTK/txt/{speaker}"
    for transcript_file in os.listdir(transcript_dir):
        if transcript_file.endswith(".txt"):
            with open(
                f"{transcript_dir}/" + transcript_file,
                "r",
                encoding="utf8",
            ) as tf:
                transcript = tf.read()
            wav_path = (
                f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{speaker}/"
                + transcript_file.rstrip(".txt")
                + "_mic1.flac"
            )
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_vctk_us_p299():
    # age: 25, sex: F, accent: American, duration in s: 1359, region: California
    path_to_transcript = dict()
    speaker = "p299"
    transcript_dir = f"/mount/resources/speech/corpora/VCTK/txt/{speaker}"
    for transcript_file in os.listdir(transcript_dir):
        if transcript_file.endswith(".txt"):
            with open(
                f"{transcript_dir}/" + transcript_file,
                "r",
                encoding="utf8",
            ) as tf:
                transcript = tf.read()
            wav_path = (
                f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{speaker}/"
                + transcript_file.rstrip(".txt")
                + "_mic1.flac"
            )
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_vctk_us_p300():
    # age: 23, sex: F, accent: American, duration in s: 1743, region: California
    path_to_transcript = dict()
    speaker = "p300"
    transcript_dir = f"/mount/resources/speech/corpora/VCTK/txt/{speaker}"
    for transcript_file in os.listdir(transcript_dir):
        if transcript_file.endswith(".txt"):
            with open(
                f"{transcript_dir}/" + transcript_file,
                "r",
                encoding="utf8",
            ) as tf:
                transcript = tf.read()
            wav_path = (
                f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{speaker}/"
                + transcript_file.rstrip(".txt")
                + "_mic1.flac"
            )
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_libritts():
    path_train = "/mount/resources/speech/corpora/LibriTTS/train-clean-100"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(
                        os.path.join(path_train, speaker, chapter, file),
                        "r",
                        encoding="utf8",
                    ) as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[
                        os.path.join(path_train, speaker, chapter, wav_file)
                    ] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_libritts_all_clean():
    path_train = "/mount/resources/speech/corpora/LibriTTS/all_clean"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(
                        os.path.join(path_train, speaker, chapter, file),
                        "r",
                        encoding="utf8",
                    ) as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[
                        os.path.join(path_train, speaker, chapter, wav_file)
                    ] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_libritts_other500():
    path_train = "/mount/resources/asr-data/LibriTTS/train-other-500"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(
                        os.path.join(path_train, speaker, chapter, file),
                        "r",
                        encoding="utf8",
                    ) as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[
                        os.path.join(path_train, speaker, chapter, wav_file)
                    ] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_ljspeech():
    path_to_transcript = dict()
    for transcript_file in os.listdir(
        "/mount/resources/speech/corpora/LJSpeech/16kHz/txt"
    ):
        with open(
            "/mount/resources/speech/corpora/LJSpeech/16kHz/txt/" + transcript_file,
            "r",
            encoding="utf8",
        ) as tf:
            transcript = tf.read()
        wav_path = (
            "/mount/resources/speech/corpora/LJSpeech/16kHz/wav/"
            + transcript_file.rstrip(".txt")
            + ".wav"
        )
        path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_att_hack():
    path_to_transcript = dict()
    for transcript_file in os.listdir(
        "/mount/resources/speech/corpora/FrenchExpressive/txt"
    ):
        if transcript_file.endswith(".txt"):
            with open(
                "/mount/resources/speech/corpora/FrenchExpressive/txt/"
                + transcript_file,
                "r",
                encoding="utf8",
            ) as tf:
                transcript = tf.read()
            wav_path = (
                "/mount/resources/speech/corpora/FrenchExpressive/wav/"
                + transcript_file.rstrip(".txt")
                + ".wav"
            )
            path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10de():
    path_to_transcript = dict()
    with open(
        "/mount/resources/speech/corpora/CSS10/german/transcript.txt", encoding="utf8"
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                "/mount/resources/speech/corpora/CSS10/german/" + line.split("|")[0]
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10cmn():
    path_to_transcript = dict()
    with open(
        "/mount/resources/speech/corpora/CSS10/chinese/transcript.txt", encoding="utf8"
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                "/mount/resources/speech/corpora/CSS10/chinese/" + line.split("|")[0]
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_vietTTS():
    path_to_transcript = dict()
    root = "/mount/resources/speech/corpora/VietTTS"
    with open(root + "/meta_data.tsv", encoding="utf8") as f:
        transcriptions = f.read()
    for line in transcriptions.split("\n"):
        if line.strip() != "":
            parsed_line = line.split(".wav")
            audio_path = parsed_line[0]
            transcript = parsed_line[1]
            path_to_transcript[
                os.path.join(root, audio_path + ".wav")
            ] = transcript.strip()
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_thorsten():
    path_to_transcript = dict()
    root = "/mount/resources/speech/corpora/Thorsten_DE/V2"
    with open(root + "/metadata_train.csv", encoding="utf8") as f:
        transcriptions = f.read()
    with open(root + "/metadata_dev.csv", encoding="utf8") as f:
        transcriptions += "\n" + f.read()
    with open(root + "/metadata_test.csv", encoding="utf8") as f:
        transcriptions += "\n" + f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                root + "/wavs/" + line.split("|")[0] + ".wav"
            ] = line.split("|")[1]
    return path_to_transcript


def build_path_to_transcript_dict_thorsten_2020():
    path_to_transcript = dict()
    with open(
        "/mount/resources/speech/corpora/Thorsten_DE/metadata_shuf.csv", encoding="utf8"
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                "/mount/resources/speech/corpora/Thorsten_DE/wavs/"
                + line.split("|")[0]
                + ".wav"
            ] = line.split("|")[1]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10el():
    path_to_transcript = dict()
    language = "greek"
    with open(
        f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt",
        encoding="utf8",
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10nl():
    path_to_transcript = dict()
    language = "dutch"
    with open(
        f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt",
        encoding="utf8",
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10fi():
    path_to_transcript = dict()
    language = "finnish"
    with open(
        f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt",
        encoding="utf8",
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10ru():
    path_to_transcript = dict()
    language = "russian"
    with open(
        f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt",
        encoding="utf8",
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10hu():
    path_to_transcript = dict()
    language = "hungarian"
    with open(
        f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt",
        encoding="utf8",
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10es():
    path_to_transcript = dict()
    language = "spanish"
    with open(
        f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt",
        encoding="utf8",
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10fr():
    path_to_transcript = dict()
    language = "french"
    with open(
        f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt",
        encoding="utf8",
    ) as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[
                f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"
            ] = line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_nvidia_hifitts():
    path_to_transcript = dict()
    transcripts = list()
    root = "/mount/resources/speech/corpora/hi_fi_tts_v0"

    import json

    for jpath in [
        f"{root}/6097_manifest_clean_dev.json",
        f"{root}/6097_manifest_clean_test.json",
        f"{root}/6097_manifest_clean_train.json",
        f"{root}/9017_manifest_clean_dev.json",
        f"{root}/9017_manifest_clean_test.json",
        f"{root}/9017_manifest_clean_train.json",
        f"{root}/92_manifest_clean_dev.json",
        f"{root}/92_manifest_clean_test.json",
        f"{root}/92_manifest_clean_train.json",
    ]:
        with open(jpath, encoding="utf-8", mode="r") as jfile:
            for line in jfile.read().split("\n"):
                if line.strip() != "":
                    transcripts.append(json.loads(line))

    for transcript in transcripts:
        path = transcript["audio_filepath"]
        norm_text = transcript["text_normalized"]
        path_to_transcript[f"{root}/{path}"] = norm_text

    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_spanish_blizzard_train():
    root = "/mount/resources/speech/corpora/Blizzard2021/spanish_blizzard_release_2021_v2/hub"
    path_to_transcript = dict()
    with open(os.path.join(root, "train_text.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, "train_wav", line.split("\t")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_aishell3():
    root = "/mount/resources/speech/corpora/aishell3/train"
    path_to_transcript_dict = dict()
    with open(root + "/label_train-set.txt", mode="r", encoding="utf8") as f:
        transcripts = f.read().replace("$", "").replace("%", ",").split("\n")
    for transcript in transcripts:
        if transcript.strip() != "" and not transcript.startswith("#"):
            parsed_line = transcript.split("|")
            audio_file = f"{root}/wav/{parsed_line[0][:7]}/{parsed_line[0]}.wav"
            kanji = parsed_line[2]
            path_to_transcript_dict[audio_file] = kanji
    return limit_to_n(path_to_transcript_dict)


def build_path_to_transcript_dict_VIVOS_viet():
    root = "/mount/resources/speech/corpora/VIVOS_vietnamese/train"
    path_to_transcript_dict = dict()
    with open(root + "/prompts.txt", mode="r", encoding="utf8") as f:
        transcripts = f.read().split("\n")
    for transcript in transcripts:
        if transcript.strip() != "":
            parsed_line = transcript.split(" ")
            audio_file = f"{root}/waves/{parsed_line[0][:10]}/{parsed_line[0]}.wav"
            path_to_transcript_dict[audio_file] = " ".join(parsed_line[1:]).lower()
    return limit_to_n(path_to_transcript_dict)


def build_path_to_transcript_dict_RAVDESS():
    root = "/mount/resources/speech/corpora/RAVDESS"
    path_to_transcript_dict = dict()
    for speaker_dir in os.listdir(root):
        for audio_file in os.listdir(os.path.join(root, speaker_dir)):
            if audio_file.split("-")[4] == "01":
                path_to_transcript_dict[
                    os.path.join(root, speaker_dir, audio_file)
                ] = "Kids are talking by the door."
            else:
                path_to_transcript_dict[
                    os.path.join(root, speaker_dir, audio_file)
                ] = "Dogs are sitting by the door."
    return path_to_transcript_dict


def build_path_to_transcript_dict_ESDS():
    root = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    path_to_transcript_dict = dict()
    for speaker_dir in os.listdir(root):
        if speaker_dir.startswith("00"):
            if int(speaker_dir) > 10:
                with open(
                    f"{root}/{speaker_dir}/fixed_unicode.txt", mode="r", encoding="utf8"
                ) as f:
                    transcripts = f.read()
                for line in (
                    transcripts.replace("\n\n", "\n").replace(",", ", ").split("\n")
                ):
                    if line.strip() != "":
                        filename, text, emo_dir = line.split("\t")
                        filename = speaker_dir + "_" + filename.split("_")[1]
                        path_to_transcript_dict[
                            f"{root}/{speaker_dir}/{emo_dir}/{filename}.wav"
                        ] = text
    return path_to_transcript_dict


def build_path_to_transcript_dict_blizzard_2013():
    path_to_transcript = dict()
    root = "/mount/resources/speech/corpora/Blizzard2013/train/segmented/"
    with open(root + "prompts.gui", encoding="utf8") as f:
        transcriptions = f.read()
    blocks = transcriptions.split("||\n")
    for block in blocks:
        trans_lines = block.split("\n")
        if trans_lines[0].strip() != "":
            transcript = (
                trans_lines[1]
                .replace("@", "")
                .replace("#", ",")
                .replace("|", "")
                .replace(";", ",")
                .replace(":", ",")
                .replace(" 's", "'s")
                .replace(", ,", ",")
                .replace("  ", " ")
                .replace(" ,", ",")
                .replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .rstrip(" ,")
            )
            path_to_transcript[root + "wavn/" + trans_lines[0] + ".wav"] = transcript
    return path_to_transcript


def build_file_list_singing_voice_audio_database():
    root = "/mount/resources/speech/corpora/singing_voice_audio_dataset/monophonic"
    file_list = list()
    for corw in os.listdir(root):
        for singer in os.listdir(os.path.join(root, corw)):
            for audio in os.listdir(os.path.join(root, corw, singer)):
                file_list.append(os.path.join(root, corw, singer, audio))
    return file_list


def build_path_to_transcript_dict_blizzard2023_ad():
    root = "/mount/resources/speech/corpora/Blizzard2023/AD"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = (
                    norm_transcript.replace("§", "")
                    .replace("#", "")
                    .replace("~", "")
                    .replace("»", '"')
                    .replace("«", '"')
                )
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb():
    root = "/mount/resources/speech/corpora/Blizzard2023/NEB"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = (
                    norm_transcript.replace("§", "")
                    .replace("#", "")
                    .replace("~", "")
                    .replace("»", '"')
                    .replace("«", '"')
                )
    return path_to_transcript


def build_path_to_transcript_dict_synpaflex_norm_subset():
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "/mount/resources/speech/corpora/synpaflex-corpus/5/v0.1/"
    path_to_transcript = dict()
    for text_path in glob.iglob(os.path.join(root, "**/*_norm.txt"), recursive=True):
        with open(text_path, "r", encoding="utf8") as file:
            norm_transcript = file.read()
        path_obj = Path(text_path)
        wav_path = str(
            (path_obj.parent.parent / path_obj.name[:-9]).with_suffix(".wav")
        )
        if Path(wav_path).exists():
            path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_synpaflex_all():
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "/mount/resources/speech/corpora/synpaflex-corpus/5/v0.1/"
    path_to_transcript = dict()
    for text_path in glob.iglob(os.path.join(root, "**/*.txt"), recursive=True):
        with open(text_path, "r", encoding="utf8") as file:
            transcript = file.read()
        path_obj = Path(text_path)
        wav_path = str(
            (path_obj.parent.parent / path_obj.name[:-4]).with_suffix(".wav")
        )
        if Path(wav_path).exists():
            path_to_transcript[wav_path] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_siwis_subset():
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "/mount/resources/speech/corpora/SiwisFrenchSpeechSynthesisDatabase/"
    # part4 and part5 are not segmented
    sub_dirs = ["part1", "part2", "part3"]
    path_to_transcript = dict()
    for sd in sub_dirs:
        for text_path in glob.iglob(os.path.join(root, "text", sd, "*.txt")):
            with open(text_path, "r", encoding="utf8") as file:
                norm_transcript = file.read()
            path_obj = Path(text_path)
            wav_path = str(
                (
                    path_obj.parent.parent.parent / "wavs" / sd / path_obj.stem
                ).with_suffix(".wav")
            )
            if Path(wav_path).exists():
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


if __name__ == "__main__":
    ptt = build_path_to_transcript_dict_blizzard2023_ad()
    print(ptt)
