import os

import torch
import argparse
import glob

from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface
from Utility.storage_config import MODELS_DIR


def read_texts(
    model_id,
    sentence,
    filename,
    device="cpu",
    language="en",
    lang_emb=None,
    speaker_reference=None,
    input_is_phones=False,
    model_checkpoint=None,
):
    tts = PortaSpeechInterface(
        device=device,
        tts_model_path=model_id,
        language=language,
        model_checkpoint=model_checkpoint,
    )
    print(f"Instantiated a PortaSpeechInterface object with language {language}.")
    tts.set_language(language)
    if lang_emb is not None:
        tts.set_language_embedding(lang_emb)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(
        text_list=sentence, file_location=filename, input_is_phones=input_is_phones
    )
    del tts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accent",
        help="Accent to use. Depending on the model, choices are accent-language combinations (e.g. 'es-en') or just languages (e.g. 'en')",
    )
    parser.add_argument("--language", help="Language to use (e.g. 'en')")
    parser.add_argument("--model_id", help="Name of the model to use (e.g. 'Meta')")
    parser.add_argument(
        "--input_is_phones",
        help="Bool whether the input is phones",
        action="store_true",
    )
    parser.add_argument(
        "--speaker_reference", help="path of sample audio from reference speaker"
    )
    args = parser.parse_args()
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    language = args.language if args.language else "en"
    if args.model_id:
        model_id = args.model_id
    else:
        model_id = "Meta"
    sentence_dict = dict()
    sentence_dict["de"] = [
        "Heute ist schönes Frühlingswetter.",
        "Die Sonne lacht.",
        "Am blauen Himmel ziehen die Wolken.",
        "Über die Felder weht ein Wind.",
        "Gestern stürmte es noch.",
        "Montag war es uns zu regnerisch.",
    ]
    sentence_dict["en"] = [
        "It is important to distinguish between dialect and accent.",
        "A dialect includes a certain variety of pronunciation, vocabulary, and grammar.",
        "In contrast, an accent only refers to the aspect of pronunciation.",
        # "I have no sentence.",
        # "I don't know what to say.",
        # "Please someone give me the power to be creative.",
    ]
    sentence_dict["es"] = [
        "Hay gemas de gran valor en la tienda.",
        "Tienen un unevo puerto para botar barcos.",
        "Para hacer hielo, echa más agua.",
        "Mandó la postal en un sobre de papel grueso.",
    ]
    sentence = (
        sentence_dict[language]
        if language in sentence_dict
        else sentence_dict["en"]
        if language == "en-gb" or language == "en-us"
        else sentence_dict[language[-2:]]  # works for language at-de, hi-en, eu-es...
    )

    models_prefix = "1uk_1us_"
    models = glob.glob(os.path.join(MODELS_DIR, f"PortaSpeech_{models_prefix}*"))
    models = [os.path.basename(m).split("_", maxsplit=1)[1] for m in models]
    print(models)
    speaker_references = [
        "/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/p228/p228_003_mic1.flac",
        "/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/p299/p299_003_mic1.flac",
    ]
    g2ps = ["en", "en-gb"]
    embs = ["en-us", "en-gb", "en-mix"]
    for m in models:
        # create output dir
        os.makedirs(f"audios/{m}", exist_ok=True)
        for g in g2ps:
            for e in embs:
                for s in speaker_references:
                    speaker = os.path.dirname(s).split("/")[-1]
                    filename = f"audios/{m}/{g}-g2p_{e}-emb_{speaker}-ref.wav"
                    read_texts(
                        model_id=m,
                        device=exec_device,
                        model_checkpoint=args.model_checkpoint,
                        sentence=sentence,
                        filename=filename,
                        language=g,
                        lang_emb=e,
                        input_is_phones=args.input_is_phones,
                        speaker_reference=s,
                    )
                # without speaker_reference
                filename = f"audios/{m}/{g}-g2p_{e}-emb.wav"
                read_texts(
                    model_id=m,
                    device=exec_device,
                    model_checkpoint=args.model_checkpoint,
                    sentence=sentence,
                    filename=filename,
                    language=g,
                    accent=e,
                    input_is_phones=args.input_is_phones,
                )
