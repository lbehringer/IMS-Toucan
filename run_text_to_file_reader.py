import os

import torch

from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface


def read_texts(
    model_id,
    sentence,
    filename,
    device="cpu",
    language="en",
    lang_emb=None,
    speaker_reference=None,
    input_is_phones=False,
):
    tts = PortaSpeechInterface(
        device=device, tts_model_path=model_id, language=language
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


def le_corbeau_et_le_renard(version, model_id="Meta", exec_device="cpu"):
    os.makedirs("audios", exist_ok=True)

    read_texts(
        model_id=model_id,
        sentence=[
            "Maître Corbeau, sur un arbre perché, tenait en son bec un fromage.",
            "Maître Renard, par l’odeur alléché, lui tint à peu près ce langage:",
            "Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau!",
            "Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois.",
            "À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie.",
            "Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute.",
            "Cette leçon vaut bien un fromage sans doute.",
            "Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus.",
        ],
        filename=f"audios/Le_corbeau_et_le_renard_{version}.wav",
        device=exec_device,
        language="fr",
        speaker_reference=None,
    )


def the_raven(version, model_id="Meta", exec_device="cpu"):
    os.makedirs("audios", exist_ok=True)

    read_texts(
        model_id=model_id,
        sentence=[
            "Once upon a midnight dreary, while I pondered, weak, and weary,",
            "Over many a quaint, and curious volume of forgotten lore,",
            "While I nodded, nearly napping, suddenly, there came a tapping,",
            "As of someone gently rapping, rapping at my chamber door.",
            "Tis some visitor, I muttered, tapping at my chamber door,",
            "Only this, and nothing more.",
            "Ah, distinctly, I remember, it was in, the bleak December,",
            "And each separate dying ember, wrought its ghost upon the floor.",
            "Eagerly, I wished the morrow, vainly, I had sought to borrow",
            "From my books surcease of sorrow, sorrow, for the lost Lenore,",
            "For the rare and radiant maiden, whom the angels name Lenore,",
            "Nameless here, for evermore.",
            "And the silken, sad, uncertain, rustling of each purple curtain",
            "Thrilled me, filled me, with fantastic terrors, never felt before",
        ],
        filename=f"audios/the_raven_{version}.wav",
        device=exec_device,
        language="en",
        speaker_reference=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_emb",
        help="Language embedding to use. Choices are accent-specific (e.g. 'en-gb') or just languages (e.g. 'en')",
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
    lang_emb = args.lang_emb if args.lang_emb else "en"
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
    # set accent-specific language
    if language == "en" and lang_emb[:2] == "en":
        language = lang_emb
    # create output directory
    os.makedirs(f"audios/{model_id}", exist_ok=True)
    filename_model_id = (
        model_id if not args.model_checkpoint else f"{model_id}_{args.model_checkpoint}"
    )

    speaker = (
        os.path.dirname(args.speaker_reference).split("/")[-1] + "-ref"
        if args.speaker_reference
        else ""
    )
    filename = (
        f"audios/{filename_model_id}/{language}-lang_{lang_emb}-emb-{speaker}.wav"
    )
    print(f"Writing file to path: {filename}")
    read_texts(
        model_id=model_id,
        model_checkpoint=args.model_checkpoint,
        sentence=sentence,
        filename=filename,
        language=language,
        lang_emb=lang_emb,
        input_is_phones=args.input_is_phones,
        speaker_reference=args.speaker_reference,
        device=exec_device,
    )
