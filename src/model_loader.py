from pathlib import Path

from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          M2M100ForConditionalGeneration, M2M100Tokenizer,
                          MarianMTModel, MBart50TokenizerFast,
                          MBartForConditionalGeneration, NllbTokenizerFast)

from src.tokenizer_utils import ID_MAP_FILENAME, load_remapped_tokenizer


def load_nmt_model(model_name: str, device):
    name_lower = model_name.lower()

    # Resolve language codes once per architecture
    if "mbart" in name_lower:
        src_lang, tgt_lang = "en_XX", "ar_AR"
    elif "nllb" in name_lower:
        src_lang, tgt_lang = "eng_Latn", "arb_Arab"
    else:  # M2M100, OpusMT
        src_lang, tgt_lang = "en", "ar"

    # Load model + tokenizer
    if "m2m100" in name_lower:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang=src_lang)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

    elif "mbart" in name_lower:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
        model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

    elif "nllb" in name_lower:
        tokenizer = NllbTokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)

    # Remapped tokenizer — reuses the same codes resolved above
    if (Path(model_name) / ID_MAP_FILENAME).exists():
        tokenizer = load_remapped_tokenizer(
            model_name,
            base_tokenizer_cls=type(tokenizer),
            src_lang=src_lang,      # ← consistent, no hardcoding twice
            tgt_lang=tgt_lang,
        )

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    return model, tokenizer
    