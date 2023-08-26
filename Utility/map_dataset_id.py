import os
import torch
from Utility.storage_config import PREPROCESSING_DIR


def map_dataset_to_dataset_id(corpus_dir):
    """create a mapping for each dataset name to an dataset_id, i.e. a LongTensor (comparable to get_language_id() in TextFrontend.py)
    Args:
    corpus_dir (Pathlike)
        We treat this as the dataset name. This is the directory where the dataset should be saved (as specified in training/finetuning script)
    """

    # if parent dirs are the same as the default preprocessing dir, only use the basename as dataset name
    if os.path.dirname(corpus_dir) == PREPROCESSING_DIR.rstrip("/"):
        corpus_dir = os.path.basename(corpus_dir)

    if corpus_dir == "VCTK_uk_p228_with_us_g2p_mix_emb":
        return torch.LongTensor([0])
    elif corpus_dir == "VCTK_uk_p229_with_us_g2p_mix_emb":
        return torch.LongTensor([1])
    elif corpus_dir == "VCTK_us_p299_with_us_g2p_mix_emb":
        return torch.LongTensor([2])
    elif corpus_dir == "VCTK_us_p300_with_us_g2p_mix_emb":
        return torch.LongTensor([3])
    elif corpus_dir == "VCTK_uk_p228_with_uk_g2p_mix_emb":
        return torch.LongTensor([4])
    elif corpus_dir == "VCTK_uk_p229_with_uk_g2p_mix_emb":
        return torch.LongTensor([5])
    elif corpus_dir == "VCTK_us_p299_with_uk_g2p_mix_emb":
        return torch.LongTensor([6])
    elif corpus_dir == "VCTK_us_p300_with_uk_g2p_mix_emb":
        return torch.LongTensor([7])
    elif corpus_dir == "VCTK_uk_p228_with_us_g2p":
        return torch.LongTensor([8])
    elif corpus_dir == "VCTK_uk_p229_with_us_g2p":
        return torch.LongTensor([9])
    elif corpus_dir == "VCTK_us_p299_with_us_g2p":
        return torch.LongTensor([10])
    elif corpus_dir == "VCTK_us_p300_with_us_g2p":
        return torch.LongTensor([11])
    elif corpus_dir == "VCTK_uk_p228_with_uk_g2p":
        return torch.LongTensor([12])
    elif corpus_dir == "VCTK_uk_p229_with_uk_g2p":
        return torch.LongTensor([13])
    elif corpus_dir == "VCTK_us_p299_with_uk_g2p":
        return torch.LongTensor([14])
    elif corpus_dir == "VCTK_us_p300_with_uk_g2p":
        return torch.LongTensor([15])
    elif corpus_dir == "HUI_bernd_335":
        return torch.LongTensor([16])
    elif corpus_dir == "HUI_julia_320":
        return torch.LongTensor([17])
    elif corpus_dir == "HUI_karlsson_324":
        return torch.LongTensor([18])
    elif corpus_dir == "HUI_availle_321":
        return torch.LongTensor([19])
    raise ValueError(f"corpus_dir not mapped to a dataset_id: {corpus_dir}")
