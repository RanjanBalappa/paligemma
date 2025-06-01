from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from image_processor import process_images

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(
    prefix_prompt: str,
    bos_token: str,
    image_seq_len: int,
    image_token: str,
):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaligemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer

    def __call_(
            self,
            text: List[str],
            images: List[Image.Image],
            padding: str = "longest",
            truncation: bool = False,
    ):
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        pixel_values = torch.stack(pixel_values, axis=0) #[batch_size, num_channels, image_size, image_size]
        pixel_values = Tensor(pixel_values)


        #image_tokens + prompts
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        input_ids = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return {"pixel_values": pixel_values, **input_ids}




       
        