import torch
from torchvision import transforms
import re

def preprocess_question(question, sos_token, eos_token):
    question = re.sub("\"", "", question)
    question = question.lower().strip().split()
    return [sos_token] + question + [eos_token]

def preprocess_answer(answer):
    answer = re.sub("\"", "", answer)
    answer = re.sub(" ", "_", answer.strip()).lower()
    return answer

def get_transform(target_size):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def reporthook(t):
    """
    https://github.com/tqdm/tqdm.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner