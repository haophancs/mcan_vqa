import torch
from transformers import AutoTokenizer
from data_utils.utils import preprocess_answer, preprocess_question
from collections import defaultdict, Counter
import logging
import six
import os
import json

logger = logging.getLogger(__name__)


def _default_unk_index():
    return 0


class VQAVocab(object):

    def __init__(self, json_prefixes, pretrained_tokenizer_name):
        self.make_vocab(json_prefixes, pretrained_tokenizer_name)

    def make_vocab(self, json_path_prefixes, pretrained_tokenizer_name):
        self.max_question_length = 0
        self.output_cats = set()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
        self.pad_token = self.tokenizer.pad_token
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}
        for json_path_prefix in json_path_prefixes:
            question_data = json.load(open(json_path_prefix + 'questions.json'))
            annotation_data = json.load(open(json_path_prefix + 'annotations.json'))
            for q_item, a_item in zip(question_data["questions"], annotation_data["annotations"]):
                question = self.tokenizer(
                    preprocess_question(
                        q_item["question"],
                        self.bos_token,
                        self.eos_token
                    ))["input_ids"]
                answer = preprocess_answer(a_item["multiple_choice_answer"])
                self.output_cats.add(answer)
                if len(question) > self.max_question_length:
                    self.max_question_length = len(question)

        self.output_cats = list(self.output_cats)


    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.ones(self.max_question_length).long() * self.stoi[self.pad_token]
        for i, token in enumerate(question):
            vec[i] = self.stoi[token]
        return vec

    def _encode_answer(self, answer):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.output_cats))
        answer_vec[self.output_cats.index(answer)] = 1

        return answer_vec

    def _decode_question(self, question_vecs):
        questions = []
        for vec in question_vecs:
            questions.append(" ".join([self.itos[idx] for idx in vec.tolist() if idx > 0]))

        return questions

    def _decode_answer(self, predicted):
        predicted = torch.argmax(predicted, dim=-1).tolist()
        answers = []
        for idx in predicted:
            answers.append(self.output_cats[idx])

        return answers

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)
