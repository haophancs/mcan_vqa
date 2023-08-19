import torch
from torch.utils import data
from torch.utils.data.dataset import random_split
from data_utils.utils import preprocess_question, preprocess_answer
from data_utils.vqa_vocab import VQAVocab
import h5py
import json
import config


class VQA(data.Dataset):
    """ VQA dataset, open-ended """

    def __init__(self, json_path_prefix, image_features_path):
        super(VQA, self).__init__()
        # vocab
        self.vocab = VQAVocab([json_path_prefix], pretrained_tokenizer_name=config.pretrained_text_model)

        # q and a
        self.questions, self.answers, self.image_ids = self.load_json(json_path_prefix)

        # v
        self.image_features_path = image_features_path
        self.image_id_to_index = self._create_image_id_to_index()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions)) + 2
        return self._max_length + 2

    @property
    def num_tokens(self):
        return len(self.vocab.stoi)

    def load_json(self, json_path_prefix):
        questions = []
        answers = []
        image_ids = []
        question_data = json.load(open(json_path_prefix + 'questions.json'))
        annotation_data = json.load(open(json_path_prefix + 'annotations.json'))
        for q_item, a_item in zip(question_data["questions"], annotation_data["annotations"]):
            assert q_item['question_id'] == a_item['question_id']
            questions.append(preprocess_question(q_item["question"]))
            answers.append(preprocess_answer(a_item["multiple_choice_answer"]))
            image_ids.append(a_item["image_id"])

        return questions, answers, image_ids

    def _create_image_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.image_id_to_index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')

        return torch.from_numpy(img)

    def __getitem__(self, idx):
        q = self.vocab._encode_question(self.questions[idx])
        a = self.vocab._encode_answer(self.answers[idx])
        image_id = self.image_ids[idx]
        v = self._load_image(image_id)

        return v, q, a

    def __len__(self):
        return len(self.questions)


def get_loader(train_dataset, test_dataset=None):
    """ Returns a data loader for the desired split """

    fold_size = int(len(train_dataset) * 0.2)

    subdatasets = random_split(train_dataset,
                               [fold_size, fold_size, fold_size, fold_size, len(train_dataset) - fold_size * 4],
                               generator=torch.Generator().manual_seed(13))

    folds = []
    for subdataset in subdatasets:
        folds.append(
            torch.utils.data.DataLoader(
                subdataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=config.data_workers))

    if test_dataset:
        test_fold = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=config.data_workers)

        return folds, test_fold

    return folds
