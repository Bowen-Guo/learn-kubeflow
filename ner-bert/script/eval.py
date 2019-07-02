from __future__ import absolute_import, division, print_function
import pyarrow.parquet as pq
import faulthandler
import logging
import pandas as pd
import json
import os
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm
from arg_opts import eval_opts
from pytorch_pretrained_bert.tokenization import BertTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
faulthandler.enable()
logging.info(f"Load pyarrow.parquet explicitly: {pq}")
logger = logging.getLogger(__name__)


class Ner:

    def __init__(self, model_dir: str):
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        self.model.eval()

    @staticmethod
    def load_model(model_dir, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir, model_config)
        model_config = json.load(open(model_config))
        output_config_file = os.path.join(model_dir, CONFIG_NAME)
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertForTokenClassification(config, num_labels=model_config["num_labels"])
        model.load_state_dict(torch.load(output_model_file))
        tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"], do_lower_case=False)
        return model, tokenizer, model_config


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def read_parquet(file_name):
    logger.info("start reading parquet.")
    df = pd.read_parquet(os.path.join(file_name, 'data.dataset.parquet'), engine='pyarrow')
    logger.info("parquet read completed.")
    return df


def read_dataframe(data_frame: pd.DataFrame):
    """
    read data from pandas.DataFrame
    :param data: panda.DataFrame, sample format
                                                    Text                                              Label
    0  SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRI...                    O O B-LOC O O O O B-PER O O O O
    1                                        Nadim Ladki                                        B-PER I-PER
    2           AL-AIN , United Arab Emirates 1996-12-06                        B-LOC O B-LOC I-LOC I-LOC O
    3  Japan began the defence of their Asian Cup tit...  B-LOC O O O O O B-MISC I-MISC O O O O O O O B-...
    4  But China saw their luck desert them in the se...  O B-LOC O O O O O O O O O O O O O O O O O O O ...
    :return: data_bak is a list of tuples,
    item format (['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
                 ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O'])
    """
    data = []

    for index, line in data_frame.iterrows():
        sentence = line['Text']
        label = line['Label']
        data.append((sentence, label))
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data_bak set."""
        raise NotImplementedError()

    @staticmethod
    def _read_dataframe(data_path):
        """Reads a tab separated value file."""
        return read_dataframe(read_parquet(data_path))


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_dataframe(data_path), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_dataframe(data_path), "dev")

    def get_test_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_dataframe(data_path), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data_bak file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids))
    return features


def main():
    parser = eval_opts()
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    if os.path.exists(args.output_eval_dir) and os.listdir(args.output_eval_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_eval_dir))
    if not os.path.exists(args.output_eval_dir):
        os.makedirs(args.output_eval_dir)

    # Load a trained model and config that you have fine-tuned
    model_class = Ner(args.trained_model_dir)
    tokenizer = model_class.tokenizer
    model = model_class.model
    model.to(device)

    processor = NerProcessor()
    label_list = processor.get_labels()
    eval_examples = processor.get_test_examples(args.test_data_path)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for test data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        for i, mask in enumerate(input_mask):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids[i][j]] != "X":
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
                else:
                    temp_1.pop()
                    temp_2.pop()
                    break
            y_true.append(temp_1)
            y_pred.append(temp_2)
    report = classification_report(y_true, y_pred, digits=4)
    output_eval_file = os.path.join(args.output_eval_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("\n%s", report)
        writer.write(report)


if __name__ == "__main__":
    main()
