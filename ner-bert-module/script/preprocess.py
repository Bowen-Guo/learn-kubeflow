from __future__ import absolute_import, division, print_function
import pyarrow.parquet as pq
import faulthandler
import logging
import os
import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .arg_opts import preprocess_opts


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
faulthandler.enable()
logging.info(f"Load pyarrow.parquet explicitly: {pq}")

logger = logging.getLogger(__name__)


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


def read_parquet(data_path):
    """

    :param file_name: str,
    :return: pandas.DataFrame
    """
    logger.info("start reading parquet.")
    df = pd.read_parquet(os.path.join(data_path, 'data.dataset.parquet'), engine='pyarrow')
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
    :return: data is a list of tuples,
    item format (['EU rejects German call to boycott British lamb .'],
                 ['B-ORG O B-MISC O O O B-MISC O O'])
    """
    data = []

    header_names = set(list(data_frame.columns.values))
    global no_label
    no_label = "Label" not in header_names

    for index, line in data_frame.iterrows():
        sentence = line['Text']
        if not no_label:
            label = line['Label']
        else:
            label = ""
        data.append((sentence, label))
    return data


class NerProcessor:
    """Processor for the CoNLL-2003 data set."""

    def get_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_dataframe(data_path))

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_examples(self, lines):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%d" % i
            text_a = sentence
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _read_dataframe(data_path):
        return read_dataframe(read_parquet(data_path))


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_list):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    raw_text_list = []
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    label_ids_list = []
    valid_positions_list = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        positions = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    positions.append(1)
                else:
                    labels.append("X")
                    positions.append(-1)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            positions = positions[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        valid_positions = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        valid_positions.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            valid_positions.append(positions[i])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        valid_positions.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid_positions.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid_positions) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s " % " ".join([str(x) for x in label_ids]))
            logger.info("valid_positions: %s " % " ".join([str(x) for x in valid_positions]))

        raw_text_list.append(example.text_a)
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
        segment_ids_list.append(segment_ids)
        label_ids_list.append(label_ids)
        valid_positions_list.append(valid_positions)

    df_feature = pd.DataFrame({'raw_text': raw_text_list, 'input_ids': input_ids_list,
                               'input_mask': input_mask_list, 'segment_ids': segment_ids_list,
                               'label_ids': label_ids_list, 'valid_positions': valid_positions_list})

    return df_feature


def convert_examples_to_features_no_label(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    raw_text_list = []
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    valid_positions_list = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        tokens = []
        positions = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    positions.append(1)
                else:
                    positions.append(-1)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            positions = positions[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        valid_positions = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid_positions.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            valid_positions.append(positions[i])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid_positions.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(valid_positions) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("valid_positions: %s " % " ".join([str(x) for x in valid_positions]))

        raw_text_list.append(example.text_a)
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
        segment_ids_list.append(segment_ids)
        valid_positions_list.append(valid_positions)

    df_feature = pd.DataFrame({'raw_text': raw_text_list, 'input_ids': input_ids_list,
                               'input_mask': input_mask_list, 'segment_ids': segment_ids_list,
                               'valid_positions': valid_positions_list})

    return df_feature


def main():
    parser = preprocess_opts()
    args, _ = parser.parse_known_args()

    processor = NerProcessor()
    label_list = processor.get_labels()
    examples = processor.get_examples(args.input_data_frame_path)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    df_features = convert_examples_to_features_no_label(
        examples, args.max_seq_length, tokenizer) if no_label else convert_examples_to_features(
        examples, args.max_seq_length, tokenizer, label_list)

    if not os.path.exists(args.output_feature_dir):
        os.makedirs(args.output_feature_dir)
    df_features.to_parquet(fname=os.path.join(args.output_feature_dir, "feature.parquet"), engine='pyarrow')

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "Dataset",
        "Name": "Dataset .NET file",
        "ShortName": "Dataset",
        "Description": "A serialized DataTable supporting partial reads and writes",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "dataset.parquet",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": True,
        "AllowModelPromotion": False,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(args.output_feature_dir, 'data_type.json'), 'w') as f:
        json.dump(dct, f)

if __name__ == "__main__":
    main()
