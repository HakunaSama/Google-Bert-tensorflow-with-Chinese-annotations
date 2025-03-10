# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## 必须的参数
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# 输入样本类，这个类只是简单的封装了以下样本索引
class InputExample(object):
  """这是一个用于简单序列分类的单个训练/测试样本。"""
  def __init__(self, guid, text_a, text_b=None, label=None):
    """构造函数，构造样本.
    参数:
      guid: 样本索引.
      text_a: string. 未分词的第一个句子
      text_b: (Optional) string. 未分词的第二个序列，有的任务是没有第二个句子的.
      label: (Optional) string. 样本标签，训练和验证样本有，测试样本没有哦
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

# 输入特征类，和InputExample做区分哦，InputExample里就是存没处理的字符串，InputFeatures存处理过的id、mask、typeid等等
class InputFeatures(object):
  """一个数据特征的结构体"""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

# 数据处理的基类
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

#### step1：首先我们需要先学习数据处理这一模块
# 这个是我们用的数据处理器，其他的数据处理器先不看，
# 遇到其他任务的时候，可能要用到其他数据处理器，或者自己定义一个数据处理器
# 这个处理器的功能就是将纯文本数据变成id+第一句话+第二句话+标签的结构化数据
class MrpcProcessor(DataProcessor):
  """MRPC数据集的数据处理器(GLUE version)."""

  def get_train_examples(self, data_dir):
    """见基类."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """见基类."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """见基类."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """见基类."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """创建训练集和验证集的示例"""
    examples = []
    # 遍历数据所有的行
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])#获取第一句话
      text_b = tokenization.convert_to_unicode(line[4])#获取第二句话
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])#获取标签
      examples.append(#将数据处理成example后用列表储存起来
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

# 将单个样本转换成特征函数
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """将单个样本转成单个输入特征.
  参数：
    ex_index：样本的索引
    example：样本
    label_list：样本可能有的标签值
    max_seq_length：样本的最大序列长度
    tokenizer：处理样本用的分词器
  """

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}# 存放标签的字典
  # 遍历所有的lable
  for (i, label) in enumerate(label_list): #构建标签，也就两个标签{‘0’：0，‘1’：1}
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a) #第一句话分词，tokens_a是切分后的字符串列表
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b) #如果有第二句话，也进行一样的分词操作

  if tokens_b:
    # 就地裁剪 tokens_a 和 tokens_b，使它们的总长度小于指定长度。
    # [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    # 考虑到 [CLS]、[SEP]、[SEP] 这三个特殊标记，因此需要减去 3。 #保留3个特殊字符
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3) #如果这俩太长了就截断操作
  else:
    # [CLS] the dog is hairy . [SEP]
    # 考虑到 [CLS]、[SEP] 这两个个特殊标记，因此需要减去 2。
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # 在 BERT 中的约定是：
  # (a) 对于句子对：
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1  # 表示来自哪句话
  # (b) 对于单个序列：
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # 其中 "type_ids" 用于指示该 token 属于第一句还是第二句。
  # 对于 type=0 和 type=1 的嵌入向量在预训练期间就已经学习好，
  # 并被加到词片段嵌入向量（以及位置嵌入向量）上。
  # 这并非严格必要，因为 [SEP] 标记可以明确地分隔句子，
  # 但它使模型更容易学习句子这一概念。
  #
  # 对于分类任务，第一个向量（对应 [CLS]）被用作“句子向量”。
  # 请注意，这只有在整个模型经过微调的情况下才有意义。

  # 现在我们已经有了切分好的两个或者一个句子的token，长度满足要求，但是还没有特殊标记，
  # 现在要做一个带有标记并且完整的tokens及其ids
  tokens = []# 初始化 tokens 和 segment_ids 的列表，用于存储最终的 token 序列和对应的哪个句子的标记，也就是token_type_ids
  segment_ids = []
  tokens.append("[CLS]")#两个都加上特殊标记
  segment_ids.append(0)#特殊标记属于第一句话
  for token in tokens_a:#把第一句话的token一个一个加进来
    tokens.append(token)
    segment_ids.append(0)#第一句话的标记是0
  tokens.append("[SEP]")#再加一个隔断特殊标记
  segment_ids.append(0)#隔断特殊标记也是属于第一句话的

  if tokens_b:#这就不详细说了，跟上面一样的
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  # 现在有了一个带有标记并且完整的tokens及带有特殊标记的id，但是普通token的id还没有，
  # 现在把普通token也转成id吧，为什么需要转成id呢，因为做词嵌入的时候，通过词直接embeddinglookup很麻烦，
  # 但是使用id去寻找就很简单，只需要把id变成onehot编码，再乘以embeddinglookup查找表即可
  input_ids = tokenizer.convert_tokens_to_ids(tokens) #转换成ID

  # mask矩阵为1的位置说明是真实token，为0说明是补全的token
  # 只有真实token需要去进行注意
  # 先搞一个跟input_ids一样长的全1列表
  input_mask = [1] * len(input_ids) #由于后续可能会有补齐操作，设置了一个mask目的是让attention只能放到mask为1的位置

  # 用0填充到规定最大序列长度
  while len(input_ids) < max_seq_length: #PAD的长度取决于设置的最大长度
    input_ids.append(0)#输入id要填充
    input_mask.append(0)#mask也要填充
    segment_ids.append(0)#token_type_ids也要填充,但是这里的填充是没有意义的，因为这些填充并不属于第二句话，但是又要操作保证后续的运算

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  #好了，到此我们获得了一句话每个token的id，mask和属于哪句话，0或者1，也就是token_type

  # 获取这个数据的标签，从label_map映射表里找就行
  label_id = label_map[example.label]
  if ex_index < 5: #打印一些例子
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
  # 构建当前样本的输入特征，相比较与InputExample，InputFeatures存的不只是两句话的字符串了，而是处理好的tokenid、mask、typeid这些信息
  feature = InputFeatures(
      input_ids=input_ids,#样本的token的id
      input_mask=input_mask,#样本的token的mask
      segment_ids=segment_ids,#样本的token的type，就是属于哪句话
      label_id=label_id,#样本的标签
      is_real_example=True)#是否是真实样本
  return feature


# 基于文件将输入样本转成特征的函数
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """将一组 `InputExample` 转换为 TFRecord 文件."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):# 遍历3688个样本
    if ex_index % 10000 == 0:#每隔10000次打印一下处理结果，我们不够10000次，不会显示的
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    #核心函数，转换单个样本为特征，这个函数要点进去看，Example和Feature的区别是Feature是处理过的，有id、mask、typeid等，样本就是字符串
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    # 这里就是全部转成int而已，并且存入到features字典里，这里其实感觉有点多此一举，直接在InputFeatures里转好返回字典应该也可以
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    # 基本操作，把数据做成tfRecoder了，我不太懂tensorflow，但是tfRecoder的作用相当于dataloader
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

#这个函数用于构建一个“输入函数”（input_fn），这个输入函数将传递给 TPUEstimator，用于读取和处理 TFRecord 格式的数据。
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """创建一个传递给 TPUEstimator 的 input_fn 闭包。"""

  name_to_features = {# 定义了 TFRecord 中每个样本的特征及其类型和形状，例如 "input_ids"、"input_mask"、"segment_ids" 等。这些定义告诉 TensorFlow 如何解析每条记录。
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """将一条记录解码为 TensorFlow 样本。"""
    #使用 tf.parse_single_example 根据 name_to_features 来解析记录。
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example 只支持 tf.int64，但是 TPU 只支持 tf.int32。
    # 因此，需要将所有 int64 类型转换为 int32 类型。
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):#接受一个 params 字典（其中包含 batch_size），用于设置批量大小。
    """实际的输入函数。"""
    batch_size = params["batch_size"]

    # 对于训练，我们希望大量并行读取数据并进行打乱。
    # 对于评估，则不需要打乱数据，并且并行读取的要求也不那么严格。
    d = tf.data.TFRecordDataset(input_file)#使用 tf.data.TFRecordDataset 读取指定的 input_file。
    if is_training:#如果处于训练模式（is_training 为 True），则对数据集进行无限重复（repeat）和打乱（shuffle），以增加训练的随机性。
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn#返回这个 input_fn 闭包，供 TPUEstimator 在训练或评估时调用，从而构建出数据输入管道。


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

#####这里和pytorch太不一样了，以至于很容易迷惑，pytorch是定义好每个层，在定义前向传播函数
# 这个函数实际上既在“创建”模型，也在定义模型的前向传播过程。这里的模型创建和前向传播是在TensorFlow 的静态计算图（graph）中完成的，而不是像 PyTorch 那样定义一个对象再调用 forward 方法。
# 详细解释：
#   1.计算图的构建
#       在 TensorFlow 1.x 中，模型不是一个运行时对象，而是一系列操作（ops）构成的计算图。create_model 函数会创建并添加这些操作到计算图中。
#       这里的每个操作（比如调用 modeling.BertModel、tf.nn.dropout、tf.matmul、tf.nn.softmax 等）都只是图中的一个节点，只有在 Session 运行时才会真正计算。
#   2.模型创建与前向传播
#       模型创建：调用 modeling.BertModel 实际上会根据 bert_config 创建 BERT 模型的各个层（比如嵌入层、Transformer 层等），并将这些层及其参数（变量）加入计算图中。
#       前向传播：接下来，通过 model.get_pooled_output() 取到对应 [CLS] 的表示，再接一个 dropout 层和一个全连接层（线性变换、加偏置），最后计算 softmax 得到预测概率，以及计算交叉熵损失。整个过程定义了输入数据经过各层转换后如何输出最终的预测和损失。
#   3.与 PyTorch 的区别
#       PyTorch：模型一般写成一个类，继承自 nn.Module，里面的 forward 方法定义了前向传播逻辑，执行时会动态构建计算过程（动态图模式）。
#       TensorFlow 1.x：则是在定义阶段（模型构建阶段）就构建好了整个计算图，所有的前向传播步骤都是图中的节点，之后需要通过 Session 运行这些节点得到输出。函数看起来像是“直接计算”结果，但实际上只是定义了如何计算。
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """创建一个分类模型."""
  #### 什么都别管，先创建模型再说，请记住这里不仅仅是创建模型，还对当前的batch进行了前向传播
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,#（8,128），每一个值存的是token的id，8是指8个batch，128是指每个序列最大长度为128，不够的填充，多的截断
      input_mask=input_mask,#（8,128），被填充的位置为0，真实token的位置为1
      token_type_ids=segment_ids,#（8,128），每一个值存的是位置的类型，0代表属于第一句话，1代表属于第二句话
      use_one_hot_embeddings=use_one_hot_embeddings)

  # 在演示中，我们对整个片段（segment）执行一个简单的分类任务，也就是获取到了<cls>的表示。
  # 如果你想使用token 级别的输出，请使用 model.get_sequence_output()。
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value# 768 cls向量的维度，也是768，和词嵌入一个维度

  # 定义用于全连接层的权重变量，将 CLS 嵌入输入映射到每个标签对应的维度
  # 这里输出维度为 [num_labels, hidden_size]，例如二分类时 num_labels=2，hidden_size=768
  output_weights = tf.get_variable(# 将cls嵌入输入到一个全连接层
      "output_weights", [num_labels, hidden_size],# num_labels=2，2分类，hidden_size=768
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  # 定义全连接层的偏置变量，维度为 [num_labels]
  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # 如果处于训练阶段，则对输出层应用 dropout（保留比例 0.9，即 dropout rate 为 0.1）
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    # 计算 logits，即未归一化的预测值
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)# 通过矩阵乘法将输出层与转置后的权重矩阵相乘
    logits = tf.nn.bias_add(logits, output_bias)# 加上偏置项
    probabilities = tf.nn.softmax(logits, axis=-1)# 计算预测的概率分布（softmax 在最后一维上进行归一化）
    log_probs = tf.nn.log_softmax(logits, axis=-1)# 计算 log softmax，用于后续交叉熵损失的计算

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)# 将标签转换为 one-hot 编码，深度为 num_labels，数据类型为 float32

    # 计算每个样本的交叉熵损失
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)# 对于每个样本，计算 one_hot_labels 与 log_probs 的乘积后求和，再取负值
    loss = tf.reduce_mean(per_example_loss)# 计算所有样本的平均损失

    # 返回总损失、每个样本的损失、logits 以及预测的概率分布
    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### 终于到了非常非常重要的创建模型的阶段，点进去看，这里只是定义了模型还没开始训练
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    # 获取当前图中所有需要训练的变量，当前图是被create_model这个函数创建出来的，这里和pytorch非常不一样，有点难理解
    tvars = tf.trainable_variables()
    # 初始化一个空字典，用于存储从 checkpoint 中初始化的变量名称
    initialized_variable_names = {}
    scaffold_fn = None# scaffold_fn 用于 TPU 模式下的构建辅助，初始设为 None
    # 如果提供了预训练的 checkpoint，则进行变量映射和初始化
    if init_checkpoint:
      # 从预训练的 checkpoint 中获取变量映射关系 assignment_map 和已初始化变量的名称列表
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:# 如果使用 TPU，则需要定义一个 scaffold 函数，这里不用管

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        # 如果不使用 TPU，则直接将预训练的 checkpoint 中的变量值初始化到当前图中的变量，这里就是加载预训练模型参数的步骤，非常重要哦
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 输出所有可训练变量的信息
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      # 如果变量从 checkpoint 中初始化，则记录标记信息
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      # 打印变量的名称、形状以及是否从 checkpoint 中初始化的标记
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    # 定义返回的输出规范，初始设为 None
    output_spec = None
    # 根据当前模式（训练、评估或预测）构建不同的 Estimator 规范
    if mode == tf.estimator.ModeKeys.TRAIN:
      # 如果是训练模式，创建训练操作（train_op）
      train_op = optimization.create_optimizer(
          total_loss,       # 总损失
          learning_rate,    # 学习率
          num_train_steps,  # 总训练步数
          num_warmup_steps, # 预热步数
          use_tpu)          # 是否使用 TPU

      # 为 TPU 训练创建 TPUEstimatorSpec，包括模式、损失、训练操作以及 scaffold_fn
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
       # 如果是评估模式，定义评估指标函数
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)# 根据 logits 得到预测类别
        accuracy = tf.metrics.accuracy(# 计算准确率，权重为 is_real_example（用于忽略 padding 部分等无效样本）
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)# 计算平均损失
        return {# 返回评估指标字典
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,# 将评估指标函数和相关张量打包为元组
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(# 为 TPU 评估创建 TPUEstimatorSpec，包括模式、损失、评估指标以及 scaffold_fn
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      # 对于预测模式，构建包含预测结果（这里是概率）的 TPUEstimatorSpec
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec# 返回构建好的 TPUEstimatorSpec，供 Estimator 使用

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """将输入数据的集合转成输入特征的列表."""

  features = []
  for (ex_index, example) in enumerate(examples):# 遍历所有的样本
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,# 将改样本转成特征
                                     max_seq_length, tokenizer)

    features.append(feature)
  # 现在我们拥有所有输入特征的列表了
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
  # 首先通过命令行参数bert_config.json文件加载配置
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  # 如果命令行参数的最大序列长度大于配置文件中的最大位置编码长度，要报错，不能使用max_seq_length序列长度，因为Bert模型只被训练到max_position_embeddings序列长度
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))
  # 创建命令行参数中的输出文件夹
  tf.gfile.MakeDirs(FLAGS.output_dir)

  # 根据任务名来获取对应的参数处理器processor
  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  # 分词器这里使用的是结合了Basic和WordPiece分词器的全分词器，可以将词分成子词，在当时已经是很好的选择了，现在不知
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  # 这一部分就可以先不用关心了
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # 如果命令行参数说要训练
  if FLAGS.do_train:
    # 把数据一行一行读进来变成样本，仅此而以，见MRPCProcessor类的get_train_examples函数，3668行数据
    # 这里的train_examples是InputExample类型的列表
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    # 迭代次数=3668/batch_size*epochs
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    # 学习率先小后大
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  #创建模型函数，tensorflow中模型是一个模型函数，这个模型函数包含了模型的前向传播、损失计算等逻辑，之后可以直接传递给 TensorFlow 的 Estimator 用于训练、评估或预测。
  model_fn = model_fn_builder(#根据传入的参数（例如 BERT 的配置、学习率、标签数量等），构建并返回一个模型函数（model_fn）
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,#是否使用 TPU 来加速训练
      use_one_hot_embeddings=FLAGS.use_tpu)#是否在嵌入层使用 one-hot 编码，这里通常在使用 TPU 时会设为 True，

  # 如果没有 TPU 可用，则会自动回退到在 CPU 或 GPU 上运行的普通 Estimator。我的会到gpu上
  # 为模型创建estimator，可以这么理解，这个以后就是我们模型的代理对象了，训练评估等都由它来执行
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # 这里是真正的数据预处理的模块
  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    # 把训练样本转成特征，这里不仅是转成特征了，还写到tfRecoder里了，这个函数需要点进去看
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    # 输出一些东西看
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(#构建一个“输入函数”，类似于dataloader，自动给模型输入数据
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    # 开始用estimator进行训练吧！！！
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    # 把数据一行一行读进来变成样本，仅此而以，见MRPCProcessor类的get_dev_examples函数
    # 这里的eval_examples是InputExample类型的列表
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)#验证集样本数
    if FLAGS.use_tpu:
      # TPU 要求所有批次的批量大小都固定，因此样本数量必须是批量大小的整数倍，
      # 否则多余的样本将被丢弃。
      # 为了解决这个问题，我们会用虚假的样本对批次进行填充，
      # 这些虚假样本在后续计算中会被忽略，
      # 并且它们不会被计入指标计算（所有 tf.metrics 都支持为每个样本指定权重，而这些虚假样本的权重设置为 0.0）。
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())
    #验证集路径拼接
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    # 把验证样本转成特征，这里不仅是转成特征了，还写到tfRecoder里了
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    # 输出一些东西看
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # 这告诉 Estimator 遍历整个数据集。
    eval_steps = None
    # 但是，如果在 TPU 上进行评估，就需要指定评估的步数。
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")#数据文件路径
  flags.mark_flag_as_required("task_name")#任务名
  flags.mark_flag_as_required("vocab_file")#语料表
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
