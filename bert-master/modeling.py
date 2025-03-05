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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
  """`BertModel`配置类."""

  def __init__(self,
               vocab_size,#语料库大小
               hidden_size=768,#encoder层的大小
               num_hidden_layers=12,#
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",#非线性激活函数
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """构建BertConfig.

    参数:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.语料库的大小
      hidden_size: Size of the encoder layers and the pooler layer.encoder层的大小
      num_hidden_layers: Number of hidden layers in the Transformer encoder.隐藏层的数量
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.多头注意力的头数
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.非线性激活函数
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

#模型定义，我们要用这个对象来创建我们的模型
class BertModel(object):
  """BERT 模型（"Bidirectional Encoder Representations from Transformers"，双向编码器表示的变换器）。.
  使用示例:
  ```python
  # 已经转换为 WordPiece 词片 ID
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  #注意 transformer中的hiddensize的大小代表的是词嵌入的维度
  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """
####额外说一句，这里的模型定义其实已经做好了模型的前向传播工作，里面的一些值看似是在初始化，但是其实已经完成计算了
  #也就是说，在完成了构造函数之后，模型的前向传播已经做好了，这个和pytorch的做法还蛮不一样，这个观点在后面看来是错的，这即定义了网络结构，也定义了前向传播路径，但是没有真的进行前向传播
  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """BertModel的构造函数.

    Args:
      config: `BertConfig` 对象.
      is_training: bool. 为true训练模型, 为false评估模型. 控制是否使用随机失活
      input_ids: int32 张量[batch_size, seq_length].每个值代表token在语料表中的id
      input_mask: (可选) int32张量[batch_size, seq_length].
      token_type_ids: (可选) int32 张量 [batch_size, seq_length]，每个值代表token的类型，属于第一句话或者第二句话.
      use_one_hot_embeddings: (可选) bool.是否使用onehot编码词嵌入还是使用tf的embeedinglookup词嵌入
      scope: (可选) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """

    ####首先我们先来做一下初始化，比如失活要不要启动、batchsize、序列长度获取一下、mask和tokentype设置一下
    config = copy.deepcopy(config)#先拷贝一份config，防止后修改了原来的config对象
    if not is_training:#非训练的模型，隐藏层的失活率和注意力概率的失活率都要为0
      config.hidden_dropout_prob = 0.0#隐藏层失活，用于BERT的前馈神经网络。
      # 激活通常会应用 hidden_dropout_prob 来进行丢弃。丢弃是在激活值传递通过激活函数（如 ReLU）后执行的。
      config.attention_probs_dropout_prob = 0.0#注意力概率失活，在计算注意力分数时随机丢弃一些权重，当计算注意力分数时，
      # attention_probs_dropout_prob 会在注意力权重矩阵（attention_probs）上进行丢弃。

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]#获取输入的批次大小
    seq_length = input_shape[1]#获取每个批次的序列大小

    if input_mask is None: #如果没设置mask 自然就都是1的 都是真实token
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:#如果没有设置tokentype，那就都是0，都是第一句话的
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    ####首先第一步应该做好全部的嵌入工作，拿到真正要输入到模型里的词嵌入向量列表
    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # 对word的id执行embedding lookup.获取到word id的embedding向量，以及嵌入查找表，这个嵌入查找表在训练中会进行更新哦
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,#输入的id
            vocab_size=config.vocab_size,#词库的大小 30522
            embedding_size=config.hidden_size,#隐藏层的大小，决定了词嵌入后向量的维度，这里是768维度
            initializer_range=config.initializer_range,#初始化的嵌入值的范围
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)#是否使用onehot编码

        # 添加位置嵌入和token类型嵌入，然后执行层归一化和失活
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
    ####现在我们已经获取了要投入transformer中的向量，接下来我们要进行计算堆叠的transformer了
      # 创建一个encoder作用域，所有在with 代码块内创建的变量都会带上 "encoder" 这个前缀，避免变量命名冲突。
      with tf.variable_scope("encoder"):
        # 这将一个形状为 [batch_size, seq_length] 的 2D 掩码（mask）input_mask 转换为 3D 掩码（mask）attention_mask，
        # 其形状为 [batch_size, seq_length, seq_length]，用于计算注意力分数（attention scores）。
        # 假设当前batch是[seq_length]的列表，那么掩码为[seq_length,seq_length],那么掩码的[i,j]表示i要不要对j进行attention
        # 实际上对于掩码而言，第二维度的数据都是一样的
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # 运行堆叠的 Transformer.
        # `sequence_output` 的形状为 = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,#之前embedding的结果
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,#hidden可以理解为最终希望得到特征的维度
            num_hidden_layers=config.num_hidden_layers,#Transformer中的隐层神经元个数
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,#全连接层神经元个数
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)#是否返回每一层的输出
    ####到这里我们就把编码器全部计算完了，得到了我们的最终输出的向量
      self.sequence_output = self.all_encoder_layers[-1]
      # “pooler”将编码后的序列张量从形状
      # [batch_size, seq_length, hidden_size] 到形状
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        #我们通过简单地取第一个标记（token）对应的隐藏状态来对模型进行“池化”（pooling）。
        # 我们假设该方法已经经过预训练。
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))
      ##### 到这里，我们创建了一个模型，并且进行了前向传播，保存了每一步的输出

  #获取池化后的输出，就是第一个token
  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """获取编码器的最终隐藏层。

    返回:
      一个 float 类型的张量，形状为 [batch_size, seq_length, hidden_size]，
      对应于 Transformer 编码器的最终隐藏层。
    """
    return self.sequence_output

  #获取所有编码器层
  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """获取嵌入查找（即 Transformer 的输入）的输出。

    返回:
      一个 float 类型的张量，形状为 [batch_size, seq_length, hidden_size]，对应于嵌入层的输出。
      该输出是在对**词嵌入（word embeddings）、位置嵌入（positional embeddings）和标记类型嵌入（token type embeddings）进行求和后，再执行层归一化（layer normalization）**的结果。
      这是 Transformer 的输入。
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


def gelu(x):
  """高斯误差线性单元（Gaussian Error Linear Unit, GELU）

  这是一种比 ReLU 更平滑的激活函数。
  原始论文：https://arxiv.org/abs/1606.08415
  参数:
    x: 需要进行激活的 float 类型张量。

  返回:
    对 x 应用 GELU 激活函数后的张量。
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """将字符串映射到 Python 函数，例如 "relu" => tf.nn.relu。

  参数:
    activation_string: 激活函数的字符串名称。

  返回:
    对应于该激活函数的 Python 函数。
    如果 activation_string 为空 (None、空字符串 "" 或 "linear")，则返回 None。
    如果 activation_string 不是字符串，则直接返回 activation_string。

  异常:
    ValueError：如果 activation_string 不对应于已知的激活函数，则抛出异常。
  """

  # 我们假设任何非字符串的输入已经是一个激活函数，因此直接返回它。
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """计算当前变量与检查点变量的并集。"""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """执行 Dropout 操作.

  参数:
    input_tensor: float 类型的张量
    dropout_prob: float 类型，表示被丢弃的概率（注意：不同于 tf.nn.dropout 这里指的是丢弃的概率，而不是保留的概率）。

  返回:
    经过 Dropout 处理后的 input_tensor 版本。
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output

#层归一化
def layer_norm(input_tensor, name=None):
  """ 对张量的最后一个维度执行层归一化（Layer Normalization）。"""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

#层归一化和随机失活层
def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """执行层归一化（Layer Normalization），然后应用 Dropout。"""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor

#初始化，默认范围是0.02
def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,##(8, 128)大小的存放id
                     vocab_size,#语料库大小，这里是30522，就是uncased_L-12_H-768_A-12这个语料库的大小，可以自己去查看一下
                     embedding_size=128,#词嵌入的维度，这里是768
                     initializer_range=0.02,#嵌入的初始化范围
                     word_embedding_name="word_embeddings",#词嵌入表的名字
                     use_one_hot_embeddings=False):#是否使用onehot编码，这里不使用
  """使用输入的id来lookup词嵌入

  Args:
    input_ids: int32 张量 [batch_size, seq_length] 包含word的id
    vocab_size: int. 嵌入词库的大小.
    embedding_size: int. 词嵌入的维度.
    initializer_range: float. 初始化的值范围.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. 如果True, 使用onehot编码. 如果False, 使用`tf.gather()`.

  Returns:
    float张量[batch_size, seq_length, embedding_size].每一个是word对应的词嵌入
  """
  # 假设输入张量的维度是[batch_size, seq_length,num_inputs].
  # 如果输入张量的维度是2D张量[batch_size, seq_length], 整形为[batch_size, seq_length, 1]
  if input_ids.shape.ndims == 2:
    # 如果输入只有两个维度,则要变成三个维度,即变成[batch_size, seq_length,num_inputs]
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  #创建随机的词嵌入查找表,语料库有30522个token,768的词向量维度
  embedding_table = tf.get_variable( #词映射矩阵，[30522,768]
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  # 将输入id张量[8, 128]张平为[1024]
  flat_input_ids = tf.reshape(input_ids, [-1])
  #如果使用onehot编码，就将张平后的输入id张量进行onehot编码[30522]，比如id是1就是[1,0,0,0,0,...],id是2就是[0，1，0，0，0...]
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)#对id进行onehot编码
    output = tf.matmul(one_hot_input_ids, embedding_table)#将onehot编码的id和嵌入查找表相乘，就可以获得对应的词嵌入。[1024,30522]*[30522,768]=[1024,768]
  else:
    output = tf.gather(embedding_table, flat_input_ids) #CPU,GPU运算1024, 768 一个batch里所有的映射结果

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size]) #现在再把我们的bacth给整形整出来(8, 128, 768)
  return (output, embedding_table)#返回词嵌入和词嵌入查找表(8, 128, 768)

#对于做了embedding之后的词嵌入做进一步处理，为了加入tokentype嵌入和位置嵌入
def embedding_postprocessor(input_tensor,#做了embedding之后的输出,大小为[batch_size, seq_length,embedding_size]
                            use_token_type=False,#是否要用typeid表示当前是第一句话还是第二句话
                            token_type_ids=None,#[batch_size, seq_length],0代表是第一句话的,1代表是第二句话的
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,#是否要加入位置嵌入
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,#初始化范围
                            max_position_embeddings=512,#位置嵌入的最大长度,只要比最大序列大即可
                            dropout_prob=0.1):
  """执行词嵌入张量的的后处理

  参数:
    input_tensor: float 张量 [batch_size, seq_length,embedding_size].
    use_token_type: bool. 是否加入token_type_id的嵌入
    token_type_ids: (可选) int32 张量 [batch_size, seq_length].
      必须被指定，如果 `use_token_type` 为 True.
    token_type_vocab_size: int. token_type_id的语料库大小.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. 是否添加position embeddings
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float.权重初始化范围.
    max_position_embeddings: int. 在这个模型中可能使用的最大序列长度，可能比最长的输入序列长度还长，但是不能更短
    dropout_prob: float. 应用在最终输出张量的失活率.

  返回值:
    返回值矩阵大小和input_tensor是相同的,只不过是加入了位置信息等等
    float 和 `input_tensor`大小相同的张量.

  异常:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  ##(8, 128, 768) 8个batch,每个batch128个token,每个token embedding768维度
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]   #8
  seq_length = input_shape[1]   #128
  width = input_shape[2]        #768

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:#[8,128]代表8个batch,每个batch有128个type_id，在这里是0和1
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    #创建tokentype查找表,2个随机编码，分别代表0和1,每个编码的维度是768,比如[0.1,0.2,0.3,0.4...0.768]
    #                                                            [1.1,1.2,1.3,1.4...1.768]
    # 注意这里的查找表和词嵌入查找表一样是可以在训练中更新的
    token_type_table = tf.get_variable(#[2,768]
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))#初始化值的范围
    # 因为这里token类型的词库很小，所以都是用onehot编码，不用gather，因为onehot编码在小语料库上总是更快
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])#[1024]8*128=1024 枚举出这个batch每一个token的type_id，这里的值是0和1
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)#[1024,2]对每一个token进行维度为2的onehot编码，也就是0和1变成了10和01
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)#用所有词嵌入的onehot编码和查找表相乘，就是查找onehot对应的嵌入[1024,2]*[2, 768]=[1024*768]
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width]) #8, 128, 768
    output += token_type_embeddings#在输出上加上token类型嵌入

  # 位置嵌入，这里不讲解，需要理解数学公式
  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(#[512*768],#位置最大是512，记得512要比序列长度长才可以
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1]) #[128, 768] 位置编码给的挺大，为了加速只需要取出有用部分就可以
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width]) # [1, 128, 768] 表示位置编码跟输入啥数据无关，因为原始的embedding是有batchsize当做第一个维度，这里为了计算也得加入
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  #完成所有嵌入操作之后的输出，再进行层归一化和随即失活
  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """从2d张量掩码中创建3d注意力掩码
  Args:
    from_tensor: 2D or 3D 张量 [batch_size, from_seq_length, ...].
    to_mask: int32 张量 [batch_size, to_seq_length].

  Returns:
    float 张量 [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # 我们不假设 from_tensor 是一个掩码（尽管它可以是）。实际上我们不关心是否  We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # 这里我们沿两个维度进行广播（broadcast）以创建掩码（mask）。
  mask = broadcast_ones * to_mask

  return mask

# 定义一下注意力计算层
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """执行多头注意力从 `from_tensor` 到 `to_tensor`.
    这是一个基于"Attention is all you Need"论文中的多头注意力机制，如果`from_tensor`和
    `to_tensor`是相同的，那这就是自注意力，from_tensor 中的每个时间步都会关注 to_tensor
    中对应的序列，并返回一个固定宽度的向量。

    这个函数首先将`from_tensor`映射为一个"query"，把`to_tensor`映射为"key" 和 "value"
    这实际上是一个长度为 num_attention_heads 的张量列表，其中每个张量的形状为
    [batch_size, seq_length, size_per_head]。[8,128,6]

    接下来，将 query 和 key 张量进行点积并缩放，然后经过 softmax 得到注意力概率。
    随后，用这些概率对 value 张量进行插值（加权相加），最后将结果拼接成一个单一的张量并返回。

    实际上，多头注意力机制的实现是通过转置和整形张量来完成的，而不是使用真正的独立头张量。


  Args:
    from_tensor: float 张量 [batch_size, from_seq_length,from_width].[8,128,768]
    to_tensor: float 张量 [batch_size, to_seq_length, to_width].[8,128,768]
    attention_mask: (可选) int32 张量 [batch_size,from_seq_length, to_seq_length].
    值应该是0或者1，在所有mask矩阵中为0的位置注意力分数将会被有效的置为无穷小，而在mask矩阵中为1的位置将会不变
    num_attention_heads: int. 注意力头的个数.
    size_per_head: int. 每个头的大小.
    query_act: (可选)  query 转换的激活函数.
    key_act: (可选)  key 转换的激活函数.
    value_act: (可选)  value 转换的激活函数.
    attention_probs_dropout_prob: (可选) float. 注意力概率的失活率.
    initializer_range: float. 权重初始化范围.
    do_return_2d_tensor: bool. 如果 True, 输出的大小将会是 [batch_size* from_seq_length, num_attention_heads * size_per_head].[8*128,6*size_per_head]
                               如果 False, 输出的大小将会是 [batch_size, from_seq_length, num_attention_heads* size_per_head].[8,128,6*size_per_head]
    batch_size: (可选) int. 如果输入为2D，则这可能是 from_tensor 和 to_tensor 的3D版本中的批次大小。
    from_seq_length: (可选) 如果输入为2D，则这可能是 from_tensor 的3D版本中的序列长度。
    to_seq_length: (可选) 如果输入为2D，则这可能是 to_tensor 的3D版本中的序列长度。

  Returns:
    float 张量 [batch_size, from_seq_length,num_attention_heads * size_per_head]. [8,128,6*size_per_head]
    (如果 `do_return_2d_tensor` 为true, 形状将会是
    [batch_size * from_seq_length,num_attention_heads * size_per_head]).[8*128,6*size_per_head]

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  # 将输入张量重塑为 [batch_size, seq_length, num_attention_heads, width]
  # 这里假设 input_tensor 原始形状为 [batch_size * seq_length, num_attention_heads * width][8*128,6*width]
  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])
    # 交换张量的维度，从而将形状变为 [batch_size, num_attention_heads, seq_length, width][8,6,128,width]
    # 这样做的目的是为了方便后续计算注意力分数（在多头注意力中对不同 head 进行独立计算）
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  # 获取 from_tensor 的形状，期望其秩（维度数）为 2 或 3
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])#[1024, 768]
  # 获取 to_tensor 的形状，期望其秩（维度数）为 2 或 3
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])#[1024, 768]

  if len(from_shape) != len(to_shape):
    raise ValueError(
        #“from_tensor 的秩必须与 to_tensor 的秩相同。”
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  # 如果 from_tensor 的形状为三维，通常形状为 [batch_size, from_seq_length, width]
  if len(from_shape) == 3:
    # 获取批次大小（batch_size）
    batch_size = from_shape[0]
    # 获取 from_tensor 的序列长度（from_seq_length）
    from_seq_length = from_shape[1]
    # 获取 to_tensor 的序列长度（to_seq_length），假设 to_tensor 也是三维张量
    to_seq_length = to_shape[1]
  # 如果 from_tensor 的形状为二维，通常形状为 [batch_size * from_seq_length, width]
  elif len(from_shape) == 2:
    # 当传入二维张量时，需要明确指定 batch_size、from_seq_length 和 to_seq_length 的值，
    # 否则无法还原出原始的三维结构，故抛出错误
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (序列数量) 8
  #   F = `from_tensor` 序列长度 128
  #   T = `to_tensor` 序列长度 128
  #   N = `num_attention_heads` 有12个头
  #   H = `size_per_head` 每个头负责64个特征

  # 不论输入张量是几维的都变成2维，bert源码里都是合并batchsize和seq_length的，说是为了加速计算
  from_tensor_2d = reshape_to_matrix(from_tensor)#(1024, 768)
  # 不论输出张量是几维的都变成2维
  to_tensor_2d = reshape_to_matrix(to_tensor)#(1024, 768)

  #B:batchsize F:`from_tensor` T:`to_tensor` N:`num_attention_heads` H:`size_per_head`

  # 使用全连接层（密集层）将二维输入 from_tensor_2d 投影到一个新的向量空间，
  # 输出的维度为 (12 * 64)。
  # 这一步主要用于生成注意力机制中的 query 向量，
  # 其中 num_attention_heads 表示注意力头的数量，
  # size_per_head 表示每个注意力头的向量维度，
  # 因此输出的向量实际上是各个注意力头向量的拼接。
  # `query_layer` = [B*F, N*H][8*128,12*64][1024*768]
  query_layer = tf.layers.dense(
      from_tensor_2d,                       # 查询的二维张量，通常形状为 [batch_size * from_seq_length, hidden_size]
      num_attention_heads * size_per_head,  # 输出维度，即多个注意力头拼接后的总维度
      activation=query_act,                 # 激活函数，可选，通常在此处可能没有激活函数
      name="query",                         # 该层的名称，用于变量作用域和命名
      kernel_initializer=create_initializer(initializer_range))# 初始化权重的函数，这里根据给定的初始化范围进行初始化

  # `key_layer` = [B*T, N*H][8*128,12*64]
  key_layer = tf.layers.dense(
      to_tensor_2d,                         # 被查询的二维张量，[batch_size * to_seq_length, hidden_size]
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H][8*128,12*64]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer`转成 = [B, N, F, H] #为了加速计算内积
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer`转成 = [B, N, T, H] #为了加速计算内积
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # 对 "query" 和 "key" 进行点积操作，计算得到原始注意力分数
  # 计算结果 attention_scores 的形状为 [B, N, F, T]：
  #   B - 批次大小 (batch_size)
  #   N - 注意力头的数量 (num_attention_heads)
  #   F - "from" 序列的长度 (from_seq_length)
  #   T - "to" 序列的长度 (to_seq_length)
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True) #结果为(8, 12, 128, 128)这里128, 128这个矩阵就是当前批次的当前注意力头，每个token的注意力矩阵
  # 对 attention_scores 进行缩放，乘以 1/sqrt(size_per_head)
  # 这样做的目的是为了抵消维度增加导致的点积值过大问题，确保 softmax 操作时数值稳定
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head))) #消除维度对结果的影响

  # 进行注意力的掩码操作，bert中的掩码只是为了遮盖填充的部分，跟解码器中的掩码还不太一样哦
  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]，F,T矩阵就是F要不要计算T的注意力
    # 原始 attention_mask 的形状为 [B, F, T]
    # 在 axis=1 位置上给 attention_mask 增加一个新的维度，以便与 attention_scores（形状 [B, N, F, T]）的维度匹配。
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # 由于掩码矩阵中我们希望关注到的位置为1，需要屏蔽的位置为0，
    # 所以构造一个掩码矩阵中为1位置为0，掩码矩阵中为0位置为-10000.0的矩阵
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0 #mask为1的时候结果为0 mask为0的时候结果为非常大的负数

    # 由于我们在 softmax 之前将其加入到原始分数中，这实际上相当于完全移除了这些掩码矩阵中为0的位置。
    attention_scores += adder #把这个加入到原始的得分里相当于mask为1的就不变，mask为0的就会变成非常大的负数

  # 将注意力分数归一化为概率。
  # `attention_probs` = [Batch, Num, From, To]
  attention_probs = tf.nn.softmax(attention_scores) #再做softmax此时负数做softmax相当于结果为0了就相当于不考虑了

  # 这实际上是在丢弃整个 token 的注意力（即完全忽略某些 token），
  # 虽然这看起来有点不寻常，但这是直接沿用了原始 Transformer 论文中的做法。
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)#将已经计算好的注意力，再进行一些随机的丢弃

  # 把value_layer也要
  # `value_layer` = [Batch, To, Num, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])#(8, 128, 12, 64)

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3]) #(8, 12, 128, 64)

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)#计算最终结果特征 (8, 12, 128, 64)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])#转换回[8, 128, 12, 64]

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head]) #(1024, 768)

  # 好了，到此为止我们已经获得了新的1024个token的向量特征了，每一个特征都是768维度的
  return context_layer # (1024, 768)


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """来自《Attention is All You Need》的多头、多层 Transformer。几乎是对原始 Transformer 编码器的完全实现。

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float 张量 [batch_size, seq_length, hidden_size].
    attention_mask: (可选) int32 张量 [batch_size, seq_length,seq_length], 其中，用 1 表示可以关注的位置，用 0 表示不应关注的位置
    hidden_size: int. Transformer的隐藏层大小.
    num_hidden_layers: int. Transformer的层数.
    num_attention_heads: int. Transformer的注意力头数.
    intermediate_size: int. "中间层"（即前馈层）的大小。
    intermediate_act_fn: function. 应用在"中间层"（即前馈层）输出的非线性激活函数
    hidden_dropout_prob: float. 隐藏层的失活率.
    attention_probs_dropout_prob: float. 注意力概率的失活率.
    initializer_range: float. 初始化范围 (stddev of truncated
      normal).
    do_return_all_layers: 是否返回所有层还是返回最后一层.

  Returns:
    float 张量 [batch_size, seq_length, hidden_size], 表示 Transformer 的最终隐藏层。

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:# 最终的输出特征的维度，一定要是头数的整数倍，不然怎么分割呢？
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads) #一共要输出768个特征，给每个头分一下，就是注意力头的尺寸，768/12=64
  input_shape = get_shape_list(input_tensor, expected_rank=3) # [8, 128, 768]
  batch_size = input_shape[0]#8
  seq_length = input_shape[1]#128
  input_width = input_shape[2]#768

  # Transformer 在所有层上执行残差加法，因此输入的大小需要与隐藏层大小，也就是输出特征维度相同。
  if input_width != hidden_size: #注意残差连接的方式，需要它俩维度一样才能相加
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # 我们将表示保持为 2D 张量，以避免在 3D 张量和 2D 张量之间来回重塑。
  # 重塑在 GPU/CPU 上通常是免费的，但在 TPU 上可能不是免费的，
  # 因此我们希望将其最小化，以帮助优化器。
  prev_output = reshape_to_matrix(input_tensor) #reshape的目的可能是为了加速 [8, 128, 768]变成[1024,768]
  # 不转换其实也行，但是原始代码中基本都这么做了，把batch和sequence维度合并了，应该是为了加速

  all_layer_outputs = []# 初始化一个空列表，用于存储所有层的输出
  for layer_idx in range(num_hidden_layers):# 遍历所有隐藏层 12层
    with tf.variable_scope("layer_%d" % layer_idx):# 为每一层创建一个变量作用域，作用域名为 "layer_层号"
      layer_input = prev_output# 前一层的输出作为当前层的输入

      with tf.variable_scope("attention"):# Attention 子模块
        attention_heads = []
        with tf.variable_scope("self"):
          # 计算自注意力，注意from_tensor和to_tensor都是当前输入哦
          attention_head = attention_layer(
              from_tensor=layer_input,                  # 查询tensor是layer_input
              to_tensor=layer_input,                    # 被查询tensor也是layer_input，所以才是自注意力，自己的token对自己的token计算注意力
              attention_mask=attention_mask,            # 注意力掩码
              num_attention_heads=num_attention_heads,  # 注意力头的数量
              size_per_head=attention_head_size,        # 每个头的大小
              attention_probs_dropout_prob=attention_probs_dropout_prob,# 注意力概率的 dropout
              initializer_range=initializer_range,      # 初始化范围
              do_return_2d_tensor=True,                 # 是否返回 2D 张量
              batch_size=batch_size,                    # 批次大小
              from_seq_length=seq_length,               # 查询序列长度
              to_seq_length=seq_length)                 # 被查询序列长度
          attention_heads.append(attention_head)

        # 将所有注意力头合并（如果有多个头的话）
        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]# 如果只有一个注意力头，直接使用
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)# 如果有多个注意力头，将它们拼接在一起

        # 执行到hidden_size的线性变换，然后增加一个残差连接
        # 对 attention 输出进行线性变换，投影到 hidden_size 维度
        with tf.variable_scope("output"): #1024, 768 残差连接
          attention_output = tf.layers.dense(
              attention_output,# 经过注意力计算后的输出
              hidden_size, # 输出维度为 hidden_size
              kernel_initializer=create_initializer(initializer_range))     # 权重初始化
          attention_output = dropout(attention_output, hidden_dropout_prob) # 对输出进行 dropout
          attention_output = layer_norm(attention_output + layer_input)     # 残差连接：将输入与输出相加并进行层归一化

      # 在 "中间" 隐藏层上应用激活函数
      with tf.variable_scope("intermediate"): #全连接层 (1024, 3072)
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # 将中间层的输出投影回到 hidden_size 维度并添加残差，这里的残差连接是前馈网络的残差连接
      with tf.variable_scope("output"): #再变回一致的维度
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)# 对输出进行 dropout
        layer_output = layer_norm(layer_output + attention_output)# 残差连接：将输出与 attention_output 相加并进行层归一化
        prev_output = layer_output# 将当前层的输出保存，作为下一层的输入
        all_layer_outputs.append(layer_output)# 将当前层的输出添加到 all_layer_outputs 中

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
  """返回张量的形状列表.不用关心这个函数的实现，只要知道是获取张量的各个维度即可

  Args:
    tensor: 一个 tf.Tensor 对象，用于查找其形状.
    expected_rank: （可选）整数。`tensor` 的预期秩。如果指定了该参数且 `tensor` 的秩不同，将会抛出异
    name: 可选的张量名称，用于错误消息中.

  Returns:
    一个列表，表示张量的形状。所有静态维度将作为 Python 整数返回，而动态维度将作为 tf.Tensor 标量返回.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """将一个秩大于等于2的张量重塑为一个秩为2的张量."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
