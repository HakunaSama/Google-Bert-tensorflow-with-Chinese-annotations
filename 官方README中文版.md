# BERT

**\*\*\*\*\* 2020年3月11日最新更新：更小的BERT模型 \*\*\*\*\***

这是一个发布版本，包含了24个更小的BERT模型（仅限英语，无大小写区分，采用WordPiece遮蔽训练），在论文[《Well-Read Students Learn Better: On the Importance of Pre-training Compact Models》](https://arxiv.org/abs/1908.08962)中有提及。

我们证明了标准的BERT方案（包括模型架构和训练目标）在各种模型大小上都能有效工作，不仅仅局限于BERT-Base和BERT-Large。这些更小的BERT模型旨在针对计算资源受限的环境。它们可以像原始BERT模型那样进行微调，但在知识蒸馏的情形下效果最佳——即由更大、更精确的教师模型生成微调标签。

我们的目标是使资源较少的研究机构也能开展相关研究，并鼓励社区探索除增加模型容量之外的其他创新方向。

你可以从[这里](https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip)下载所有24个模型，或者从下表中单独下载：

|   |H=128|H=256|H=512|H=768|
|---|:---:|:---:|:---:|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**][2_128]|[2/256][2_256]|[2/512][2_512]|[2/768][2_768]|
| **L=4**  |[4/128][4_128]|[**4/256 (BERT-Mini)**][4_256]|[**4/512 (BERT-Small)**][4_512]|[4/768][4_768]|
| **L=6**  |[6/128][6_128]|[6/256][6_256]|[6/512][6_512]|[6/768][6_768]|
| **L=8**  |[8/128][8_128]|[8/256][8_256]|[**8/512 (BERT-Medium)**][8_512]|[8/768][8_768]|
| **L=10** |[10/128][10_128]|[10/256][10_256]|[10/512][10_512]|[10/768][10_768]|
| **L=12** |[12/128][12_128]|[12/256][12_256]|[12/512][12_512]|[**12/768 (BERT-Base)**][12_768]|

请注意，本版本中的BERT-Base模型仅作为完整性参考；它是在与原始模型相同的训练机制下重新训练的。

以下是对应的GLUE测试集得分：

|Model|Score|CoLA|SST-2|MRPC|STS-B|QQP|MNLI-m|MNLI-mm|QNLI(v2)|RTE|WNLI|AX|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BERT-Tiny|64.2|0.0|83.2|81.1/71.1|74.3/73.6|62.2/83.4|70.2|70.3|81.5|57.2|62.3|21.0|
|BERT-Mini|65.8|0.0|85.9|81.1/71.8|75.4/73.3|66.4/86.2|74.8|74.3|84.1|57.9|62.3|26.1|
|BERT-Small|71.2|27.8|89.7|83.4/76.2|78.8/77.0|68.1/87.0|77.6|77.0|86.4|61.8|62.3|28.6|
|BERT-Medium|73.5|38.0|89.6|86.6/81.6|80.4/78.4|69.6/87.9|80.0|79.1|87.7|62.2|62.3|30.5|

- 对于每个任务，我们从下面列出的超参数中选取最佳设置，并训练了4个epoch：
  - 批量大小：8, 16, 32, 64, 128
  - 学习率：3e-4, 1e-4, 5e-5, 3e-5

如果你使用这些模型，请引用以下论文:

```
@article{turc2019,
  title={Well-Read Students Learn Better: On the Importance of Pre-training Compact Models},
  author={Turc, Iulia and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1908.08962v2 },
  year={2019}
}
```

[2_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip
[2_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-256_A-4.zip
[2_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-512_A-8.zip
[2_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-768_A-12.zip
[4_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-128_A-2.zip
[4_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip
[4_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip
[4_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-768_A-12.zip
[6_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-128_A-2.zip
[6_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-256_A-4.zip
[6_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-512_A-8.zip
[6_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-768_A-12.zip
[8_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-128_A-2.zip
[8_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-256_A-4.zip
[8_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip
[8_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-768_A-12.zip
[10_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-128_A-2.zip
[10_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-256_A-4.zip
[10_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-512_A-8.zip
[10_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-768_A-12.zip
[12_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-128_A-2.zip
[12_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-256_A-4.zip
[12_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-512_A-8.zip
[12_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
[all]: https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip

**\*\*\*\*\* 2021年5月31日最新更新：Whole Word Masking模型 \*\*\*\*\***

这是一个全新模型发布，基于预处理代码的改进。

在原始预处理代码中，我们会随机选择WordPiece标记进行遮蔽。例如：

`输入文本: the man jumped up , put his basket on phil ##am ##mon ' s head`
`原始遮蔽输入: [MASK] man [MASK] up , put his [MASK] on phil
[MASK] ##mon ' s head`

新技术称为Whole Word Masking。在这种方法中，我们总是一次性遮蔽一个单词所对应的**所有**标记。整体遮蔽比例保持不变。

`Whole Word Masked Input: the man [MASK] up , put his basket on [MASK] [MASK]
[MASK] ' s head`

训练过程保持不变——我们仍然独立预测每个被遮蔽的WordPiece标记。改进的关键在于原始预测任务对那些被拆分为多个WordPiece的单词来说太“简单”。

在数据生成时，可通过向`create_pretraining_data.py`传递标志`--do_whole_word_mask=True`来启用此功能。

带有Whole Word Masking的预训练模型链接如下。数据和训练过程与原模型完全一致，且模型结构和词表相同。我们仅包含BERT-Large模型。使用这些模型时，请在论文中明确说明你使用的是Whole Word Masking版本的BERT-Large。

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24层，1024隐藏单元，16注意力头，参数量340M

*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24层，1024隐藏单元，16注意力头，参数量340M

Model                                    | SQUAD 1.1 F1/EM | Multi NLI Accuracy
---------------------------------------- | :-------------: | :----------------:
BERT-Large, Uncased (Original)           | 91.0/84.3       | 86.05
BERT-Large, Uncased (Whole Word Masking) | 92.8/86.7       | 87.07
BERT-Large, Cased (Original)             | 91.5/84.8       | 86.09
BERT-Large, Cased (Whole Word Masking)   | 92.9/86.7       | 86.46

**\*\*\*\*\* 2021年2月7日最新更新：TfHub模块 \*\*\*\*\***

BERT现已上传至[TensorFlow Hub](https://tfhub.dev)。参见`run_classifier_with_tfhub.py`了解如何使用TF Hub模块，或在[Colab](https://colab.sandbox.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)中运行示例。

**\*\*\*\*\* 2018年11月23日最新更新：未归一化的多语种模型 + 泰语 + 蒙古语 \*\*\*\*\***

我们上传了一个新的多语种模型，该模型对输入不进行任何归一化处理（不转换小写、不去除重音、也不进行Unicode归一化），同时新增了泰语和蒙古语支持。

**建议在开发多语种模型时（尤其是处理非拉丁字母语言）使用此版本。**

此版本无需任何代码修改，可从下列链接下载：

*   **[`BERT-Base, Multilingual Cased`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104种语言，12层，768隐藏单元，12注意力头，参数量110M

**\*\*\*\*\* 2018年11月15日最新更新：SOTA SQuAD 2.0系统 \*\*\*\*\***

我们发布了代码变更，用以重现我们的83% F1 SQuAD 2.0系统，该系统目前在排行榜上领先3%。详情请参阅README中关于SQuAD 2.0的部分。

**\*\*\*\*\* 2018年11月5日最新更新：第三方 PyTorch 和 Chainer 版本的 BERT 可用 \*\*\*\*\***

NLP 研究人员来自HuggingFace 发布了一个
[[PyTorch 版本的 BERT](https://github.com/huggingface/pytorch-pretrained-BERT)，](https://github.com/huggingface/pytorch-pretrained-BERT)
该版本兼容我们的预训练检查点并能重现我们的结果。Sosuke Kobayashi 也发布了一个
[[Chainer 版本的 BERT](https://github.com/soskek/bert-chainer)（感谢！）](https://github.com/soskek/bert-chainer)
我们未参与该 PyTorch 实现的开发或维护，如有问题请直接联系相关作者。

**\*\*\*\*\* 2018年11月3日最新更新：多语种和中文模型可用
\*\*\*\*\***

我们发布了两个新的 BERT 模型：

*   **（不推荐，建议使用 `Multilingual Cased`）**：支持102种语言，12层，768隐藏单元，12注意力头，参数量110M
*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    简体和繁体中文，12层，768隐藏单元，12注意力头，参数量110M

对于中文，我们采用基于字符的分词，而其他语言则使用 WordPiece 分词。两个模型均可直接使用，无需代码修改。我们已更新 `tokenization.py` 中的 `BasicTokenizer` 以支持中文字符分词；如果你曾 fork 过此代码，请及时更新。不过分词 API 保持不变。

更多信息请参阅[多语种 README](https://github.com/google-research/bert/blob/master/multilingual.md)。

**\*\*\*\*\* 以上为最新信息。 \*\*\*\*\***

## 简介

**BERT**（全称 **双向编码器表示的 Transformer**）是一种全新的预训练语言表示方法，在多种自然语言处理（NLP）任务上取得了最先进的结果。

详细描述 BERT 以及在多个任务上取得完整结果的论文可见于：
https://arxiv.org/abs/1810.04805。

例如，下面是 SQuAD v1.1 问答任务的一些成绩：

SQuAD v1.1 Leaderboard (Oct 8th 2018) | Test EM  | Test F1
------------------------------------- | :------: | :------:
第一名集成模型 - BERT             | **87.4** | **93.2**
第二名集成模型 - nlnet            | 86.0     | 91.7
第一名单模型 - BERT         | **85.1** | **91.8**
第二名单模型 - nlnet        | 83.5     | 90.1

以及多个自然语言推断任务的成绩：

System                  | MultiNLI | Question NLI | SWAG
----------------------- | :------: | :----------: | :------:
BERT                    | **86.7** | **91.1**     | **86.3**
OpenAI GPT (Prev. SOTA) | 82.2     | 88.1         | 75.0

还有许多其他任务。

更重要的是，这些结果几乎无需针对具体任务设计特殊的神经网络架构。

如果你已经了解 BERT 并希望快速上手，你可以
[下载预训练模型](#pre-trained-models) 并在几分钟内
[运行最先进的微调](#fine-tuning-with-bert)。

## 什么是 BERT？

BERT 是一种预训练语言表示的方法，这意味着我们在一个大型文本语料库（例如维基百科）上训练一个通用的“语言理解”模型，然后将该模型用于我们关心的下游 NLP 任务（例如问答）。BERT 之所以优于以往方法，是因为它是第一个**无监督**、**深度双向**的 NLP 预训练系统。

**无监督**意味着 BERT 仅使用纯文本语料进行训练，这一点非常重要，因为网络上公开的纯文本数据极其丰富，并覆盖多种语言。

预训练表示可以是**上下文无关**或**上下文相关**的，而上下文相关表示又可分为**单向**或**双向**。上下文无关模型（如 [word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) 或 [GloVe](https://nlp.stanford.edu/projects/glove/)）为词汇表中每个词生成一个固定的“词嵌入”，因此单词 “bank” 在 “bank deposit” 和 “river bank” 中的表示相同。而上下文相关模型则会根据句子中其它单词生成每个单词的表示。

BERT 建立在最近关于预训练上下文相关表示的工作基础上——包括 [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432)、[Generative Pre-Training](https://blog.openai.com/language-unsupervised/)、[ELMo](https://allennlp.org/elmo) 以及 [ULMFit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)——但这些模型都是**单向**或**浅层双向**的。这意味着每个单词的上下文化仅依赖于其左侧（或右侧）的单词。例如，在句子 “I made a bank deposit” 中，单向表示的 “bank” 仅基于 “I made a” 而不考虑 “deposit”。一些早期工作确实将左右上下文模型的表示浅层结合，但 BERT 则从深层网络的最底层开始，同时利用左右上下文来表示单词 “bank”（即 “I made a ... deposit”），从而实现了**深度双向**表示。

BERT 的方法很简单：我们将输入中15%的单词随机遮蔽，然后将整个序列输入到深度双向 [Transformer](https://arxiv.org/abs/1706.03762) 编码器中，最后只预测被遮蔽的单词。例如：

```
输入: the man went to the [MASK1] . he bought a [MASK2] of milk.
标签: [MASK1] = store; [MASK2] = gallon
```

为了学习句子之间的关系，我们还训练了一个简单任务，该任务可以从任何单语语料中生成：给定两个句子 A 和 B，判断 B 是否为 A 的真实后续句，还是语料库中随机选取的句子？

```
句子 A: the man went to the store .
句子 B: he bought a gallon of milk .
标签: IsNextSentence
```

```
句子  A: the man went to the store .
句子  B: penguins are flightless .
标签: NotNextSentence
```

接着，我们在一个大型语料库（维基百科 + [BookCorpus](http://yknzhu.wixsite.com/mbweb)）上训练了一个大型模型（从12层到24层的 Transformer），训练长达 100 万步，这就是 BERT。

使用 BERT 包含两个阶段：**预训练**和**微调**。

**预训练**过程相当昂贵（在4到16个 Cloud TPU 上训练大约4天），但每种语言只需进行一次（目前模型仅支持英语，未来会发布多语种模型）。我们发布了论文中使用的多个由 Google 预训练的模型。大多数 NLP 研究人员无需从头预训练自己的模型。

**微调**则成本低廉。论文中的所有结果都可以在单个 Cloud TPU 上一小时内，或在 GPU 上几小时内从相同预训练模型复现。例如，SQuAD 可在单个 Cloud TPU 上大约30分钟训练，达到 91.0% 的开发集 F1 分数，这是单系统的最先进水平。

另一个重要特点是 BERT 能非常容易地适应各种类型的 NLP 任务。在论文中，我们展示了在句子级（如 SST-2）、句子对级（如 MultiNLI）、词级（如 NER）和片段级（如 SQuAD）任务上几乎无需任何任务特定修改即可达到最先进的结果。

## 本仓库发布内容

我们发布了以下内容：

*   BERT 模型架构的 TensorFlow 代码（基本上是一个标准的 [Transformer](https://arxiv.org/abs/1706.03762) 架构）。
*   论文中使用的 `BERT-Base` 和 `BERT-Large` 模型的预训练检查点（包括小写版和保留大小写版）。
*   用于一键重现论文中最重要微调实验（包括 SQuAD、MultiNLI 和 MRPC）的 TensorFlow 代码。

本仓库中所有代码均可在 CPU、GPU 和 Cloud TPU 上开箱即用。

## 预训练模型

我们发布了论文中的 `BERT-Base` 和 `BERT-Large` 模型。
“Uncased” 表示在 WordPiece 分词前已将文本转换为小写，例如 “John Smith” 转为 “john smith”，同时还去除了所有重音。
“Cased” 则保留原始大小写和重音信息。通常，除非你的任务对大小写信息（例如命名实体识别或词性标注）很敏感，否则 “Uncased” 模型效果更佳。

这些模型均在与源代码相同的许可证（Apache 2.0）下发布。

关于多语种和中文模型的信息，请参阅[多语种 README](https://github.com/google-research/bert/blob/master/multilingual.md)。

**使用保留大小写的模型时，请确保向训练脚本传递 `--do_lower=False`。（或者如果使用自定义脚本，请直接传递 `do_lower_case=False` 给 `FullTokenizer`。）**

模型下载链接如下（右键点击链接名称选择“另存为...”即可下载）：

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24层，1024隐藏单元，16注意力头，参数量340M
*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24层，1024隐藏单元，16注意力头，参数量340M
*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**:
    12层，768隐藏单元，12注意力头，参数量110M
*   **[`BERT-Large, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)**:
    24层，1024隐藏单元，16注意力头，参数量340M
*   **[`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**:
    12层，768隐藏单元，12注意力头，参数量110M
*   **[`BERT-Large, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)**:
    24层，1024隐藏单元，16注意力头，参数量340M
*   **[`BERT-Base, Multilingual Cased (New, recommended)`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104种语言，12层，768隐藏单元，12注意力头，参数量110M
*   **[`BERT-Base, Multilingual Uncased (Orig, not recommended)`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
    ****（不推荐，建议使用 `Multilingual Cased`）**：支持102种语言，12层，768隐藏单元，12注意力头，参数量110M
*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    简体及繁体中文，12层，768隐藏单元，12注意力头，参数量110M

每个 .zip 文件包含三部分内容：

*   一个 TensorFlow 检查点（`bert_model.ckpt`），包含预训练权重（实际上是3个文件）。
*   一个词表文件（`vocab.txt`），用于将 WordPiece 映射为词 ID。
*   一个配置文件（`bert_config.json`），指定模型的超参数。

## 使用 BERT 进行微调

**重要提示**：论文中的所有结果均在单个具有 64GB 内存的 Cloud TPU 上进行微调。目前，使用 12GB-16GB 显存的 GPU 很难复现论文中大多数 `BERT-Large` 结果，因为可容纳的最大批量大小过小。我们正在为本仓库添加代码，以允许在 GPU 上使用更大的有效批量大小。更多细节请参阅下文的[内存不足问题](#out-of-memory-issues)。

该代码已在 TensorFlow 1.11.0 上测试。支持 Python2 和 Python3（但内部主要使用 Python2）。

使用 `BERT-Base` 的微调示例在至少 12GB 显存的 GPU 上应能运行给定超参数。

### 使用 Cloud TPU 进行微调

下面的大部分示例假设你将在本地机器上使用 GPU（如 Titan X 或 GTX 1080）进行训练/评估。

如果你有 Cloud TPU，只需在 `run_classifier.py` 或 `run_squad.py` 中添加如下标志：

```
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

请参阅[Google Cloud TPU 教程](https://cloud.google.com/tpu/docs/tutorials/mnist)了解如何使用 Cloud TPU。或者，你也可以使用 Google Colab 笔记本
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)"。

在 Cloud TPU 上，预训练模型和输出目录需要存放在 Google Cloud Storage 上。例如，如果你有一个名为 `some_bucket` 的存储桶，可使用如下标志：

```
  --output_dir=gs://some_bucket/my_output_dir/
```

解压后的预训练模型文件也可以在 Google Cloud Storage 文件夹 `gs://bert_models/2018_10_18` 中找到，例如：

```
export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
```

### 句子（及句子对）分类任务

在运行此示例之前，你必须下载 [GLUE 数据](https://gluebenchmark.com/tasks)（运行[该脚本](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)并解压至目录 `$GLUE_DIR`）。接着，下载 `BERT-Base` 检查点并解压到目录 `$BERT_BASE_DIR`。

此示例代码将 `BERT-Base` 在微软研究院同义句任务（MRPC）数据集上微调，该数据集仅包含 3600 个样本，通常在大多数 GPU 上几分钟即可完成微调。

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

你应看到类似如下的输出：

```
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248
```

这意味着开发集准确率为 84.55%。由于 MRPC 数据集样本较少，即便使用相同预训练检查点，多次运行结果可能在 84% 到 88% 之间波动较大。

其他一些预训练模型在 `run_classifier.py` 中也有现成实现，因此你可以轻松参考这些示例，使用 BERT 处理任何单句或句子对分类任务。

注意：你可能会看到 “Running train on CPU” 的提示，这仅表示程序在使用非 Cloud TPU 的设备运行（包括 GPU）。

#### 分类器的预测

完成训练后，你可以通过设置 `--do_predict=true` 使用分类器进行推断。你需要在输入文件夹中提供一个名为 test.tsv 的文件。输出将存放在输出文件夹中名为 test_results.tsv 的文件中，每行包含对应样本的类别概率。

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/mrpc_output/
```

### SQuAD 1.1

斯坦福问答数据集（SQuAD）是一个广受欢迎的问答基准数据集。BERT 在 SQuAD 上取得了最先进的结果，几乎无需针对特定任务修改网络架构或数据增强，但需要对 SQuAD 段落的变长和基于字符的答案标注进行半复杂的预处理与后处理。相关处理在 `run_squad.py` 中已有实现和说明。

要在 SQuAD 上运行，首先需下载数据集。虽然 [SQuAD 网站](https://rajpurkar.github.io/SQuAD-explorer/) 似乎不再提供 v1.1 数据集链接，但你可以从以下链接获取必要文件：

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

将这些文件下载到目录 `$SQUAD_DIR`。

论文中最先进的 SQuAD 结果目前无法在 12GB-16GB GPU 上复现（事实上，即便批量大小为1，在使用 `BERT-Large` 时也无法在 12GB GPU 上运行）。然而，一个相当强大的 `BERT-Base` 模型可在 GPU 上使用以下超参数训练：

```shell
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

开发集预测结果将保存在输出目录中的 `predictions.json` 文件内：

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```

输出大致如下：

```shell
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

你应获得与论文中 `BERT-Base` 报告的 88.5% 相近的结果。

如果你有 Cloud TPU 资源，可用 `BERT-Large` 训练。下面是一组（略有不同于论文）的超参数，通常可获得约 90.5%-91.0% F1 的单模型成绩，仅在 SQuAD 上训练：

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

例如，一次随机运行该参数得到以下开发集分数：

```shell
{"f1": 90.87081895814865, "exact_match": 84.38978240302744}
```

如果在此之前对 [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) 数据集进行一个 epoch 的微调，结果会更好，但你需要将 TriviaQA 转换成 SQuAD 的 JSON 格式。

### SQuAD 2.0

该模型同样在 `run_squad.py` 中实现和说明。

要在 SQuAD 2.0 上运行，首先需下载数据集，必要文件可从以下链接获得：

- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

将这些文件下载到目录 `$SQUAD_DIR`。

在 Cloud TPU 上，你可以使用 BERT-Large 如下训练：

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
```

假设你已将输出目录中的所有内容复制到本地目录 `./squad/`，初始开发集预测将保存在 `./squad/predictions.json` 中，而每个问题中 “无答案”（""）与最佳非空答案之间的分数差将存储在 `./squad/null_odds.json` 文件中。

运行以下脚本以调整预测无答案与非无答案的阈值：

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
```

### 内存不足问题

论文中所有实验均在具有 64GB 内存的 Cloud TPU 上进行微调。因此，当使用 12GB-16GB 显存的 GPU 时，若采用论文中描述的相同超参数，很可能会遇到内存不足问题。

影响内存使用的因素包括：

- **`max_seq_length`**：发布模型训练时的序列长度最高可达512，但在微调时可使用更短的最大序列长度以节省大量内存。此参数由示例代码中的 `max_seq_length` 控制。
- **`train_batch_size`**：内存使用量与批量大小成正比。
- **模型类型，`BERT-Base` 与 `BERT-Large`**：`BERT-Large` 需要的内存远高于 `BERT-Base`。
- **优化器**：BERT 默认使用 Adam 优化器，该优化器需要额外大量内存来存储 `m` 和 `v` 向量。切换为更节省内存的优化器可降低内存占用，但可能会影响结果。我们尚未在微调时试验其他优化器。

使用默认训练脚本（`run_classifier.py` 和 `run_squad.py`）时，我们在单个 Titan X GPU（12GB 内存）上使用 TensorFlow 1.11.0 测得的最大批量大小如下：

System       | Seq Length | Max Batch Size
------------ | ---------- | --------------
`BERT-Base`  | 64         | 64
...          | 128        | 32
...          | 256        | 16
...          | 320        | 14
...          | 384        | 12
...          | 512        | 6
`BERT-Large` | 64         | 12
...          | 128        | 6
...          | 256        | 2
...          | 320        | 1
...          | 384        | 0
...          | 512        | 0

不幸的是，`BERT-Large` 的这些最大批量大小太小，会对模型精度产生负面影响，无论学习率如何。我们正在添加代码，使 GPU 上可使用更大的有效批量大小。该方案可能基于以下一种或两种技术：

- **梯度累积**：由于 mini-batch 内的样本在梯度计算上通常相互独立（除批归一化外，此处未使用），可在更新权重前累计多个较小的 mini-batch 梯度，这与一次大批量更新等效。
- **梯度检查点**：[梯度检查点](https://github.com/openai/gradient-checkpointing) 技术：训练深度神经网络时，主要内存消耗来自于前向传播中存储以供反向传播使用的中间激活。梯度检查点通过智能重算这些激活，在牺牲计算时间的同时节省内存。

**但目前这些方法尚未在本版本中实现。**

## 使用 BERT 提取固定特征向量（类似于 ELMo）

在某些情况下，与其对整个预训练模型进行端到端微调，不如提取 *预训练的上下文嵌入*——即从预训练模型的隐藏层中生成每个输入标记的固定上下文表示，这也能缓解大部分内存不足问题。

例如，我们提供了 `extract_features.py` 脚本，可如下使用：

```shell
# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > /tmp/input.txt

python extract_features.py \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

该脚本会生成一个 JSON 文件（每行对应一行输入），其中包含指定 `layers`（-1 表示 Transformer 的最后一层隐藏层，依此类推）的 BERT 激活值。

注意：默认情况下，该脚本会生成非常大的输出文件（大约每个输入标记 15KB）。如果你需要保持原始文本与分词后词语之间的对齐（用于映射训练标签），请参阅下文的 [分词](#tokenization) 部分。

**注意：**你可能会看到 “Could not find trained model in model_dir: /tmp/tmpuB5g5c, running initialization to predict.” 的提示。这是预期中的，表示我们正在使用 `init_from_checkpoint()` API 而非保存模型 API。如果你没有指定有效检查点，脚本会报错。

## 分词

对于句子级（或句子对）任务，分词过程非常简单。你只需参考 `run_classifier.py` 和 `extract_features.py` 中的示例代码。句子级任务的基本步骤如下：

1. 实例化 `tokenizer = tokenization.FullTokenizer`
2. 使用 `tokens = tokenizer.tokenize(raw_text)` 对原始文本进行分词。
3. 截断至最大序列长度。（你最多可用 512，但为节省内存和提升速度通常希望使用较短序列）
4. 在适当位置添加 `[CLS]` 和 `[SEP]` 标记。

对于词级和片段级任务（如 SQuAD 和 NER）较为复杂，因为你需要保持输入文本与输出文本的对齐以映射训练标签。SQuAD 尤其复杂，因为输入标签是基于 *字符* 的，且 SQuAD 段落往往比最大序列长度长。请参阅 `run_squad.py` 中的代码了解如何处理此问题。

在介绍处理词级任务的通用方法前，先了解我们的分词器具体做了哪些操作。它主要包含三个步骤：

1. **文本规范化**：将所有空白字符转换为空格，并（对于 `Uncased` 模型）将文本转为小写并去除重音。例如，`John Johanson's,` 变为 `john johanson's,`。
2. **标点分割**：在所有标点符号的两侧分割（即在所有标点符号周围添加空格）。标点符号定义为 (a) Unicode 类为 `P*` 的字符，(b) 任何非字母/数字/空格的 ASCII 字符（例如技术上不属于标点的 `$`）。例如，`john johanson's,` 变为 `john johanson ' s ,`。
3. **WordPiece 分词**：对上一步结果用空格分词，然后对每个词应用 [WordPiece](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py) 分词。（我们的实现直接基于 `tensor2tensor` 的实现）例如，`john johanson ' s ,` 变为 `john johan ##son ' s ,`。

这种方法的优点在于它与大多数现有的英语分词器具有“兼容性”。例如，假设你有一个词性标注任务如下：

```
输入：  John Johanson 's   house
标签： NNP  NNP      POS NN
```

分词后输出可能为：

```
标记： john johan ##son ' s house
```

关键在于，这与如果原始文本为 `John Johanson's house`（无额外空格）的分词输出相同。

如果你已有预分词的文本及对应词级标注，可以对每个输入词独立分词，并确定性地保持原始与分词后词语的对齐：

```python
### 输入
orig_tokens = ["John", "Johanson", "'s", "house"]
labels      = ["NNP", "NNP", "POS", "NN"]

### 输出
bert_tokens = []

# token map 是一个 int -> int 的映射，表示 orig_tokens 中的索引与 bert_tokens 中对应的索引。
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
  orig_to_tok_map.append(len(bert_tokens))
  bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# 此时 bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# 而 orig_to_tok_map == [1, 2, 4, 6]
```

这样，你就可以利用 `orig_to_tok_map` 将 `labels` 映射到分词后的表示上。

需要注意的是，某些常见的英语分词方案可能会与 BERT 预训练时使用的方法略有不匹配。例如，如果你的预处理会将缩写如 `do n't` 拆分开来，可能会导致不匹配。若有可能，建议预处理数据使其恢复成原始格式；如果不行，这种不匹配通常不会有太大影响。

## 使用 BERT 进行预训练

我们发布了用于在任意文本语料上执行“masked LM”和“next sentence prediction”任务的代码。请注意，这并非用于论文中精确使用的代码（原始代码由 C++ 编写，且包含额外复杂性），但该代码确实生成了与论文描述一致的预训练数据。

以下是数据生成方法。输入为纯文本文件，每行一句话。（注意，这些必须是真正的句子，以便于“next sentence prediction”任务。）文档之间以空行分隔。输出是一组序列化为 `TFRecord` 格式的 `tf.train.Example`。

你可以使用现成的 NLP 工具（如 [spaCy](https://spacy.io/)）进行句子分割。`create_pretraining_data.py` 脚本会将段落拼接直至达到最大序列长度，以减少填充造成的计算浪费（具体细节请参阅脚本）。不过，你也可以有意在输入数据中加入少量噪声（例如随机截断 2% 的输入段），使模型在微调时对非句子输入更为鲁棒。

该脚本会将整个输入文件中的所有示例加载到内存中，因此对于大文件，建议将输入文件拆分后多次调用该脚本。（你可以对 `run_pretraining.py` 传递文件通配符，例如 `tf_examples.tf_record*`。）

`max_predictions_per_seq` 表示每个序列中被遮蔽的 LM 预测的最大数量。你应将其设置为大约 `max_seq_length` * `masked_lm_prob`（脚本不会自动计算，因为需要手动在两个脚本中同时传入精确值）。

```shell
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

下面是如何进行预训练。如果你从零开始预训练，请不要指定 `init_checkpoint`。模型配置（包括词汇表大小）在 `bert_config_file` 中指定。此示例代码仅预训练少量步数（20 步），但实际操作中你可能需要设置 `num_train_steps` 至 10000 步或更多。传递给 `run_pretraining.py` 的 `max_seq_length` 和 `max_predictions_per_seq` 参数必须与传递给 `create_pretraining_data.py` 的保持一致。

```shell
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

该过程会输出类似如下结果：

```
***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
```

注意，由于我们的 `sample_text.txt` 文件非常小，该示例训练在少量步数内便会过拟合，从而产生不切实际的高准确率。

* ### 预训练提示和注意事项

  - **如果使用自己的词汇表，请务必在 `bert_config.json` 中修改 `vocab_size`。否则，若词汇表较大而未修改该参数，可能会因索引越界而在 GPU 或 TPU 上训练时出现 NaN。**
  - 如果你的任务有大量领域特定语料（如“影评”或“科学论文”），在现有 BERT 检查点上进行额外预训练可能会带来好处。
  - 论文中使用的学习率为 1e-4。但如果你从现有 BERT 检查点进行额外预训练，建议使用更小的学习率（如 2e-5）。
  - 当前 BERT 模型仅支持英语，但我们计划近期发布一个多语种模型，该模型将在多种语言上进行预训练（希望在 2018 年 11 月底前发布）。
  - 长序列计算成本极高，因为注意力机制的计算复杂度为序列长度的平方。换句话说，一批 64 个长度为 512 的序列远比一批 256 个长度为 128 的序列计算成本高。全连接/卷积部分成本相同，但注意力部分成本差异巨大。因此，一个好的策略是先以序列长度 128 预训练约 90,000 步，然后以长度 512 再预训练 10,000 步。长序列主要用于学习位置嵌入，而这部分知识通常能较快学到。注意，这需要针对不同 `max_seq_length` 值分别生成数据两次。
  - 如果你从零开始预训练，请准备好预训练是计算上非常昂贵的，尤其在 GPU 上。我们建议在单个 [可抢占的 Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing) 上预训练 `BERT-Base`，大约需要 2 周时间，成本约 500 美元（基于 2018 年 10 月定价）。在仅使用单个 Cloud TPU 时，批量大小需缩小，建议使用能最大化利用 TPU 内存的批量大小。

  ### 预训练数据

  我们**不会**发布论文中使用的预处理数据集。对于维基百科，推荐的预处理方法是下载
  [最新的 dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)，
  使用 [`WikiExtractor.py`](https://github.com/attardi/wikiextractor) 提取文本，然后进行必要清理以转换为纯文本。

  不幸的是，收集 [BookCorpus](http://yknzhu.wixsite.com/mbweb) 数据的研究者已不再公开该数据集。
  [Project Guttenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html) 是一个较小（约2亿词）的公共领域旧书集合。
  [Common Crawl](http://commoncrawl.org/) 也是一个非常庞大的文本集合，但你可能需要进行大量预处理和清理才能提取适用于 BERT 预训练的语料。

  ### 学习新的 WordPiece 词汇表

  本仓库不包含用于**学习**新 WordPiece 词汇表的代码。原因在于论文中使用的代码由 C++ 编写，并依赖 Google 内部库。对于英语来说，通常最好直接使用我们的词汇表和预训练模型。对于其他语言的词汇表学习，已有多种开源方案可供选择，但请注意这些方案与我们的 `tokenization.py` 库不兼容：

  - [Google 的 SentencePiece 库](https://github.com/google/sentencepiece)
  - [tensor2tensor 的 WordPiece 生成脚本](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py)
  - [Rico Sennrich 的 Byte Pair Encoding 库](https://github.com/rsennrich/subword-nmt)

## 在 Colab 中使用 BERT

如果你想在 [Colab](https://colab.research.google.com) 上使用 BERT，可以使用笔记本
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)"。
**截至本文撰写时（2018年10月31日），Colab 用户可以完全免费使用 Cloud TPU。**
注意：每个用户仅限一个 TPU，资源有限，需拥有 Google Cloud Platform 存储账户（尽管注册时可能获得免费额度），且将来可能不再免费。点击上述 BERT Colab 链接了解更多信息。

## FAQ（常见问题）

#### 这段代码是否兼容 Cloud TPU？GPU 呢？

是的，本仓库中的所有代码均可在 CPU、GPU 和 Cloud TPU 上开箱即用。不过，GPU 训练仅支持单 GPU。

#### 我遇到了内存不足错误，这是怎么回事？

请参阅上文[内存不足问题](#内存不足问题)部分了解更多信息。

#### 有 PyTorch 版本可用吗？

目前没有官方的 PyTorch 实现。不过， HuggingFace 的 NLP 研究人员发布了一个
[PyTorch 版本的 BERT](https://github.com/huggingface/pytorch-pretrained-BERT)，
该版本兼容我们的预训练检查点，并能重现我们的结果。我们未参与该实现的开发或维护，如有问题请直接联系相关作者。

#### 有 Chainer 版本可用吗？

目前没有官方的 Chainer 实现。不过， Sosuke Kobayashi 发布了一个
[Chainer 版本的 BERT](https://github.com/soskek/bert-chainer)，
该版本兼容我们的预训练检查点，并能重现我们的结果。我们未参与其开发或维护，如有问题请直接联系相关作者。

#### 是否会发布其他语言的模型？

是的，我们计划近期发布一个多语种 BERT 模型。具体支持哪些语言尚不确定，但很可能涵盖大多数拥有足够大维基百科的语言。

#### 会发布比 `BERT-Large` 更大的模型吗？

目前我们尚未尝试训练比 `BERT-Large` 更大的模型。如有显著改进，有可能会发布更大模型。

#### 本库采用何种许可证？

所有代码**及**模型均在 Apache 2.0 许可证下发布。详情请参阅 `LICENSE` 文件。

#### 如何引用 BERT？

目前，请引用[Arxiv 论文](https://arxiv.org/abs/1810.04805)：

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

若我们将论文提交给会议或期刊，BibTeX 引用会作相应更新。
