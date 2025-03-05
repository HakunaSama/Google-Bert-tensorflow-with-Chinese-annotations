本项目是Google官方版本bert的注释版本，其中将官方的README进行翻译，并且对于官方源码进行了更加详细的中文注释，但是本份源码暂时只针MRPC数据集及其任务的训练代码进行了注释翻译和讲解，且暂时只对其中的核心代码的注释进行翻译，包括以下内容：

数据处理模块、词向量嵌入、位置嵌入、token类型嵌入（在MRPC数据集的句子对分类任务中，0代表属于第一句话，1代表属于第二句话）、注意力计算层、训练过程

环境要求：推荐python==3.6，TensorFlow==1.11.0

基本的架构和官方版本其实是差不多的

对在阅读本份源码的时候需要您有基本的transformer的知识。

下载预训练模型https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip，然后放到GLUE/BERT_BASE_DIR文件夹下，然后使用download_data.py文件下载数据集，放入GLUE/glue_data文件夹下。

对run_classifier.py文件的配置中写入以下参数：

--task_name=MRPC

--do_train=True

--do_eval=True

--data_dir=../GLUE/glue_data/MRPC

--vocab_file=../GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/vocab.txt

--bert_config_file=../GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/bert_config.json

--init_checkpoint=../GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/bert_model.ckpt

--max_seq_length=128

--train_batch_size=8

--learning_rate=2e-5

--num_train_epochs=3.0

--output_dir=../GLUE/output/

