import re

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

max_seq_len = 50

all_train_releases = {'ambari': 'ambari-1.2.0', 'amq': 'amq-5.0.0', 'bookkeeper': 'bookkeeper-4.0.0',
                      'calcite': 'calcite-1.6.0', 'cassandra': 'cassandra-0.7.4', 'flink': 'flink-1.4.0',
                      'groovy': 'groovy-1.0', 'hbase': 'hbase-0.94.1', 'hive': 'hive-0.14.0',
                      'ignite': 'ignite-1.0.0', 'log4j2': 'log4j2-2.0', 'mahout': 'mahout-0.3',
                      'mng': 'mng-3.1.0', 'nifi': 'nifi-0.4.0', 'nutch': 'nutch-1.1', 'storm': 'storm-0.9.0',
                      'tika': 'tika-0.7', 'ww': 'ww-2.0.0', 'zookeeper': 'zookeeper-3.4.6',
                      }

all_eval_releases = {
    'ambari': ['ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0', 'ambari-2.7.0'],
    'amq': ['amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0', 'amq-5.9.0',
            'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0', 'amq-5.15.0'],
    'bookkeeper': ['bookkeeper-4.2.0', 'bookkeeper-4.4.0'],
    'calcite': ['calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0', 'calcite-1.15.0', 'calcite-1.16.0',
                'calcite-1.17.0', 'calcite-1.18.0'],
    'cassandra': ['cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11', 'cassandra-1.2.11'],
    'flink': ['flink-1.6.0'],
    'groovy': ['groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6',
               'groovy-1.8.1', 'groovy-1.8.7', 'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4',
               'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0', 'groovy-2.5.5'],
    'hbase': ['hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5', 'hbase-0.98.11'],
    'hive': ['hive-1.2.0', 'hive-2.0.0', 'hive-2.1.0'],
    'ignite': ['ignite-1.4.0', 'ignite-1.6.0'],
    'log4j2': ['log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5',
               'log4j2-2.6', 'log4j2-2.7', 'log4j2-2.8', 'log4j2-2.9', 'log4j2-2.10'],
    'mahout': ['mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7', 'mahout-0.8'],
    'mng': ['mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0', 'mng-3.6.0'],
    'nifi': ['nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0'],
    'nutch': ['nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7',
              'nutch-1.8', 'nutch-1.9', 'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14',
              'nutch-1.15'],
    'storm': ['storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3', 'storm-1.0.5'],
    'tika': ['tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5',
             'tika-1.7', 'tika-1.10', 'tika-1.13', 'tika-1.15', 'tika-1.17'],
    'ww': ['ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0',
           'ww-2.2.2', 'ww-2.3.1', 'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20',
           'ww-2.3.24'],
    'zookeeper': ['zookeeper-3.5.1', 'zookeeper-3.5.2', 'zookeeper-3.5.3']
    }

all_releases = {
    'ambari': ['ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0',
               'ambari-2.7.0'],
    'amq': ['amq-5.0.0', 'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0',
            'amq-5.9.0', 'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0', 'amq-5.15.0'],
    'bookkeeper': ['bookkeeper-4.0.0', 'bookkeeper-4.2.0', 'bookkeeper-4.4.0'],
    'calcite': ['calcite-1.6.0', 'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0', 'calcite-1.15.0',
                'calcite-1.16.0', 'calcite-1.17.0', 'calcite-1.18.0'],
    'cassandra': ['cassandra-0.7.4', 'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11',
                  'cassandra-1.2.11'],
    'flink': ['flink-1.4.0', 'flink-1.6.0'],
    'groovy': ['groovy-1.0', 'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6', 'groovy-1.8.1',
               'groovy-1.8.7', 'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4', 'groovy-2.4.6', 'groovy-2.4.8',
               'groovy-2.5.0', 'groovy-2.5.5'],
    'hbase': ['hbase-0.94.1', 'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5', 'hbase-0.98.11'],
    'hive': ['hive-0.14.0', 'hive-1.2.0', 'hive-2.0.0', 'hive-2.1.0'],
    'ignite': ['ignite-1.0.0', 'ignite-1.4.0', 'ignite-1.6.0'],
    'log4j2': ['log4j2-2.0', 'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6',
               'log4j2-2.7', 'log4j2-2.8', 'log4j2-2.9', 'log4j2-2.10'],
    'mahout': ['mahout-0.3', 'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7', 'mahout-0.8'],
    'mng': ['mng-3.1.0', 'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0', 'mng-3.6.0'],
    'nifi': ['nifi-0.4.0', 'nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0'],
    'nutch': ['nutch-1.1', 'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7', 'nutch-1.8', 'nutch-1.9',
              'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14', 'nutch-1.15'],
    'storm': ['storm-0.9.0', 'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3', 'storm-1.0.5'],
    'tika': ['tika-0.7', 'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5', 'tika-1.7',
             'tika-1.10', 'tika-1.13', 'tika-1.15', 'tika-1.17'],
    'ww': ['ww-2.0.0', 'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0', 'ww-2.2.2', 'ww-2.3.1',
           'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20', 'ww-2.3.24'],
    'zookeeper': ['zookeeper-3.4.6', 'zookeeper-3.5.1', 'zookeeper-3.5.2', 'zookeeper-3.5.3']
    }

all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/preprocessed_data/'

word2vec_dir = '../output/Word2Vec_model/'


def get_df(rel, is_baseline=False):
    if is_baseline:
        df = pd.read_csv('../' + file_lvl_gt + rel + ".csv")

    else:
        df = pd.read_csv(file_lvl_gt + rel + ".csv")

    df = df.fillna('')

    df = df[df['is_blank'] == False]
    df = df[df['is_test_file'] == False]

    return df


def prepare_code2d(code_list, to_lowercase=False):
    code2d = []

    for c in code_list:
        c = re.sub('\\s+', ' ', c)

        if to_lowercase:
            c = c.lower()

        token_list = c.strip().split()
        total_tokens = len(token_list)

        token_list = token_list[:max_seq_len]

        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)

        code2d.append(token_list)

    return code2d


def get_code3d_and_label(df, to_lowercase=False):
    code3d = []
    all_file_label = []

    for filename, group_df in df.groupby('filename'):
        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        code2d = prepare_code2d(code, to_lowercase)
        code3d.append(code2d)

        all_file_label.append(file_label)

    return code3d, all_file_label


def get_w2v_path():
    return word2vec_dir


def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.syn0).cuda()

    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1, embed_dim).cuda()))

    return word2vec_weights


def pad_code(code_list_3d, max_sent_len, limit_sent_len=True, mode='train'):
    paded = []

    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)

        if mode == 'train':
            if max_sent_len - len(file) > 0:
                for i in range(0, max_sent_len - len(file)):
                    sent_list.append([0] * max_seq_len)

        if limit_sent_len:
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)

    return paded


def get_dataloader(code_vec, label_list, batch_size, max_sent_len):
    y_tensor = torch.cuda.FloatTensor([label for label in label_list])
    code_vec_pad = pad_code(code_vec, max_sent_len)
    tensor_dataset = TensorDataset(torch.tensor(code_vec_pad), y_tensor)

    dl = DataLoader(tensor_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    return dl


def get_x_vec(code_3d, word2vec):
    x_vec = [
        [[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text]
         for text in texts] for texts in code_3d]

    return x_vec