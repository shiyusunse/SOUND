import pandas as pd
import numpy as np
import subprocess, re, os, time

from multiprocessing import Pool

from tqdm import tqdm

all_eval_releases = [
    'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0', 'ambari-2.7.0',
    'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0', 'amq-5.9.0',
    'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0', 'amq-5.15.0',
    'bookkeeper-4.2.0', 'bookkeeper-4.4.0',
    'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0', 'calcite-1.15.0', 'calcite-1.16.0', 'calcite-1.17.0',
    'calcite-1.18.0',
    'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11', 'cassandra-1.2.11',
    'flink-1.6.0',
    'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6',
    'groovy-1.8.1', 'groovy-1.8.7', 'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4',
    'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0', 'groovy-2.5.5',
    'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5', 'hbase-0.98.11',
    'hive-1.2.0', 'hive-2.0.0', 'hive-2.1.0',
    'ignite-1.4.0', 'ignite-1.6.0',
    'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5',
    'log4j2-2.6', 'log4j2-2.7', 'log4j2-2.8', 'log4j2-2.9', 'log4j2-2.10',
    'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7', 'mahout-0.8',
    'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0', 'mng-3.6.0',
    'nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0',
    'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7',
    'nutch-1.8', 'nutch-1.9', 'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14',
    'nutch-1.15',
    'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3', 'storm-1.0.5',
    'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5',
    'tika-1.7', 'tika-1.10', 'tika-1.13', 'tika-1.15', 'tika-1.17',
    'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0',
    'ww-2.2.2', 'ww-2.3.1', 'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20',
    'ww-2.3.24',
    'zookeeper-3.5.1', 'zookeeper-3.5.2', 'zookeeper-3.5.3'
]


all_dataset_name = ['ambari', 'amq', 'bookkeeper', 'calcite', 'cassandra', 'flink', 'groovy', 'hbase', 'hive', 'ignite',
                    'log4j2', 'mahout', 'mng', 'nifi', 'nutch', 'storm', 'tika', 'ww', 'zookeeper']

base_file_dir = './ErrorProne_data/'
base_command = "javac -J-Duser.language=en -J-Duser.country=US -J-Xbootclasspath/p:javac-9+181-r4173-1.jar -XDcompilePolicy=simple -processorpath error_prone_core-2.4.0-with-dependencies.jar:dataflow-shaded-3.1.2.jar:jFormatString-3.0.0.jar "

result_dir = './ErrorProne_result/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def run_ErrorProne(rel):
    df_list = []
    java_file_dir = base_file_dir + rel + '/'

    file_list = os.listdir(java_file_dir)

    for java_filename in tqdm(file_list):
        f = open(java_file_dir + java_filename, 'r', encoding='utf-8', errors='ignore')
        java_code = f.readlines()

        code_len = len(java_code)

        output = subprocess.getoutput(base_command + java_file_dir + java_filename)

        reported_lines = re.findall('\d+: error:', output)
        reported_lines = [int(l.replace(':', '').replace('error', '')) for l in reported_lines]
        reported_lines = list(set(reported_lines))

        line_df = pd.DataFrame()

        line_df['filename'] = [java_filename.replace('_', '/')] * code_len
        line_df['test-release'] = [rel] * len(line_df)
        line_df['line_number'] = np.arange(1, code_len + 1)
        line_df['EP_prediction_result'] = line_df['line_number'].isin(reported_lines)

        df_list.append(line_df)

    if len(df_list) != 0:
        final_df = pd.concat(df_list)
        final_df.to_csv(result_dir + rel + '-line-lvl-result.txt', index=False)
    print('finished', rel)


if __name__ == '__main__':
    agents = 5
    chunksize = 8
    with Pool(processes=agents) as pool:
        pool.map(run_ErrorProne, all_eval_releases, chunksize)