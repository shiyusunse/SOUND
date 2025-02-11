import json
import pandas as pd

releases = ['ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0',
            'amq-5.0.0', 'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0',
            'amq-5.9.0', 'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0',
            'bookkeeper-4.0.0', 'bookkeeper-4.2.0',
            'calcite-1.6.0', 'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0',
            'calcite-1.15.0', 'calcite-1.16.0', 'calcite-1.17.0',
            'cassandra-0.7.4', 'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11',
            'flink-1.4.0',
            'groovy-1.0', 'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6', 'groovy-1.8.1', 'groovy-1.8.7',
            'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4', 'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0',
            'hbase-0.94.1', 'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5',
            'hive-0.14.0', 'hive-1.2.0', 'hive-2.0.0',
            'ignite-1.0.0', 'ignite-1.4.0',
            'log4j2-2.0', 'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6', 'log4j2-2.7',
            'log4j2-2.8', 'log4j2-2.9',
            'mahout-0.3', 'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7',
            'mng-3.0.0', 'mng-3.1.0', 'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0',
            'nifi-0.4.0', 'nifi-1.2.0', 'nifi-1.5.0',
            'nutch-1.1', 'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7', 'nutch-1.8', 'nutch-1.9',
            'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14',
            'storm-0.9.0', 'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3',
            'tika-0.7', 'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5', 'tika-1.7', 'tika-1.10',
            'tika-1.13', 'tika-1.15',
            'ww-2.0.0', 'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0', 'ww-2.2.2', 'ww-2.3.1',
            'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20',
            'zookeeper-3.4.6', 'zookeeper-3.5.1', 'zookeeper-3.5.2',
]


def select_top100(release):
    json_file_name = '../Data/score/'+release+'_barinel_normal.json'
    op2_file_name = '../Data/score/'+release+'_op2_normal.json'
    dstar_file_name = '../Data/score/'+release+'_dstar_normal.json'
    tarantula_file_name = '../Data/score/'+release+'_tarantula_normal.json'
    ochiai_file_name = '../Data/score/'+release+'_ochiai_normal.json'

    cf_json_file_name = '../Data/n_score/'+release+'_cf.json'

    tokens_file_name = '../Data/tokens/'+release+'_tokens.csv'

    selected_file_name = '../Data/selected/'+release+'_barinel.csv'
    op2_selected_file_name = '../Data/selected/'+release+'_op2.csv'
    dstar_selected_file_name = '../Data/selected/'+release+'_dstar.csv'
    tarantula_selected_file_name = '../Data/selected/'+release+'_tarantula.csv'
    ochiai_selected_file_name = '../Data/selected/'+release+'_ochiai.csv'

    with open(json_file_name, 'r') as f:
        token_score = json.load(f)
    with open(cf_json_file_name, 'r') as f:
        cf_times = json.load(f)
    with open(op2_file_name, 'r') as f:
        op2_score = json.load(f)
    with open(dstar_file_name, 'r') as f:
        dstar_score = json.load(f)
    with open(tarantula_file_name, 'r') as f:
        tarantula_score = json.load(f)
    with open(ochiai_file_name, 'r') as f:
        ochiai_score = json.load(f)

    df = pd.read_csv(tokens_file_name)

    sorted_keys = sorted(token_score.keys(), key=lambda k: (token_score[k], cf_times.get(k, float('inf'))), reverse=True)
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(selected_file_name, index=False)

    sorted_keys = sorted(dstar_score.keys(), key=lambda k: (dstar_score[k], cf_times.get(k, float('inf'))), reverse=True)
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(dstar_selected_file_name, index=False)

    sorted_keys = sorted(tarantula_score.keys(), key=lambda k: (tarantula_score[k], cf_times.get(k, float('inf'))), reverse=True)
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(tarantula_selected_file_name, index=False)

    sorted_keys = sorted(ochiai_score.keys(), key=lambda k: (ochiai_score[k], cf_times.get(k, float('inf'))), reverse=True)
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(ochiai_selected_file_name, index=False)

    sorted_keys = sorted(op2_score.keys(), key=lambda k: (op2_score[k], cf_times.get(k, float('inf'))), reverse=True)
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(op2_selected_file_name, index=False)


for release in releases:
    select_top100(release)
