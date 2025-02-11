import json
import pandas as pd
from operator import itemgetter
from utils.helper import read_line_level_dataset, read_file_level_dataset
import math

releases = [
            'ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0',
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


def get_tokens_split(tokens_file_name) -> set:
    tokens_csv = pd.read_csv(tokens_file_name, encoding='ISO-8859-1')
    tokens_set = set(tokens_csv.columns)
    return tokens_set


def tarantula(n_cf, n_cs, num_of_bug_lines, num_of_lines):
    if n_cf/num_of_bug_lines+n_cs/(num_of_lines-num_of_bug_lines) == 0:
        return 0
    return (n_cf/num_of_bug_lines)/(n_cf/num_of_bug_lines+n_cs/(num_of_lines-num_of_bug_lines))


def ochiai(n_cf, n_cs, num_of_bug_lines):
    if num_of_bug_lines*(n_cf+n_cs) == 0:
        return 0
    return n_cf/math.sqrt(num_of_bug_lines*(n_cf+n_cs))


def op2(n_cf, n_cs, num_of_bug_lines, num_of_lines):
    if num_of_lines - num_of_bug_lines + 1 == 0:
        return 0
    return n_cf - n_cs/(num_of_lines - num_of_bug_lines + 1)


def barinel(n_cf, n_cs):
    if (n_cs+n_cf) == 0:
        return 0
    return 1-n_cs/(n_cs+n_cf)


def dstar(n_cf, n_cs, num_of_bug_lines):
    if n_cs+num_of_bug_lines-n_cf == 0:
        return 0
    return (n_cf*n_cf)/(n_cs+num_of_bug_lines-n_cf)


def get_oracle_lines(release):
    oracle_line_dict, oracle_line_list = read_line_level_dataset(release), set()
    for file_name in oracle_line_dict:
        oracle_line_list.update([f'{file_name}:{line}' for line in oracle_line_dict[file_name]])
    return oracle_line_dict, oracle_line_list


def normalize_score(score_dict: dict):
    max_value = max(score_dict.values())
    min_value = min(score_dict.values())
    normalized_dict = {k: (v - min_value) / (max_value - min_value) for k, v in score_dict.items()}
    return normalized_dict


def evaluate_tarantula_score(release:str):
    n_cf_file_name = '../Data/n_score/' + release + '_cf.json'
    n_cs_file_name = '../Data/n_score/' + release + '_cs.json'

    tokens_file_name = '../Data/tokens/' + release + '_tokens.csv'

    tarantula_score_file_name = '../Data/score/' + release + '_tarantula_normal.json'
    with open(n_cf_file_name, 'r') as f:
        n_cf = json.load(f)
    with open(n_cs_file_name, 'r') as f:
        n_cs = json.load(f)

    tokens = get_tokens_split(tokens_file_name)

    test_text, test_text_lines, test_labels, test_filename, _ = read_file_level_dataset(release)
    num_of_lines = sum([len(lines) for lines in test_text_lines])

    oracle_line_dict, oracle_line_set = get_oracle_lines(release)
    num_of_bug_lines = len(oracle_line_set)

    score = {}
    for token in tokens:
        score[token] = tarantula(n_cf[token], n_cs[token], num_of_bug_lines, num_of_lines)
    normal_score = normalize_score(score)
    sorted_dict = dict(sorted(normal_score.items(), key=itemgetter(1), reverse=True))
    with open(tarantula_score_file_name, 'w') as f:
        json.dump(sorted_dict, f)


def evaluate_ochiai_score(release:str):
    n_cf_file_name = '../Data/n_score/' + release + '_cf.json'
    n_cs_file_name = '../Data/n_score/' + release + '_cs.json'

    tokens_file_name = '../Data/tokens/' + release + '_tokens.csv'

    ochiai_score_file_name = '../Data/score/' + release + '_ochiai_normal.json'
    with open(n_cf_file_name, 'r') as f:
        n_cf = json.load(f)
    with open(n_cs_file_name, 'r') as f:
        n_cs = json.load(f)

    tokens = get_tokens_split(tokens_file_name)

    oracle_line_dict, oracle_line_set = get_oracle_lines(release)
    num_of_bug_lines = len(oracle_line_set)

    score = {}
    for token in tokens:
        score[token] = ochiai(n_cf[token], n_cs[token], num_of_bug_lines)
    normal_score = normalize_score(score)
    sorted_dict = dict(sorted(normal_score.items(), key=itemgetter(1), reverse=True))
    with open(ochiai_score_file_name, 'w') as f:
        json.dump(sorted_dict, f)


def evaluate_op2_score(release: str):
    n_cf_file_name = '../Data/n_score/' + release + '_cf.json'
    n_cs_file_name = '../Data/n_score/' + release + '_cs.json'

    tokens_file_name = '../Data/tokens/' + release + '_tokens.csv'

    op2_score_file_name = '../Data/score/' + release + '_op2_normal.json'
    with open(n_cf_file_name, 'r') as f:
        n_cf = json.load(f)
    with open(n_cs_file_name, 'r') as f:
        n_cs = json.load(f)

    tokens = get_tokens_split(tokens_file_name)

    test_text, test_text_lines, test_labels, test_filename, _ = read_file_level_dataset(release)
    num_of_lines = sum([len(lines) for lines in test_text_lines])

    oracle_line_dict, oracle_line_set = get_oracle_lines(release)
    num_of_bug_lines = len(oracle_line_set)

    score = {}
    for token in tokens:
        score[token] = op2(n_cf[token], n_cs[token], num_of_bug_lines, num_of_lines)
    normal_score = normalize_score(score)
    sorted_dict = dict(sorted(normal_score.items(), key=itemgetter(1), reverse=True))
    with open(op2_score_file_name, 'w') as f:
        json.dump(sorted_dict, f)


def evaluate_barinel_score(release:str):
    n_cf_file_name = '../Data/n_score/' + release + '_cf.json'
    n_cs_file_name = '../Data/n_score/' + release + '_cs.json'

    tokens_file_name = '../Data/tokens/' + release + '_tokens.csv'

    barinel_score_file_name = '../Data/score/' + release + '_barinel_normal.json'
    with open(n_cf_file_name, 'r') as f:
        n_cf = json.load(f)
    with open(n_cs_file_name, 'r') as f:
        n_cs = json.load(f)

    tokens = get_tokens_split(tokens_file_name)

    score = {}
    for token in tokens:
        score[token] = barinel(n_cf[token], n_cs[token])
    normal_score = normalize_score(score)
    sorted_dict = dict(sorted(normal_score.items(), key=itemgetter(1), reverse=True))
    with open(barinel_score_file_name, 'w') as f:
        json.dump(sorted_dict, f)


def evaluate_dstar_score(release:str):
    n_cf_file_name = '../Data/n_score/' + release + '_cf.json'
    n_cs_file_name = '../Data/n_score/' + release + '_cs.json'

    tokens_file_name = '../Data/tokens/' + release + '_tokens.csv'

    dstar_score_file_name = '../Data/score/' + release + '_dstar_normal.json'
    with open(n_cf_file_name, 'r') as f:
        n_cf = json.load(f)
    with open(n_cs_file_name, 'r') as f:
        n_cs = json.load(f)

    tokens = get_tokens_split(tokens_file_name)

    oracle_line_dict, oracle_line_set = get_oracle_lines(release)
    num_of_bug_lines = len(oracle_line_set)

    score = {}
    for token in tokens:
        score[token] = dstar(n_cf[token], n_cs[token], num_of_bug_lines)
    normal_score = normalize_score(score)
    sorted_dict = dict(sorted(normal_score.items(), key=itemgetter(1), reverse=True))
    with open(dstar_score_file_name, 'w') as f:
        json.dump(sorted_dict, f)


for release in releases:
    evaluate_tarantula_score(release)
    evaluate_ochiai_score(release)
    evaluate_op2_score(release)
    evaluate_barinel_score(release)
    evaluate_dstar_score(release)
