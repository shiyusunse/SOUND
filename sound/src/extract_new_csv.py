from tools import read_java_file_without_comments
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import json
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

dataset_string = '../Dataset'
result_string = '../Result'

file_level_path = f'{dataset_string}/File-level/'
line_level_path = f'{dataset_string}/Line-level/'
result_path = f'{result_string}'

file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'

releases = [

    'ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0', 'ambari-2.7.0',
    'amq-5.0.0', 'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0',
    'amq-5.9.0', 'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0', 'amq-5.15.0',
    'bookkeeper-4.0.0', 'bookkeeper-4.2.0', 'bookkeeper-4.4.0',
    'calcite-1.6.0', 'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0',
    'calcite-1.15.0', 'calcite-1.16.0', 'calcite-1.17.0', 'calcite-1.18.0',
    'cassandra-0.7.4', 'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11', 'cassandra-1.2.11',
    'flink-1.4.0', 'flink-1.6.0',
    'groovy-1.0', 'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6', 'groovy-1.8.1', 'groovy-1.8.7',
    'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4', 'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0', 'groovy-2.5.5',
    'hbase-0.94.1', 'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5', 'hbase-0.98.11',
    'hive-0.14.0', 'hive-1.2.0', 'hive-2.0.0', 'hive-2.1.0',
    'ignite-1.0.0', 'ignite-1.4.0', 'ignite-1.6.0',
    'log4j2-2.0', 'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6', 'log4j2-2.7',
    'log4j2-2.8', 'log4j2-2.9', 'log4j2-2.10',
    'mahout-0.3', 'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7', 'mahout-0.8',
    'mng-3.0.0', 'mng-3.1.0', 'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0', 'mng-3.6.0',
    'nifi-0.4.0', 'nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0',
    'nutch-1.1', 'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7', 'nutch-1.8', 'nutch-1.9',
    'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14', 'nutch-1.15',
    'storm-0.9.0', 'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3', 'storm-1.0.5',
    'tika-0.7', 'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5', 'tika-1.7', 'tika-1.10',
    'tika-1.13', 'tika-1.15', 'tika-1.17',
    'ww-2.0.0', 'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0', 'ww-2.2.2', 'ww-2.3.1',
    'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20', 'ww-2.3.24',
    'zookeeper-3.4.6', 'zookeeper-3.5.1', 'zookeeper-3.5.2', 'zookeeper-3.5.3',

]


def extract_tokens_from(release):
    tokens1_file_name = '../Data/tokens/'+release+'_tokens_part1.csv'
    tokens2_file_name = '../Data/tokens/'+release+'_tokens_part2.csv'
    tokens_file_name = '../Data/tokens/'+release+'_tokens.csv'

    path = f'{file_level_path}{release}{file_level_path_suffix}'
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    src_file_indices = [lines.index(line) for line in lines if r'.java,true,"' in line or r'.java,false,"' in line]
    src_files = [lines[index].split(',')[0] for index in src_file_indices]
    string_labels = [lines[index].split(',')[1] for index in src_file_indices]
    numeric_labels = [1 if label == 'true' else 0 for label in string_labels]
    assert (len(numeric_labels) == len(src_file_indices))

    for i in range(0, int(len(src_file_indices)/2)):
        dfs = []

        s_index = src_file_indices[i]
        e_index = src_file_indices[i + 1] if i + 1 < len(src_file_indices) else len(lines)
        code_lines = [line.strip() for line in lines[s_index:e_index]]
        code_lines[0] = code_lines[0].split(',')[-1][1:]
        code_lines[-1] = code_lines[-1][:-1]
        if len(code_lines) == 0:
            continue

        f = open("../Result/tmp.java", 'w', encoding='utf-8')
        for line in code_lines:
            print(line, end='', file=f)
        f.close()
        java_contents = [read_java_file_without_comments('../Result/tmp.java')]
        if java_contents == ['']:
            continue
        vectorizer = CountVectorizer(lowercase=False, min_df=1)
        X = vectorizer.fit_transform(java_contents)
        feature_names = vectorizer.get_feature_names_out()

        df = pd.DataFrame(X.toarray(), columns=feature_names)
        df['Bug'] = numeric_labels[i]

        if i == 0:
            df.to_csv('../Result/concat.csv', sep=',', index=False, header=True)
        else:
            a = pd.read_csv('../Result/concat.csv')
            dfs.append(df)
            dfs.append(a)
            c = pd.concat(dfs, ignore_index=True)
            c = c.fillna(0)
            popped_col = c.pop("Bug")
            c["Bug"] = popped_col
            c.to_csv('../Result/concat.csv', sep=',', index=False, header=True)
    final_df = pd.read_csv('../Result/concat.csv')
    final_df.to_csv(tokens1_file_name, sep=',', index=False, header=True)

    for i in range(int(len(src_file_indices) / 2), len(src_file_indices)):
        dfs = []

        s_index = src_file_indices[i]
        e_index = src_file_indices[i + 1] if i + 1 < len(src_file_indices) else len(lines)
        code_lines = [line.strip() for line in lines[s_index:e_index]]
        code_lines[0] = code_lines[0].split(',')[-1][1:]
        code_lines[-1] = code_lines[-1][:-1]
        if len(code_lines) == 0:
            continue

        f = open("../Result/tmp.java", 'w', encoding='utf-8')
        for line in code_lines:
            print(line, end='', file=f)
        f.close()
        java_contents = [read_java_file_without_comments('../Result/tmp.java')]
        if java_contents == ['']:
            continue
        vectorizer = CountVectorizer(lowercase=False, min_df=1)
        X = vectorizer.fit_transform(java_contents)
        feature_names = vectorizer.get_feature_names_out()

        df = pd.DataFrame(X.toarray(), columns=feature_names)
        df['Bug'] = numeric_labels[i]

        if i == int(len(src_file_indices) / 2):
            df.to_csv('../Result/concat.csv', sep=',', index=False, header=True)
        else:
            a = pd.read_csv('../Result/concat.csv')
            dfs.append(df)
            dfs.append(a)
            c = pd.concat(dfs, ignore_index=True)
            c = c.fillna(0)

            popped_col = c.pop("Bug")
            c["Bug"] = popped_col
            c.to_csv('../Result/concat.csv', sep=',', index=False, header=True)
    final_df = pd.read_csv('../Result/concat.csv')
    final_df.to_csv(tokens2_file_name, sep=',', index=False, header=True)

    dfs = []
    df1 = pd.read_csv(tokens1_file_name)
    df2 = pd.read_csv(tokens2_file_name)
    dfs.append(df1)
    dfs.append(df2)
    c = pd.concat(dfs, ignore_index=True)
    c = c.fillna(0)
    popped_col = c.pop("Bug")
    c["Bug"] = popped_col
    c.to_csv(tokens_file_name, sep=',', index=False, header=True)


def get_tokens_split(tokens_file_name) -> set:
    tokens_csv = pd.read_csv(tokens_file_name, encoding='ISO-8859-1')
    tokens_set = set(tokens_csv.columns)
    return tokens_set


def get_n_score(release):
    file_save_path = '../Dataset/File-level'
    line_save_path = '../Dataset/Line-level'
    file_level_name = release + '_ground-truth-files_dataset.csv'
    line_level_name = release + '_defective_lines_dataset.csv'
    tokens_file_name = '../Data/tokens/' + release + '_tokens.csv'
    cf_file_name = '../Data/n_score/' + release + '_cf.json'
    cs_file_name = '../Data/n_score/' + release + '_cs.json'
    us_file_name = '../Data/n_score/' + release + '_us.json'
    uf_file_name = '../Data/n_score/' + release + '_uf.json'

    n_cf = {}
    n_cs = {}


    tokens = get_tokens_split(tokens_file_name)
    for token in tokens:
        n_cf[token] = 0
        n_cs[token] = 0


    path = file_save_path + '/' + file_level_name
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        src_file_indices = [lines.index(line) for line in lines if r'.java,true,"' in line or r'.java,false,"' in line]
        src_files = [lines[index].split(',')[0] for index in src_file_indices]
        string_labels = [lines[index].split(',')[1] for index in src_file_indices]
        numeric_labels = [1 if label == 'true' else 0 for label in string_labels]
        assert (len(numeric_labels) == len(src_file_indices))

    for i in range(len(src_file_indices)):
        file_name = src_files[i]
        s_index = src_file_indices[i]
        e_index = src_file_indices[i + 1] if i + 1 < len(src_file_indices) else len(lines)
        file_lines = [line.strip() for line in lines[s_index:e_index]]
        file_lines[0] = file_lines[0].split(',')[-1][1:]
        file_lines[-1] = file_lines[-1][:-1]
        with open('../Result/tmp_ratio.java', 'w', encoding='utf-8') as file:
            for line in file_lines:
                print(line, end='\n', file=file)

        with open('../Result/tmp_ratio.java', 'r', encoding='utf-8') as file:
            file_lines = file.readlines()

        path = line_save_path + '/' + line_level_name
        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            line_level_lines = file.readlines()
            file_buggy_lines = []
            for line in line_level_lines[1:]:
                temp = line.split(',', 2)
                bug_line_file_name, bug_line = temp[0], temp[2]
                if bug_line_file_name == file_name:
                    file_buggy_lines.append(bug_line)


        for line in file_lines:
            s = line.strip()
            s = s.replace('"', '""')
            s = "\"" + s + "\"\n"
            if line in file_buggy_lines:
                if re.findall(r'\b[a-zA-Z]{2,}\b', line.strip()):
                    vectorizer = CountVectorizer(lowercase=False, min_df=2)
                    tokenizer = vectorizer.build_tokenizer()
                    line_tokens = tokenizer(line)
                    for token in tokens:
                        if token in line_tokens:
                            n_cf[token] += 1
                file_buggy_lines.remove(line)
            elif line.strip() in file_buggy_lines:
                if re.findall(r'\b[a-zA-Z]{2,}\b', line.strip()):
                    vectorizer = CountVectorizer(lowercase=False, min_df=2)
                    tokenizer = vectorizer.build_tokenizer()
                    line_tokens = tokenizer(line)
                    for token in tokens:
                        if token in line_tokens:
                            n_cf[token] += 1
                file_buggy_lines.remove(line.strip())
            elif s in file_buggy_lines:
                if re.findall(r'\b[a-zA-Z]{2,}\b', s.strip()):
                    vectorizer = CountVectorizer(lowercase=False, min_df=2)
                    tokenizer = vectorizer.build_tokenizer()
                    line_tokens = tokenizer(s)
                    for token in tokens:
                        if token in line_tokens:
                            n_cf[token] += 1
                file_buggy_lines.remove(s)
            else:
                if re.findall(r'\b[a-zA-Z]{2,}\b', line.strip()):
                    vectorizer = CountVectorizer(lowercase=False, min_df=2)
                    tokenizer = vectorizer.build_tokenizer()
                    line_tokens = tokenizer(line)
                    for token in tokens:
                        if token in line_tokens:
                            n_cs[token] += 1

    with open(cf_file_name, 'w') as f:
        json.dump(n_cf, f)
    with open(cs_file_name, 'w') as f:
        json.dump(n_cs, f)


for release in releases:
    extract_tokens_from(release)
    get_n_score(release)
