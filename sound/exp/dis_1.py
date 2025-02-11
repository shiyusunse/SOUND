from sound.src.models.glance import *
from sound.src.utils.helper import save_csv_result
from sound.src.models.linedp import *
from sound.src.models.mymodel import *
from numpy import *


def get_tp_buggy_lines(model, project, release):
    model.project_name, model.test_release = project, release

    model.line_level_result_file = f'{model.line_level_result_path}{model.project_name}/{model.test_release}-result.csv'
    model.predicted_buggy_lines = []
    model.load_line_level_result()
    predicted_buggy_lines = model.predicted_buggy_lines

    model.oracle_line_dict = dict()
    _, actual_buggy_lines = model.get_oracle_lines()

    tp_buggy_lines = actual_buggy_lines.intersection(predicted_buggy_lines)

    return tp_buggy_lines


def classification_difference():
    my_models = [Barinel()]
    target_models = [Glance_LR_Mixed_Sort, Glance_MD_Mixed_Sort, Glance_EA_Mixed_Sort,
                     LineDP_mixedsort, DeepLineDP(),
                     Ngram(), ErrorProne()]
    text = ''
    for my_model in my_models:
        for target_model in target_models:

            tp_data, hit_data, over_data = '', '', ''
            hit_list, over_list = [], []

            projects = {
                'ambari': ['ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0', 'ambari-2.7.0'],
                'amq': ['amq-5.0.0', 'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0', 'amq-5.9.0', 'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0', 'amq-5.15.0'],
                'bookkeeper': ['bookkeeper-4.0.0', 'bookkeeper-4.2.0', 'bookkeeper-4.4.0'],
                'calcite': ['calcite-1.6.0', 'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0', 'calcite-1.15.0', 'calcite-1.16.0', 'calcite-1.17.0', 'calcite-1.18.0'],
                'cassandra': ['cassandra-0.7.4', 'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11', 'cassandra-1.2.11'],
                'flink': ['flink-1.4.0', 'flink-1.6.0'],
                'groovy': ['groovy-1.0', 'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6', 'groovy-1.8.1', 'groovy-1.8.7', 'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4', 'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0', 'groovy-2.5.5'],
                'hbase': ['hbase-0.94.1', 'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5', 'hbase-0.98.11'],
                'ignite': ['ignite-1.0.0', 'ignite-1.4.0', 'ignite-1.6.0'],
                'log4j2': ['log4j2-2.0', 'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6', 'log4j2-2.7', 'log4j2-2.8', 'log4j2-2.9', 'log4j2-2.10'],
                'mahout': ['mahout-0.3', 'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7', 'mahout-0.8'],
                'mng': ['mng-3.1.0', 'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0', 'mng-3.6.0'],
                'nifi': ['nifi-0.4.0', 'nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0'],
                'nutch': ['nutch-1.1', 'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7', 'nutch-1.8', 'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14', 'nutch-1.15'],
                'storm': ['storm-0.9.0', 'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3', 'storm-1.0.5'],
                'tika': ['tika-0.7', 'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5', 'tika-1.7', 'tika-1.10', 'tika-1.13', 'tika-1.15', 'tika-1.17'],
                'ww': ['ww-2.0.0', 'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0', 'ww-2.2.2', 'ww-2.3.1', 'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20', 'ww-2.3.24'],
                'zookeeper': ['zookeeper-3.4.6', 'zookeeper-3.5.1', 'zookeeper-3.5.2', 'zookeeper-3.5.3']
            }

            for project, releases in projects.items():
                print(my_model.model_name, target_model.model_name, project)
                release_tp_data, release_hit_data, release_over_data = project + ',', project + ',', project + ','
                for test_release in releases[1:]:
                    target_buggy_lines = get_tp_buggy_lines(target_model, project, test_release)
                    glance_buggy_lines = get_tp_buggy_lines(my_model, project, test_release)

                    tp_target = len(target_buggy_lines)
                    if tp_target == 0:
                        num_hit = 0.0
                        num_over = 1.0
                    else:
                        num_hit = len(glance_buggy_lines.intersection(target_buggy_lines)) / tp_target
                        num_over = len(glance_buggy_lines - target_buggy_lines) / tp_target

                    release_tp_data += str(tp_target) + ','
                    release_hit_data += str(round(num_hit, 3)) + ','
                    release_over_data += str(round(num_over, 3)) + ','

                    hit_list.append(num_hit)
                    over_list.append(num_over)

                tp_data += release_tp_data + '\n'
                hit_data += release_hit_data + '\n'
                over_data += release_over_data + '\n'

            save_csv_result(f'../result/{exp}/Difference/',
                            f'{my_model.model_name}-{target_model.model_name}_TP_data.csv', tp_data)
            save_csv_result(f'../result/{exp}/Difference/',
                            f'{my_model.model_name}-{target_model.model_name}_Hit_data.csv', hit_data)
            save_csv_result(f'../result/{exp}/Difference/',
                            f'{my_model.model_name}-{target_model.model_name}_Over_data.csv', over_data)

            text += f'{my_model.model_name},{target_model.model_name},{mean(hit_list)},{mean(over_list)},{median(hit_list)},{median(over_list)}\n'
    save_csv_result(f'../result/{exp}/', 'Difference_summary.csv', text)


if __name__ == '__main__':
    classification_difference()
