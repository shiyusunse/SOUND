import math

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sound.src.utils.config import USE_CACHE
from sound.src.utils.helper import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class BaseModel(object):
    model_name = 'BaseModel'

    def __init__(self, train_release: str = '', test_release: str = '', test_result_path='', is_realistic=False):
        self.result_path = test_result_path if test_result_path != '' else f'{result_path}/{self.model_name}/'

        self.file_level_result_path = f'{self.result_path}file_result/'
        self.line_level_result_path = f'{self.result_path}line_result/'
        self.buggy_density_path = f'{self.result_path}buggy_density/'

        self.file_level_evaluation_file = f'{self.file_level_result_path}evaluation.csv'
        self.line_level_evaluation_file = f'{self.line_level_result_path}evaluation.csv'
        self.execution_time_file = f'{self.result_path}time.csv'

        self.barinel_score_name = '../Data/score/'+train_release+'_barinel_normal.json'
        self.dice_score_name = '../Data/score/' + train_release + '_dice_normal.json'
        self.dstar_score_name = '../Data/score/' + train_release + '_dstar_normal.json'
        self.ochiai_score_name = '../Data/score/' + train_release + '_ochiai_normal.json'
        self.op2_score_name = '../Data/score/' + train_release + '_op2_normal.json'
        self.tarantula_score_name = '../Data/score/' + train_release + '_tarantula_normal.json'

        self.graph_file_name = '../Data/graph/'+train_release+'_SF_graph.txt'
        self.barinel_graph_name = '../Data/graph/'+train_release+'_barinel_SF_graph.txt'
        self.op2_graph_name = '../Data/graph/'+train_release+'_op2_SF_graph.txt'
        self.dstar_graph_name = '../Data/graph/'+train_release+'_dstar_SF_graph.txt'
        self.ochiai_graph_name = '../Data/graph/'+train_release+'_ochiai_SF_graph.txt'
        self.tarantula_graph_name = '../Data/graph/'+train_release+'_tarantula_SF_graph.txt'

        self.project_name = train_release.split('-')[0]
        np.random.seed(0)
        self.random_state = 0
        self.threshold_effort = 0.2

        self.train_release = train_release
        self.test_release = test_release

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

        if is_realistic:
            self.train_text, self.train_text_lines, self.train_label, self.train_filename, self.train_text_lines_without_comments = read_file_level_dataset(
                train_release, file_path=f'{root_path}Dataset/File-level/')
        else:
            self.train_text, self.train_text_lines, self.train_label, self.train_filename, self.train_text_lines_without_comments = read_file_level_dataset(
                train_release)
        self.test_text, self.test_text_lines, self.test_labels, self.test_filename, self.test_text_lines_without_comments = read_file_level_dataset(
            test_release)

        self.file_level_result_file = f'{self.file_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.line_level_result_file = f'{self.line_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.buggy_density_file = f'{self.buggy_density_path}{self.test_release}-density.csv'
        self.commit_buggy_path = f'{dataset_path}{self.test_release.split("-")[0]}'

        self.init_file_path()

        self.test_pred_labels = []
        self.test_pred_scores = []
        self.test_pred_density = dict()

        self.oracle_line_dict, self.oracle_line_set = self.get_oracle_lines()
        self.predicted_buggy_lines = []
        self.predicted_buggy_score = []
        self.predicted_density = []

        self.num_total_lines = sum([len(lines) for lines in self.test_text_lines])
        self.num_total_lines_without_comments = sum([len(lines) for lines in self.test_text_lines_without_comments])
        self.num_actual_buggy_lines = len(self.oracle_line_set)

        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments
        else:
            self.line_threshold = 0.0

        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

    def init_file_path(self):
        make_path(self.result_path)
        make_path(self.file_level_result_path)
        make_path(self.line_level_result_path)
        make_path(self.buggy_density_path)
        make_path(f'{self.file_level_result_path}{self.project_name}/')
        make_path(f'{self.line_level_result_path}{self.project_name}/')

    def get_oracle_lines(self):
        oracle_line_dict, oracle_line_list = read_line_level_dataset(self.test_release), set()
        for file_name in oracle_line_dict:
            oracle_line_list.update([f'{file_name}:{line}' for line in oracle_line_dict[file_name]])
        return oracle_line_dict, oracle_line_list

    def file_level_prediction(self):
        print(f"Prediction\t=>\t{self.test_release}")
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        train_vtr = self.vector.fit_transform(self.train_text)
        test_vtr = self.vector.transform(self.test_text)
        self.clf.fit(train_vtr, self.train_label)

        self.test_pred_labels = self.clf.predict(test_vtr)
        if self.model_name == 'MIT-TMI-SVM':
            self.test_pred_scores = np.array([score for score in self.test_pred_labels])
        else:
            self.test_pred_scores = np.array([score[1] for score in self.clf.predict_proba(test_vtr)])

        self.save_file_level_result()

    def line_level_prediction(self):
        print(f'Line level prediction for: {self.model_name}')
        pass

    def analyze_file_level_result(self):
        self.load_file_level_result()

        total_file, identified_file, total_line, identified_line, predicted_file, predicted_line, dropped_line = 0, 0, 0, 0, 0, 0, 0

        for index in range(len(self.test_labels)):
            buggy_line = len(self.test_text_lines[index])
            if self.test_pred_labels[index] == 1:
                predicted_file += 1
                predicted_line += buggy_line

        for index in range(len(self.test_labels)):
            if self.test_labels[index] == 1:
                buggy_line = len(self.oracle_line_dict[self.test_filename[index]])
                if self.test_pred_labels[index] == 1:
                    identified_line += buggy_line
                    identified_file += 1
                else:
                    dropped_line += len(self.oracle_line_dict[self.test_filename[index]])
                total_line += buggy_line
                total_file += 1

        print(f'Buggy file hit info: {identified_file}/{total_file} - {round(identified_file / total_file * 100, 1)}%')
        print(f'Buggy line hit info: {identified_line}/{total_line} - {round(identified_line / total_line * 100, 1)}%')
        print(f'Predicted {predicted_file} buggy files contain {predicted_line} lines')

        append_title = True if not os.path.exists(self.file_level_evaluation_file) else False
        title = 'release,precision,recall,f1-score,accuracy,mcc,identified/total files,max identified/total lines\n'
        with open(self.file_level_evaluation_file, 'a') as file:
            file.write(title) if append_title else None
            file.write(f'{self.test_release},'
                       f'{metrics.precision_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.recall_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.f1_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.accuracy_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.matthews_corrcoef(self.test_labels, self.test_pred_labels)},'
                       f'{identified_file}/{total_file},'
                       f'{identified_line}/{total_line},'
                       f'\n')
        return

    def analyze_line_level_result(self):
        self.load_file_level_result()
        total_lines_in_defective_files, buggy_lines_in_defective_files = 0, 0
        for index in range(len(self.test_pred_labels)):
            if self.test_pred_labels[index] == 1:
                total_lines_in_defective_files += len(self.test_text_lines[index])
                if self.test_labels[index] == 1:
                    buggy_lines_in_defective_files += len(self.oracle_line_dict[self.test_filename[index]])

        self.load_line_level_result()

        tp = len(self.oracle_line_set.intersection(self.predicted_buggy_lines))
        fp = self.num_predict_buggy_lines - tp
        fn = self.num_actual_buggy_lines - tp
        tn = self.num_total_lines - tp - fp - fn
        print(f'Total lines: {self.num_total_lines}\n'
              f'Buggy lines: {self.num_actual_buggy_lines}\n'
              f'Predicted lines: {len(self.predicted_buggy_lines)}\n'
              f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')

        prec = .0 if tp + fp == .0 else tp / (tp + fp)
        recall = .0 if tp + fn == .0 else tp / (tp + fn)
        far = .0 if fp + tn == 0 else fp / (fp + tn)
        ce = .0 if fn + tn == .0 else fn / (fn + tn)

        d2h = math.sqrt(math.pow(1 - recall, 2) + math.pow(0 - far, 2)) / math.sqrt(2)
        mcc = .0 if tp + fp == .0 or tp + fn == .0 or tn + fp == .0 or tn + fn == .0 else \
            (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        x, y = tp + fp, tp
        n, N = self.num_actual_buggy_lines, self.num_total_lines

        ER = .0 if (y * N) == .0 else (y * N - x * n) / (y * N)
        RI = .0 if (x * n) == 0 else (y * N - x * n) / (x * n)

        ifa, r_20, effort_20 = self.rank_strategy()

        buggy_lines_dict = {}
        total_bugs = len(buggy_lines_dict.keys())
        hit_bugs = set()
        for line in self.predicted_buggy_lines:
            for bug_commit, lines in buggy_lines_dict.items():
                if line in lines:
                    hit_bugs.add(bug_commit)

        ratio = 0 if total_bugs == 0 else round(len(hit_bugs) / total_bugs, 3)

        append_title = True if not os.path.exists(self.line_level_evaluation_file) else False
        title = 'release,line_threshold,precision,recall,far,ce,d2h,mcc,ifa,recall_20,effort@20%recall,ER,RI,ratio\n'
        with open(self.line_level_evaluation_file, 'a') as file:
            file.write(title) if append_title else None
            file.write(f'{self.test_release},{self.line_threshold},{prec},{recall},{far},{ce},{d2h},{mcc},{ifa},{r_20},{effort_20},{ER},{RI},{ratio}\n')
        return

    def rank_strategy(self):
        ranked_predicted_buggy_lines = []
        test_pred_density = [self.test_pred_density[filename] for filename in self.test_filename]

        defective_file_index = [i for i in np.argsort(test_pred_density)[::-1] if self.test_pred_labels[i] == 1]

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]

            temp_lines, temp_scores = [], []
            for index in range(len(self.predicted_buggy_lines)):
                if self.predicted_buggy_lines[index].startswith(defective_filename):
                    temp_lines.append(self.predicted_buggy_lines[index])
                    temp_scores.append(self.predicted_buggy_score[index])

            sorted_index = np.argsort(temp_scores)[::-1]
            ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        return self.get_rank_performance(ranked_predicted_buggy_lines)

    def get_rank_performance(self, ranked_predicted_buggy_lines):
        count, ifa, recall_20, max_effort = 0, 0, 0, int(self.num_total_lines_without_comments * self.threshold_effort)
        has_found = False
        for line in ranked_predicted_buggy_lines[:max_effort]:
            if line in self.oracle_line_set:
                recall_20 += 1
                if not has_found:
                    ifa = count
                    has_found = True
            count += 1

        effort_20_recall = self.num_total_lines_without_comments
        count = 0
        buggy_lines = 0
        for line in ranked_predicted_buggy_lines:
            count += 1
            if line in self.oracle_line_set:
                buggy_lines += 1
                if buggy_lines / self.num_actual_buggy_lines >= 0.2:
                    effort_20_recall = count
                    break

        return ifa, recall_20 / self.num_actual_buggy_lines, effort_20_recall / self.num_total_lines_without_comments

    def save_file_level_result(self):
        data = {'filename': self.test_filename,
                'line_threshold': self.line_threshold,
                'oracle': self.test_labels,
                'predicted_label': self.test_pred_labels,
                'predicted_score': self.test_pred_scores}
        data = pd.DataFrame(data, columns=['filename', 'line_threshold', 'oracle', 'predicted_label', 'predicted_score'])
        data.to_csv(self.file_level_result_file, index=False)

    def save_line_level_result(self):
        data = {'predicted_buggy_lines': self.predicted_buggy_lines,
                'predicted_buggy_score': self.predicted_buggy_score,
                'predicted_density': self.predicted_density}
        data = pd.DataFrame(data, columns=['predicted_buggy_lines', 'predicted_buggy_score', 'predicted_density'])
        data.to_csv(self.line_level_result_file, index=False)

    def save_buggy_density_file(self):
        df = pd.read_csv(self.line_level_result_file)
        self.predicted_buggy_lines = list(df['predicted_buggy_lines'])

        buggy_density, file_buggy_lines_dict = dict(), dict()
        for line in self.predicted_buggy_lines:
            filename = line.strip().split(':')[0]
            if not filename in file_buggy_lines_dict:
                file_buggy_lines_dict[filename] = 1
            else:
                file_buggy_lines_dict[filename] += 1
        for index in range(len(self.test_text_lines)):
            filename = self.test_filename[index]
            if filename not in file_buggy_lines_dict or len(self.test_text_lines[index]) == 0:
                buggy_density[filename] = 0
            else:
                buggy_density[filename] = file_buggy_lines_dict[filename] / len(self.test_text_lines[index])

        self.test_pred_density = buggy_density

        data = {'test_pred_density': self.test_pred_density}
        data = pd.DataFrame(data, columns=['test_pred_density'])
        data.to_csv(self.buggy_density_file, index=False)

    def load_file_level_result(self):
        if len(self.test_pred_labels) == 0 or len(self.test_pred_scores) == 0:
            df = pd.read_csv(self.file_level_result_file)
            self.test_pred_labels = np.array(df['predicted_label'])
            self.test_pred_scores = np.array(df['predicted_score'])

    def load_line_level_result(self):
        if len(self.predicted_buggy_lines) == 0:
            df = pd.read_csv(self.line_level_result_file)
            self.predicted_buggy_lines = list(df['predicted_buggy_lines'])
            self.predicted_buggy_score = list(df['predicted_buggy_score'])
            self.predicted_density = list(df['predicted_density'])
            self.num_predict_buggy_lines = len(self.predicted_buggy_lines)
            self.save_buggy_density_file()
