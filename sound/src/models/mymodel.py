import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from glance import BaseModel
import json


class BarinelWithoutCA(BaseModel):
    model_name = 'BarinelWithoutCA'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

    def line_level_prediction(self):
        super(BarinelWithoutCA, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')

        predicted_lines, predicted_score, predicted_density = [], [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()

        with open(self.barinel_score_name, 'r') as f:
            token_score = json.load(f)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]

            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1
            sorted_index_by_buggy = np.argsort(buggy_count).tolist()

            sorted_index = [i for i in sorted_index_by_buggy if hit_count[i] > 0.0]

            predicted_score.extend([hit_count[i] for i in sorted_index])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0
            predicted_density.extend([density for i in sorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = []

        temp_lines, temp_scores = [], []
        for index in range(len(self.predicted_buggy_lines)):
            temp_lines.append(self.predicted_buggy_lines[index])
            temp_scores.append(self.predicted_buggy_score[index])

        indexed_lst = [(i, x) for i, x in enumerate(temp_scores)]
        sorted_lst = sorted(indexed_lst, key=lambda x: (-x[1], x[0]))
        sorted_index = [x[0] for x in sorted_lst]

        ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        return self.get_rank_performance(ranked_predicted_buggy_lines)


class DstarWithoutCA(BaseModel):
    model_name = 'DstarWithoutCA'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

    def line_level_prediction(self):
        super(DstarWithoutCA, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')

        predicted_lines, predicted_score, predicted_density = [], [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()

        with open(self.dstar_score_name, 'r') as f:
            token_score = json.load(f)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]

            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sorted_index_by_buggy = np.argsort(buggy_count).tolist()

            sorted_index = [i for i in sorted_index_by_buggy if hit_count[i] > 0.0]

            predicted_score.extend([hit_count[i] for i in sorted_index])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0
            predicted_density.extend([density for i in sorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = []

        temp_lines, temp_scores = [], []
        for index in range(len(self.predicted_buggy_lines)):
            temp_lines.append(self.predicted_buggy_lines[index])
            temp_scores.append(self.predicted_buggy_score[index])

        indexed_lst = [(i, x) for i, x in enumerate(temp_scores)]
        sorted_lst = sorted(indexed_lst, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_lst]

        ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        return self.get_rank_performance(ranked_predicted_buggy_lines)


class OchiaiWithoutCA(BaseModel):
    model_name = 'OchiaiWithoutCA'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

    def line_level_prediction(self):
        super(OchiaiWithoutCA, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')

        predicted_lines, predicted_score, predicted_density = [], [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()

        with open(self.ochiai_score_name, 'r') as f:
            token_score = json.load(f)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]

            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sorted_index_by_buggy = np.argsort(buggy_count).tolist()

            sorted_index = [i for i in sorted_index_by_buggy if hit_count[i] > 0.0]

            predicted_score.extend([hit_count[i] for i in sorted_index])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0
            predicted_density.extend([density for i in sorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = []

        temp_lines, temp_scores = [], []
        for index in range(len(self.predicted_buggy_lines)):
            temp_lines.append(self.predicted_buggy_lines[index])
            temp_scores.append(self.predicted_buggy_score[index])

        indexed_lst = [(i, x) for i, x in enumerate(temp_scores)]
        sorted_lst = sorted(indexed_lst, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_lst]

        ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Op2WithoutCA(BaseModel):
    model_name = 'Op2WithoutCA'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

    def line_level_prediction(self):
        super(Op2WithoutCA, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')

        predicted_lines, predicted_score, predicted_density = [], [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()

        with open(self.op2_score_name, 'r') as f:
            token_score = json.load(f)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]

            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sorted_index_by_buggy = np.argsort(buggy_count).tolist()

            sorted_index = [i for i in sorted_index_by_buggy if hit_count[i] > 0.0]

            predicted_score.extend([hit_count[i] for i in sorted_index])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0
            predicted_density.extend([density for i in sorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = []

        temp_lines, temp_scores = [], []
        for index in range(len(self.predicted_buggy_lines)):
            temp_lines.append(self.predicted_buggy_lines[index])
            temp_scores.append(self.predicted_buggy_score[index])

        indexed_lst = [(i, x) for i, x in enumerate(temp_scores)]
        sorted_lst = sorted(indexed_lst, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_lst]

        ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        return self.get_rank_performance(ranked_predicted_buggy_lines)


class TarantulaWithoutCA(BaseModel):
    model_name = 'TarantulaWithoutCA'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

    def line_level_prediction(self):
        super(TarantulaWithoutCA, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')

        predicted_lines, predicted_score, predicted_density = [], [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()

        with open(self.tarantula_score_name, 'r') as f:
            token_score = json.load(f)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]

            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sorted_index_by_buggy = np.argsort(buggy_count).tolist()

            sorted_index = [i for i in sorted_index_by_buggy if hit_count[i] > 0.0]

            predicted_score.extend([hit_count[i] for i in sorted_index])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0
            predicted_density.extend([density for i in sorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = []

        temp_lines, temp_scores = [], []
        for index in range(len(self.predicted_buggy_lines)):
            temp_lines.append(self.predicted_buggy_lines[index])
            temp_scores.append(self.predicted_buggy_score[index])

        indexed_lst = [(i, x) for i, x in enumerate(temp_scores)]
        sorted_lst = sorted(indexed_lst, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_lst]

        ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Barinel(BaseModel):
    model_name = 'Barinel'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Barinel, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()
        with open(self.barinel_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.barinel_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Barinel_without_filevel(BaseModel):
    model_name = 'Barinel_without_filevel'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Barinel_without_filevel, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        defective_file_index = sorted(range(len(self.test_text_lines)), key=lambda k: len(self.test_text_lines[k]))

        tokenizer = self.vector.build_tokenizer()
        with open(self.barinel_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.barinel_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0.0
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class BarinelFFLLSort(BaseModel):
    model_name = 'BarinelFFLLSort'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(BarinelFFLLSort, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                     )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()

        with open(self.barinel_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.barinel_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            index_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('* '):
                    hit_count[line_index] = 0.0
                    cc_count = False
                index_count[line_index] = line_index

            sort_key = list(zip(hit_count, index_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            sorted_index_buggy_ratio = [x[0] for x in sorted_index][:int(self.num_total_lines_without_comments * self.line_threshold)]

            resorted_index = [i for i in sorted_index_buggy_ratio if cc_count[i]]
            resorted_index_remain = [i for i in sorted_index if i not in resorted_index]

            predicted_score.extend([hit_count[i] for i in resorted_index])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in resorted_index])

            predicted_score.extend([hit_count[i] for i in resorted_index_remain])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in resorted_index_remain])

            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0
            predicted_density.extend([density for i in sorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Dstar(BaseModel):
    model_name = 'Dstar'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Dstar, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()
        with open(self.dstar_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.dstar_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Dstar_without_filevel(BaseModel):
    model_name = 'Dstar_without_filevel'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Dstar_without_filevel, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        defective_file_index = sorted(range(len(self.test_text_lines)), key=lambda k: len(self.test_text_lines[k]))

        tokenizer = self.vector.build_tokenizer()
        with open(self.dstar_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.dstar_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0.0
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Ochiai(BaseModel):
    model_name = 'Ochiai'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Ochiai, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()
        with open(self.ochiai_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.ochiai_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Ochiai_without_filevel(BaseModel):
    model_name = 'Ochiai_without_filevel'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Ochiai_without_filevel, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        defective_file_index = sorted(range(len(self.test_text_lines)), key=lambda k: len(self.test_text_lines[k]))

        tokenizer = self.vector.build_tokenizer()
        with open(self.ochiai_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.ochiai_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0.0
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Op2(BaseModel):
    model_name = 'Op2'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Op2, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')

        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()
        with open(self.op2_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.op2_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = sorted_index[:int(self.num_total_lines_without_comments * self.line_threshold)]
        self.resort_lines = int(self.num_total_lines_without_comments * self.line_threshold)

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        self.causal_lines = len(resorted_index)
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Op2_without_filevel(BaseModel):
    model_name = 'Op2_without_filevel'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Op2_without_filevel, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        defective_file_index = sorted(range(len(self.test_text_lines)), key=lambda k: len(self.test_text_lines[k]))

        tokenizer = self.vector.build_tokenizer()
        with open(self.op2_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.op2_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0.0
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]


        sorted_index_buggy_ratio = sorted_index[:int(self.num_total_lines_without_comments * self.line_threshold)]
        self.resort_lines = int(self.num_total_lines_without_comments * self.line_threshold)

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        self.causal_lines = len(resorted_index)
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Tarantula(BaseModel):
    model_name = 'Tarantula'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Tarantula, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]

        tokenizer = self.vector.build_tokenizer()
        with open(self.tarantula_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.tarantula_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Tarantula_without_filevel(BaseModel):
    model_name = 'Tarantula_without_filevel'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments

    def read_graph(self, file_path):
        tmp = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            token = line.split()
            if len(token) > 1:
                num = float(token[3].replace(":", ""))
                tmp[token[0]] = num
        return tmp

    def line_level_prediction(self):
        super(Tarantula_without_filevel, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_raw, predicted_score_raw, predicted_density_raw, cc_count_all = [], [], [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []
        predicted_density_cc_count_0, predicted_density_cc_count_1 = [], []

        defective_file_index = sorted(range(len(self.test_text_lines)), key=lambda k: len(self.test_text_lines[k]))

        tokenizer = self.vector.build_tokenizer()
        with open(self.tarantula_score_name, 'r') as f:
            token_score = json.load(f)
        token_causal_score = self.read_graph(self.tarantula_graph_name)

        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            buggy_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in token_score:
                        hit_count[line_index] += token_score[token]
                    if token in token_causal_score:
                        if token_causal_score[token] >= 0.99:
                            cc_count[line_index] = True
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0.0
                    cc_count[line_index] = False

                if line_index in self.oracle_line_dict[defective_filename]:
                    buggy_count[line_index] = 1

            sort_key = list(zip(hit_count, buggy_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0.0]

            predicted_score_raw.extend([hit_count[i] for i in sorted_index])
            predicted_lines_raw.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            if not len(hit_count) == 0:
                density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            else:
                density = 0.0
            predicted_density_raw.extend([density for i in sorted_index])
            cc_count_all.extend([cc_count[i] for i in sorted_index])

        index_list = [(i, x) for i, x in enumerate(predicted_score_raw)]
        sorted_list = sorted(index_list, key=lambda x: (x[1], -x[0]), reverse=True)
        sorted_index = [x[0] for x in sorted_list]

        sorted_index_buggy_ratio = [x[0] for x in sorted_list][:int(self.num_total_lines_without_comments * self.line_threshold)]

        resorted_index = [i for i in sorted_index_buggy_ratio if cc_count_all[i]]
        predicted_score_cc_count_1.extend([predicted_score_raw[i] for i in resorted_index])
        predicted_lines_cc_count_1.extend([predicted_lines_raw[i] for i in resorted_index])
        predicted_density_cc_count_1.extend([predicted_density_raw[i] for i in resorted_index])

        resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
        predicted_score_cc_count_0.extend([predicted_score_raw[i] for i in resorted_index_remain])
        predicted_lines_cc_count_0.extend([predicted_lines_raw[i] for i in resorted_index_remain])
        predicted_density_cc_count_0.extend([predicted_density_raw[i] for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
            predicted_density.extend([predicted_density_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])
            predicted_density.extend([predicted_density_cc_count_0[indexes_0[i]]])
        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class DeepLineDP(BaseModel):
    model_name = 'DeepLineDP'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)


class Ngram(BaseModel):
    model_name = 'Ngram'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)


class ErrorProne(BaseModel):
    model_name = 'ErrorProne'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

