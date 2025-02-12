from sound.src.utils.config import *
from sound.src.utils.helper import *
from base_model import BaseModel


root_path = r'D:/line-level-defect-prediction-master/'
dataset_string = 'Dataset'
result_string = 'Result'
result_path = f'{root_path}{result_string}'

file_level_path = f'{root_path}{dataset_string}/File-level/'
line_level_path = f'{root_path}{dataset_string}/Line-level/'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'


def call_number(statement):
    statement = statement.strip('\"')
    score = 0
    for char in statement:
        if char == '(':
            score += 1
    return score


class Glance_Mixed_Sort(BaseModel):
    model_name = 'Glance_Mixed_Sort'

    def __init__(self, train_release='', test_release='', line_threshold=0.5, test_result_path='', is_realistic=False):

        super().__init__(train_release, test_release, test_result_path, is_realistic)
        self.line_threshold = line_threshold
        self.tokenizer = self.vector.build_tokenizer()
        self.tags = ['todo', 'hack', 'fixme', 'xxx']

    def line_level_prediction(self):
        super(Glance_Mixed_Sort, self).line_level_prediction()
        if USE_CACHE and os.path.exists(self.line_level_result_file):
            return

        predicted_lines, predicted_score = [], []

        predicted_lines_cc_count_0, predicted_lines_cc_count_1 = [], []
        predicted_score_cc_count_0, predicted_score_cc_count_1 = [], []

        predicted_lines_cc_count_0_clean, predicted_lines_cc_count_1_clean = [], []
        predicted_score_cc_count_0_clean, predicted_score_cc_count_1_clean = [], []

        sort_key = list(
            zip(self.test_pred_scores,
                list(
                    len(self.test_text_lines[i]) for i in range(len(self.test_pred_scores))
                )
                )
        )
        sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]

        defective_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 1]
        clean_file_index = [i for i in sorted_index if self.test_pred_labels[i] == 0]
        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)
            line_index_count = np.zeros(num_of_lines, dtype=int)
            for line_index in range(num_of_lines):
                line_index_count[line_index] = line_index
                tokens_in_line = self.tokenizer(defective_file_line_list[line_index])
                nt = len(tokens_in_line)
                nfc = call_number(defective_file_line_list[line_index])
                if nt == 0:
                    hit_count[line_index] = 0
                else:
                    hit_count[line_index] = nt * (nfc + 1)

                if 'for' in tokens_in_line:
                    cc_count[line_index] = True
                if 'while' in tokens_in_line:
                    cc_count[line_index] = True
                if 'do' in tokens_in_line:
                    cc_count[line_index] = True
                if 'if' in tokens_in_line:
                    cc_count[line_index] = True
                if 'else' in tokens_in_line:
                    cc_count[line_index] = True
                if 'switch' in tokens_in_line:
                    cc_count[line_index] = True
                if 'case' in tokens_in_line:
                    cc_count[line_index] = True
                if 'continue' in tokens_in_line:
                    cc_count[line_index] = True
                if 'break' in tokens_in_line:
                    cc_count[line_index] = True
                if 'return' in tokens_in_line:
                    cc_count[line_index] = True

                line = defective_file_line_list[line_index]
                if line.startswith('/') or line.startswith('*'):
                    hit_count[line_index] = -1
                    cc_count[line_index] = False

            sort_key = list(zip(hit_count, line_index_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] >= 0]

            resorted_index = [i for i in sorted_index[:int(len(sorted_index) * 0.5)] if cc_count[i]]
            predicted_score_cc_count_1.extend([hit_count[i] for i in resorted_index])
            predicted_lines_cc_count_1.extend([f'{defective_filename}:{i + 1}' for i in resorted_index])

            resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
            predicted_score_cc_count_0.extend([hit_count[i] for i in resorted_index_remain])
            predicted_lines_cc_count_0.extend([f'{defective_filename}:{i + 1}' for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0[indexes_0[i]]])

        for i in range(len(clean_file_index)):
            clean_filename = self.test_filename[clean_file_index[i]]
            if clean_filename not in self.oracle_line_dict:
                self.oracle_line_dict[clean_filename] = []
            clean_file_line_list = self.test_text_lines[clean_file_index[i]]

            num_of_lines = len(clean_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=int)
            cc_count = np.zeros(num_of_lines, dtype=bool)
            line_index_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                line_index_count[line_index] = line_index
                tokens_in_line = self.tokenizer(clean_file_line_list[line_index])
                nt = len(tokens_in_line)
                nfc = call_number(clean_file_line_list[line_index])
                if nt == 0:
                    hit_count[line_index] = 0
                else:
                    hit_count[line_index] = nt * (nfc + 1)

                if 'for' in tokens_in_line:
                    cc_count[line_index] = True
                if 'while' in tokens_in_line:
                    cc_count[line_index] = True
                if 'do' in tokens_in_line:
                    cc_count[line_index] = True
                if 'if' in tokens_in_line:
                    cc_count[line_index] = True
                if 'else' in tokens_in_line:
                    cc_count[line_index] = True
                if 'switch' in tokens_in_line:
                    cc_count[line_index] = True
                if 'case' in tokens_in_line:
                    cc_count[line_index] = True
                if 'continue' in tokens_in_line:
                    cc_count[line_index] = True
                if 'break' in tokens_in_line:
                    cc_count[line_index] = True
                if 'return' in tokens_in_line:
                    cc_count[line_index] = True

                line = clean_file_line_list[line_index]
                if line.startswith('/') or line.startswith('*'):
                    hit_count[line_index] = -1
                    cc_count[line_index] = False

            sort_key = list(zip(hit_count, line_index_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] >= 0]

            resorted_index = [i for i in sorted_index[:int(len(hit_count) * 0.5)] if cc_count[i]]
            predicted_score_cc_count_1_clean.extend([hit_count[i] for i in resorted_index])
            predicted_lines_cc_count_1_clean.extend([f'{clean_filename}:{i + 1}' for i in resorted_index])

            resorted_index_remain = [i for i in sorted_index if i not in resorted_index]
            predicted_score_cc_count_0_clean.extend([hit_count[i] for i in resorted_index_remain])
            predicted_lines_cc_count_0_clean.extend([f'{clean_filename}:{i + 1}' for i in resorted_index_remain])

        indexed_lst_1 = [(i, x) for i, x in enumerate(predicted_score_cc_count_1_clean)]
        sorted_lst_1 = sorted(indexed_lst_1, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_1 = [x[0] for x in sorted_lst_1]

        indexed_lst_0 = [(i, x) for i, x in enumerate(predicted_score_cc_count_0_clean)]
        sorted_lst_0 = sorted(indexed_lst_0, key=lambda x: (x[1], -x[0]), reverse=True)
        indexes_0 = [x[0] for x in sorted_lst_0]

        for i in range(len(indexes_1)):
            predicted_lines.extend([predicted_lines_cc_count_1_clean[indexes_1[i]]])
            predicted_score.extend([predicted_score_cc_count_1_clean[indexes_1[i]]])
        for i in range(len(indexes_0)):
            predicted_lines.extend([predicted_lines_cc_count_0_clean[indexes_0[i]]])
            predicted_score.extend([predicted_score_cc_count_0_clean[indexes_0[i]]])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()

    def rank_strategy(self):
        ranked_predicted_buggy_lines = self.predicted_buggy_lines
        max_effort = int(self.num_total_lines_without_comments * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')
        return self.get_rank_performance(ranked_predicted_buggy_lines)


class Glance_LR_Mixed_Sort(Glance_Mixed_Sort):
    model_name = 'BASE-Glance-LR_Mixed_Sort'

    def __init__(self, train_release='', test_release='', line_threshold=0.5, test=False, is_realistic=False):
        test_result_path = ''
        if test:
            self.model_name = f'Glance-LR-{str(int(line_threshold * 100))}'
            test_result_path = f'{root_path}Result/Dis1/{self.model_name}/'
        super().__init__(train_release, test_release, line_threshold, test_result_path, is_realistic)

    def file_level_prediction(self):
        print(f"Prediction\t=>\t{self.test_release}")
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        train_vtr = self.vector.fit_transform(self.train_text)
        test_vtr = self.vector.transform(self.test_text)
        self.clf.fit(train_vtr, self.train_label)

        self.test_pred_labels = self.clf.predict(test_vtr)
        self.test_pred_scores = np.array([score[1] for score in self.clf.predict_proba(test_vtr)])

        self.save_file_level_result()


class Glance_EA_Mixed_Sort(Glance_Mixed_Sort):
    model_name = 'BASE-Glance-EA_Mixed_Sort'

    def __init__(self, train_release='', test_release='', line_threshold=0.5, file_threshold=0.5, test=False,
                 is_realistic=False):
        test_result_path = ''
        if test:
            self.model_name = f'Glance-EA-{str(int(file_threshold * 100))}-{str(int(line_threshold * 100))}'
            test_result_path = f'{root_path}Result/Dis1/{self.model_name}/'
        super().__init__(train_release, test_release, line_threshold, test_result_path, is_realistic)
        self.file_threshold = file_threshold

    def file_level_prediction(self):
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        num_of_files = len(self.test_text)
        test_prediction = np.zeros(num_of_files, dtype=int).tolist()

        loc, debts, score = [], [], []
        for file_index in range(num_of_files):
            loc.append(len([line for line in self.test_text_lines[file_index] if line.strip() != '']))
            debts.append(len([tag for tag in self.tags if tag in self.test_text[file_index].lower()]))

        score = loc
        effort_all, effort_acc = sum(loc), 0
        sorted_index = np.argsort(score).tolist()[::-1]

        file_count = 0
        for index in sorted_index:
            if effort_acc < effort_all * self.file_threshold:
                test_prediction[index] = 1
                effort_acc += loc[index]
                file_count += 1
            else:
                break

        self.test_pred_labels = test_prediction
        self.test_pred_scores = np.array(score)

        self.save_file_level_result()


class Glance_MD_Mixed_Sort(Glance_Mixed_Sort):
    model_name = 'BASE-Glance-MD_Mixed_Sort'

    def __init__(self, train_release='', test_release='', line_threshold=0.5, file_threshold=0.5, test=False,
                 is_realistic=False):
        test_result_path = ''
        if test:
            self.model_name = f'Glance-MD-{str(int(file_threshold * 100))}-{str(int(line_threshold * 100))}'
            test_result_path = f'{root_path}Result/Dis1/{self.model_name}/'
        super().__init__(train_release, test_release, line_threshold, test_result_path, is_realistic)
        self.file_threshold = file_threshold

    def file_level_prediction(self):
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        num_of_files = len(self.test_text)
        test_prediction = np.zeros(num_of_files, dtype=int).tolist()

        loc, debts, score = [], [], []
        for file_index in range(num_of_files):
            loc.append(len([line for line in self.test_text_lines[file_index] if line.strip() != '']))
            debts.append(len([tag for tag in self.tags if tag in self.test_text[file_index].lower()]))

        score = loc
        sorted_index = np.argsort(score).tolist()[::-1]

        file_count = 0
        for index in sorted_index:
            if file_count <= len(loc) * self.file_threshold:
                test_prediction[index] = 1
                file_count += 1
            else:
                break

        self.test_pred_labels = test_prediction
        self.test_pred_scores = np.array(score)

        self.save_file_level_result()

