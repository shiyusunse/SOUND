import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from glance import BaseModel
from lime.lime_text import LimeTextExplainer


class LineDP_mixedsort(BaseModel):
    model_name = 'MIT-LineDP_mixedsort'

    def __init__(self, train_release: str = '', test_release: str = '', is_realistic=False):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

    def line_level_prediction(self):
        super(LineDP_mixedsort, self).line_level_prediction()
        if os.path.exists(self.line_level_result_file):
            return

        if len(self.test_pred_labels) == 0 or len(self.test_pred_scores) == 0:
            df = pd.read_csv(self.file_level_result_file)
            self.test_pred_labels = np.array(df['predicted_label'])
            self.test_pred_scores = np.array(df['predicted_score'])

        print(f'Predicting line level defect prediction of {self.model_name}')
        predicted_lines, predicted_score, predicted_density = [], [], []

        predicted_lines_initial, predicted_score_initial, predicted_density_initial = [], [], []

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
        c = make_pipeline(self.vector, self.clf)
        explainer = LimeTextExplainer(class_names=['defect', 'non-defect'], random_state=self.random_state)

        for i in range(len(defective_file_index)):
            print(f'{i}/{len(defective_file_index)}')
            defective_filename = self.test_filename[defective_file_index[i]]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            exp = explainer.explain_instance(' '.join(defective_file_line_list), c.predict_proba, num_features=100)
            risky_tokens = [x[0] for x in exp.as_list() if x[1] > 0][:20]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=int)
            line_index_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                line_index_count[line_index] = line_index
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                for token in tokens_in_line:
                    if token in risky_tokens:
                        hit_count[line_index] += 1
                if defective_file_line_list[line_index].startswith('*') or defective_file_line_list[line_index].startswith('/'):
                    hit_count[line_index] = 0


            sort_key = list(zip(hit_count, line_index_count))
            sorted_index = [i[0] for i in sorted(enumerate(sort_key), key=lambda x: (-x[1][0], x[1][1]))]
            sorted_index = [i for i in sorted_index if hit_count[i] > 0]

            predicted_score_initial.extend([hit_count[i] for i in sorted_index])
            predicted_lines_initial.extend([f'{defective_filename}:{i + 1}' for i in sorted_index])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density_initial.extend([density for i in sorted_index])

        indexed_lst = [(i, x) for i, x in enumerate(predicted_score_initial)]
        sorted_lst_1 = sorted(indexed_lst, key=lambda x: (-x[1], x[0]))
        indexes = [x[0] for x in sorted_lst_1]

        for i in range(len(indexes)):
            predicted_lines.extend([predicted_lines_initial[indexes[i]]])
            predicted_score.extend([predicted_score_initial[indexes[i]]])
            predicted_density.extend([predicted_density_initial[indexes[i]]])

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
