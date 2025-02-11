import pandas as pd
import glob
import os

for project in ['ambari', 'amq','bookkeeper', 'calcite', 'cassandra', 'flink', 'groovy', 'hbase', 'hive', 'ignite',
                'log4j2', 'mahout', 'mng', 'nifi', 'nutch', 'storm', 'tika', 'ww', 'zookeeper']:
    for method in ['DeepLineDP', 'ErrorProne', 'Ngram']:
        csv_folder = "../Result/"+method+"/line_result/"+project
        print(csv_folder)

        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

        if method == 'DeepLineDP':
            columns = ["filename", "line.number", "attention_score", "prediction.prob"]
        else:
            columns = ["filename", "line.number", "line.score", "prediction.prob"]

        for file in csv_files:
            df = pd.read_csv(file, usecols=columns)

            df['predicted_buggy_lines'] = df['filename'].str.replace('"', '') + ':' + df['line.number'].astype(str)

            if method == 'DeepLineDP':
                df = df.rename(columns={"attention_score": "predicted_buggy_score", "prediction.prob": "predicted_density"})
            else:
                df = df.rename(columns={"line.score": "predicted_buggy_score", "prediction.prob": "predicted_density"})

            df = df[["predicted_buggy_lines", "predicted_buggy_score", "predicted_density"]]

            output_file = os.path.join(csv_folder, os.path.basename(file).replace(".csv", "-result.csv"))
            df.to_csv(output_file, index=False)

            print(f"Processed file saved as: {output_file}")
