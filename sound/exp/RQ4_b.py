import pandas as pd

file1 = pd.read_csv("../Result/Barinel/line_result/evaluation.csv")
file2 = pd.read_csv("../Result/BarinelWithoutCA/line_result/evaluation.csv")

data = pd.merge(file1, file2, on="release", how="inner")
print(data.columns)

data_to_save = data[["release", "ifa_x", "ifa_y", "effort@20%recall_x", "effort@20%recall_y"]]

data_to_save.columns = ["release", "ifa_file1", "ifa_file2", "effort_file1", "effort_file2"]

data_to_save = data_to_save.sort_values("ifa_file2", ascending=False)

data_to_save.to_csv("RQ4_b_result.csv", index=False)

n = int(len(data_to_save) * 0.25)

data_subset_front = data_to_save.head(n)
data_subset_front.to_csv("Non_effective.csv", index=False)

data_subset_back = data_to_save.tail(n)
data_subset_back.to_csv("Effective.csv", index=False)


