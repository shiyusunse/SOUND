library(tidyverse)
library(gridExtra)

library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'
  
  return(top.k)
}

prediction_dir = '../output/prediction/DeepLineDP/within-release/0/'

project = 'mng'

filtered_files <- grep("^mng", list.files(prediction_dir), value = TRUE)

all_eval_releases = c('mng-3.1.0')

df_all <- NULL

for(f in filtered_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth,is.comment.line)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5)
line.ground.truth = distinct(line.ground.truth)

get.line.metrics.result = function(baseline.df, cur.df.file, method_name, rel)
{
  baseline.df.with.ground.truth = merge(baseline.df, cur.df.file, by=c("filename", "line.number"))
  
  sorted = baseline.df.with.ground.truth %>%
    filter(is.comment.line == "False") %>%
    filter(line.score > 0) %>%
    arrange(desc(line.score), desc(prediction.prob), line.number, .by_group = TRUE) %>%
    mutate(order = row_number())
  
  write.csv(sorted, file = file.path(method_name, project, paste0(rel, ".csv")), row.names = FALSE)

  IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test)  %>% top_n(1, -order)
  
  ifa.list = IFA %>% select(test, order)
  
  total_true = sorted %>% group_by(test) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
  
  recall20LOC = sorted %>% group_by(test) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
  
  recall.list = recall20LOC$recall20LOC
  
  effort20Recall = sorted %>% merge(total_true) %>% group_by(test) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())
  
  effort.list = effort20Recall$effort20Recall
  
  result.df = data.frame(ifa.list, recall.list, effort.list)
  
  return(result.df)
}



error.prone.result.dir = '../output/ErrorProne_result/'
ngram.result.dir = '../output/n_gram_result/'
n.gram.result.df = NULL
error.prone.result.df = NULL

for(rel in all_eval_releases)
{
  error.prone.result = read.csv(paste0(error.prone.result.dir,rel,'-line-lvl-result.txt'),quote="")

  if (!is.factor(error.prone.result$EP_prediction_result)) {
    error.prone.result$EP_prediction_result = as.factor(error.prone.result$EP_prediction_result)
  }

  levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="False"] = 0
  levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="True"] = 1

  error.prone.result$EP_prediction_result = as.numeric(as.character(error.prone.result$EP_prediction_result))
  
  names(error.prone.result) = c("filename","test","line.number","line.score")
  
  n.gram.result = read.csv(paste0(ngram.result.dir,rel,'-line-lvl-result.txt'), quote = "")
  n.gram.result = select(n.gram.result, "file.name", "test.release", "line.number",  "line.score")
  n.gram.result = distinct(n.gram.result)
  names(n.gram.result) = c("filename", "test","line.number", "line.score")
  
  cur.df.file = filter(line.ground.truth, test==rel)
  cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth, is.comment.line, prediction.prob)
  
  n.gram.eval.result = get.line.metrics.result(n.gram.result, cur.df.file, 'Ngram', rel)
  error.prone.eval.result = get.line.metrics.result(error.prone.result, cur.df.file, 'ErrorProne', rel)
  
  n.gram.result.df = rbind(n.gram.result.df, n.gram.eval.result)
  error.prone.result.df = rbind(error.prone.result.df, error.prone.eval.result)
  
  print(paste0('finished ', rel))
}

df_all[df_all$is.comment.line == "True",]$token.attention.score = 0

tmp.top.k = get.top.k.tokens(df_all, 1500)

merged_df_all = merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)

merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0

sum_line_attn = merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, prediction.prob, line.number, line.level.ground.truth) %>%
  summarize(attention_score = sum(token.attention.score), num_tokens = n())

sorted = sum_line_attn %>%
  filter(is.comment.line == "False") %>%
  filter(attention_score > 0) %>%
  group_by(test) %>%
  arrange(desc(attention_score), desc(prediction.prob), line.number,  .by_group = TRUE) %>%
  mutate(order = row_number())

grouped_data <- sorted %>% group_split(test)

test_names <- sorted %>% group_keys(test) %>% pull(test)

output_path <- file.path('DeepLineDP', project)

dir.create(output_path, recursive = TRUE, showWarnings = FALSE)

walk2(grouped_data, test_names, ~ {
  write.csv(.x, file = file.path(output_path, paste0(.y, ".csv")), row.names = FALSE)
})

IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test) %>% top_n(1, -order)

total_true = sorted %>% group_by(test) %>% summarize(total_true = sum(line.level.ground.truth == "True"))

recall20LOC = sorted %>% group_by(test) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
  summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
  merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)

effort20Recall = sorted %>% merge(total_true) %>% group_by(test) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
  summarise(effort20Recall = sum(recall <= 0.2)/n())
deeplinedp.ifa <- IFA %>% select(test, order) %>% rename(release = test, IFA = order)
results_df <- data.frame(
  release = deeplinedp.ifa$release,
  IFA = deeplinedp.ifa$IFA,
  recall20LOC = recall20LOC$recall20LOC,
  effort20Recall = effort20Recall$effort20Recall
)

write.csv(results_df, file = "DeepLineDP_results.csv", row.names = FALSE)
write.csv(error.prone.result.df, file = "ErrorProne_results.csv", row.names = FALSE)
write.csv(n.gram.result.df, file = "Ngram_results.csv", row.names = FALSE)
