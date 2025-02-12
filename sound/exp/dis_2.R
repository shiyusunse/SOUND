library(ggplot2)
library(scales)

file_names <- c("Barinel.csv", "Barinel_file.csv")

data <- data.frame()

group_order <- c("Barinel", "Barinel_file")

for (name in file_names) {
  temp_data <- read.csv(name)

  temp_data_ifa <- data.frame(ifa = (temp_data$ifa + 1), group = name)


  temp_data_ifa$group <- gsub(".csv", "", temp_data_ifa$group)
  temp_data_ifa$group <- factor(temp_data_ifa$group, levels = group_order)

  data <- rbind(data, temp_data_ifa)
}
group_colors <- c("Barinel" = "#F8766D", "Tarantula" = "#F8766D", "Ochiai" = "#F8766D",
                  "Dstar" = "#F8766D", "Op2" = "#F8766D", "DeepLineDP" = "#00BFC4","GLANCE-LR" = "#00BFC4",
                  "LineDP" = "#00BFC4", "GLANCE-EA" = "#00BFC4", "GLANCE-MD" = "#00BFC4", "ErrorProne" = "#00BFC4", "Ngram" = "#00BFC4", "Barinel_file" = "#00BFC4")


medians <- aggregate(ifa ~ group, data = data, FUN = median)

data$group <- with(data, reorder(group, ifa, median))


ggplot(data, aes(x = group, y = ifa)) +
  geom_boxplot(aes(fill=group), outlier.shape = NA) +
  scale_fill_manual(values = group_colors) +
  stat_summary(fun.y = mean, geom = "point", shape = 23, size=2, fill = "white", color = "black") +
  theme_minimal() +
  labs(x = "", y = "", title = "") +
  coord_trans(y="log10") +
  scale_y_continuous(
    breaks = c(1, 10, 100, 1000),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  guides(fill=FALSE) +
  theme(axis.text.x=element_blank(),
        axis.title.y=element_text(size=14),
        axis.text.y=element_text(size=24))
ggsave(paste('./result/dis_2_IFA.pdf',sep=""),width = 3,height = 6,dpi=200)

data <- data.frame()

group_order <- c("Barinel", "Barinel_file")

for (name in file_names) {
  temp_data <- read.csv(name)
  
  temp_data_recall <- data.frame(recall_20 = temp_data$recall_20, group = name)
  
  temp_data_recall$group <- gsub(".csv", "", temp_data_recall$group)
  temp_data_recall$group <- factor(temp_data_recall$group, levels = group_order)
  
  data <- rbind(data, temp_data_recall)
}
group_colors <- c("Barinel" = "#F8766D", "Tarantula" = "#F8766D", "Ochiai" = "#F8766D",
                  "Dstar" = "#F8766D", "Op2" = "#F8766D", "DeepLineDP" = "#00BFC4", "GLANCE-LR" = "#00BFC4",
                  "LineDP" = "#00BFC4", "GLANCE-EA" = "#00BFC4", "GLANCE-MD" = "#00BFC4", "ErrorProne" = "#00BFC4", "Ngram" = "#00BFC4", "Barinel_file" = "#00BFC4")

medians <- aggregate(recall_20 ~ group, data = data, FUN = median)

data$group <- with(data, reorder(group, recall_20, median))


ggplot(data, aes(x = group, y = recall_20)) +
  geom_boxplot(aes(fill=group), outlier.shape = NA) +
  scale_fill_manual(values = group_colors) +
  stat_summary(fun.y = mean, geom = "point", shape = 23, size=2, fill = "white", color = "black") +
  theme_minimal() +
  labs(x = "", y = "", title = "") +
  scale_x_discrete(limits = group_order) +
  guides(fill=FALSE) +
  theme(axis.text.x=element_blank(),
        axis.title.y=element_text(size=14),
        axis.text.y=element_text(size=24))
ggsave(paste('./result/dis_2_recall.pdf',sep=""),width = 3,height = 6,dpi=200)


data <- data.frame()

group_order <- c("Barinel", "Barinel_file")

for (name in file_names) {
  temp_data <- read.csv(name)
  
  temp_data_effort <- data.frame(effort = temp_data$effort, group = name)
  
  temp_data_effort$group <- gsub(".csv", "", temp_data_effort$group)
  temp_data_effort$group <- factor(temp_data_effort$group, levels = group_order)
  
  data <- rbind(data, temp_data_effort)
}
group_colors <- c("Barinel" = "#F8766D", "Tarantula" = "#F8766D", "Ochiai" = "#F8766D",
                  "Dstar" = "#F8766D", "Op2" = "#F8766D", "DeepLineDP" = "#00BFC4","GLANCE-LR" = "#00BFC4",
                  "LineDP" = "#00BFC4", "GLANCE-EA" = "#00BFC4", "GLANCE-MD" = "#00BFC4", "Ngram" = "#00BFC4", "ErrorProne" = "#00BFC4", "Barinel_file" = "#00BFC4")

medians <- aggregate(effort ~ group, data = data, FUN = median)

data$group <- with(data, reorder(group, effort, median))

ggplot(data, aes(x = group, y = effort)) +
  geom_boxplot(aes(fill=group), outlier.shape = NA) +
  scale_fill_manual(values = group_colors) +
  stat_summary(fun.y = mean, geom = "point", shape = 23, size=2, fill = "white", color = "black") +
  theme_minimal() +
  labs(x = "", y = "", title = "") +
  scale_x_discrete(limits = group_order) +
  coord_cartesian(ylim = c(0.00, 0.3)) +
  guides(fill=FALSE) +
  theme(axis.text.x=element_blank(),
        axis.title.y=element_text(size=14),
        axis.text.y=element_text(size=24))
ggsave(paste('./result/dis_2_effort.pdf',sep=""),width = 3,height = 6,dpi=200)

