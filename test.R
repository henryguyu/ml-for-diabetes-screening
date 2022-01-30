library(finalfit)


df <- read.csv("/data/lxh/prediction/data/processed_data_0214.csv")


df$label <- factor(df$label_WHO) %>% ff_label("Label")
df$lwork <- factor(df$lwork) %>% ff_label("Work")
df$tea <- factor(df$tea) %>% ff_label("Tea")
df$WHR100 <- df$WHR * 100


# Examine with ff_glimpse
explanatory <- c(
    "Ahr", "age",
    "WHR100", "ASBP",
    "lwork", "BMI", "tea"
)
dependent <- "label"

df %>% ff_glimpse(dependent, explanatory)

#df[explanatory] %>% missing_plot()

#df %>% missing_pattern(dependent, explanatory)

# df %>% finalfit(dependent, explanatory)
