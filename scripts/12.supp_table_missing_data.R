library(finalfit)


df <- read.csv("/data/lxh/prediction/data/processed_data_0214.csv")


df$label <- factor(df$label_WHO) %>% ff_label("Label")


df$Weight <- df$weight2
df$Height <- df$height2 * 100 # cm
df$WHR <- df$WHR * 100
df$WHtR <- df$WHtR * 100

df$Diabetes_family_history <- factor(df$ldiafamily)
df$Wake_time <- df$lgetup
df$Education <- factor(df$culutrue)
df$Work <- factor(df$lwork)
df$Daytime_sleep_duraiton <- df$ntime
df$Years_cellphone_use <- df$lusephy

# Examine with ff_glimpse
explanatory <- c(
    "Diabetes_family_history", "Wake_time", "Education", "Work",
    "Daytime_sleep_duraiton", "Years_cellphone_use",
    "ASBP", "ADBP", "Ahr", "Weight", "Height", "wc", "hc"
)

dependent <- "label"

df %>% ff_glimpse(dependent, explanatory)

df[explanatory] %>% missing_plot()

df %>% finalfit(dependent, explanatory)
