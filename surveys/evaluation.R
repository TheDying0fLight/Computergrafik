# File location
setwd("C:/Users/meike/Documents/Uni/Master/Semester 5/Computergrafik/surveys")

# Clean environment
rm(list = ls(all = TRUE))

# Packages
#install.packages("readxl")
#install.packages("tidyverse")
#install.packages("ggplot2")
#install.packages("Rmisc")
library(ggplot2)
library(Rmisc)
library(readxl)
library(tidyverse)
library(RColorBrewer)
require(ez)
require(schoRsch)
library(dplyr)

# --------
# SURVEY 1
# --------

# Load data
dat_Shrek <- data.frame(read_excel("Shrek.xlsx"))
dat_PW <- data.frame(read_excel("PrettyWoman.xlsx"))
dat_D <- data.frame(read_excel("Deadpool.xlsx"))
dat_main <- data.frame(read_excel("ForThumbnail.xlsx"))

# Delet participant who does not know the movie
dat_PW <- dat_PW[dat_PW$Know.PW == 1,]
dat_D <- dat_D[dat_D$Know.Deadpool == 1,]
dat_Shrek <- dat_Shrek[dat_Shrek$Know.S == 1,]

# Raw dataframe
raw_data <- data.frame(
  "Vp" = c(rep(dat_PW$Vp, 4), rep(dat_D$Vp, 4), rep(dat_Shrek$Vp, 4)),
  "Moon F" = c(dat_PW$PW.Moon.F.1, dat_PW$PW.Moon.F.2, dat_PW$PW.Moon.F.3, dat_PW$PW.Moon.F.4, dat_D$D.Moon.F.1, dat_D$D.Moon.F.2, dat_D$D.Moon.F.3, dat_D$D.Moon.F.4, dat_Shrek$S.Moon.F.1,
                    dat_Shrek$S.Moon.F.2, dat_Shrek$S.Moon.F.3, dat_Shrek$S.Moon.F.4),
  "Moon" = c(dat_PW$PW.Moon.1, dat_PW$PW.Moon.2, dat_PW$PW.Moon.3, dat_PW$PW.Moon.4, dat_D$D.Moon.1, dat_D$D.Moon.2, dat_D$D.Moon.3, dat_D$D.Moon.4, dat_Shrek$S.Moon.1,
                  dat_Shrek$S.Moon.2, dat_Shrek$S.Moon.3, dat_Shrek$S.Moon.4),
  "Gem" = c(dat_PW$PW.Gem.1, dat_PW$PW.Gem.2, dat_PW$PW.Gem.3, dat_PW$PW.Gem.4, dat_D$D.Gem.1, dat_D$D.Gem.2, dat_D$D.Gem.3, dat_D$D.Gem.4, dat_Shrek$S.Gem.1,
                 dat_Shrek$S.Gem.2, dat_Shrek$S.Gem.3, dat_Shrek$S.Gem.4)
)
raw_data_long <- raw_data %>%
  pivot_longer(cols = c(Gem, Moon, Moon.F), names_to = "LLM", values_to = "response")

# Aggregate data
m_dat <- aggregate(response ~ Vp * LLM, data = raw_data_long, FUN = mean)
m_dat$Vp <- factor(m_dat$Vp)
m_dat$LLM <- factor(m_dat$LLM)

# ANOVA
anova_main <- ezANOVA(data = m_dat, 
                      dv = response, 
                      within = .(LLM),
                      wid = Vp,
                      detailed = TRUE) 
anova_out(anova_main)
# LLM F(2, 10) =  28.54, p < .001, np2 = .85 (***)

# t tests
t1 = t.test(m_dat$response[m_dat$LLM == "Gem"], m_dat$response[m_dat$LLM == "Moon"], paired=TRUE)
t2 = t.test(m_dat$response[m_dat$LLM == "Gem"], m_dat$response[m_dat$LLM == "Moon.F"], paired=TRUE)
t3 = t.test(m_dat$response[m_dat$LLM == "Moon.F"], m_dat$response[m_dat$LLM == "Moon"], paired=TRUE)
t_out(toutput = t1)
t_out(toutput = t2)
t_out(toutput = t3)
p_korr <- p.adjust(c(t1$p.value, t2$p.value, t3$p.value), method = 'holm')
p_korr
# t1: t(5) = 5.23, p_holm = .007, d = 2.14 (**)
# t2: t(5) = 6.28, p_holm = .005, d = 2.56 (**)
# t3: t(5) = 1.74, p_holm = .143, d = 0.71

# Plot
summary_data <- summarySEwithin(m_dat, measurevar = "response", withinvars = "LLM", idvar = "Vp")
colors <- brewer.pal(8, "Greens")[c(2,5,8)]
ggplot(summary_data, aes(x = LLM, y = response, fill = LLM)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black") +
  geom_errorbar(aes(ymin = response - se, ymax = response + se), 
                width = 0.2, position = position_dodge(0.9)) +
  labs(title = "How suitable is this image as a YouTube thumbnail?", 
       y = "Mean ranking (0 = not at all, 10 = totally)", x = "Large Language Models") +
  theme_minimal() +
  scale_fill_manual(values = colors) +
  scale_y_continuous(limits = c(0, 10))

# --------
# SURVEY 2
# --------

# Load data
dat_sur2 <- data.frame(read_excel("SecondSurvey.xlsx"))

# Delet participant who did not watch the video
dat_sur2 <- dat_sur2[dat_sur2$video == 1,]

# Raw dataframe
raw_data_2 <- data.frame(
  "Vp" = dat_sur2$VP,
  "using" = dat_sur2$using,
  "intuitive" = dat_sur2$intuitive,
  "thumbnail" = dat_sur2$as.a.thumbnail,
  "quality" = dat_sur2$image.quality,
  "recomment" = dat_sur2$Recomment
)
raw_data_long_2 <- raw_data_2 %>%
  pivot_longer(cols = c(using, intuitive, thumbnail, quality), names_to = "questions", values_to = "response")

# Aggregate data
m_dat_2 <- aggregate(response ~ Vp * questions, data = raw_data_long_2, FUN = mean)
m_dat_2$Vp <- factor(m_dat_2$Vp)
m_dat_2$questions <- factor(m_dat_2$questions)

# t tests
t4 = t.test(m_dat_2$response[m_dat_2$questions == "intuitive"], mu = 5, alternative = "greater")
t5 = t.test(m_dat_2$response[m_dat_2$questions == "quality"], mu = 5, alternative = "greater")
t6 = t.test(m_dat_2$response[m_dat_2$questions == "thumbnail"], mu = 5, alternative = "greater")
t7 = t.test(m_dat_2$response[m_dat_2$questions == "using"], mu = 5, alternative = "greater")
t_out(toutput = t4)
t_out(toutput = t5)
t_out(toutput = t6)
t_out(toutput = t7)
# t4: t(6) = 1.58, p = .083, d = 0.60
# t5: t(6) = 9.72, p < .001, d = 3.67 (***)
# t6: t(6) = 2.20, p = .035, d = 0.83 (*)
# t7: t(6) = 2.77, p = .016, d = 1.05 (*)

# Plot
summary_data_2 <- summarySEwithin(m_dat_2, measurevar = "response", withinvars = "questions", idvar = "Vp")
summary_data_2 <- summary_data_2 %>%
  slice(match(c("intuitive", "using", "quality", "thumbnail"), questions)) %>%
  mutate(questions = factor(questions, levels = c("intuitive", "using", "quality", "thumbnail")))
colors_2 <- brewer.pal(8, "Blues")[c(2,4,6,8)]
ggplot(summary_data_2, aes(x = questions, y = response, fill = questions)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black") +
  geom_errorbar(aes(ymin = response - se, ymax = response + se), 
                width = 0.2, position = position_dodge(0.9)) +
  labs(title = "Survey 2", 
       y = "Mean ranking (0 = not at all, 10 = totally)", x = "Questions") +
  theme_minimal() +
  scale_fill_manual(values = colors_2) +
  scale_y_continuous(limits = c(0, 10))