require('ggplot2')
require('reshape2')
setwd('/Users/erfan/Main/oasis/test_augmentation/results/')

d<-read.csv('data/final_results/E2-results.bmi.csv',sep=' ', header=F)
d$V12<-as.factor(d$V12)
d$V13<-as.factor(d$V13)

levels(d$V12)<-list("10-vs-90"="10.9",
                    "20-vs-80"="20.8",
                    "33-vs-66"="33.66")

levels(d$V13)<-list("10-vs-90"="10.9",
                    "20-vs-80"="20.8",
                    "33-vs-66"="33.66",
                    "50-vs-50"="50.5",
                    "80-vs-20"="80.2")


levels(d$V10)<-list("RF"="rf","NN"="nn" )

d1<-d[d$V1=="beta-binomial" & d$V5=="50" & d$V9=="original" & d$V2 == "phylogeny",]

d3<-droplevels(d1)
levels(d3$V1)[1]<-"no-augmentation"
d3$V2<-"non"
d3$V4<-"non"
d3$V5<-"non"
d<-droplevels(d[d$V9 != "original",])
d<-rbind(d,d3)

d[d$V1=="downsampling","V5"] = 0
d$method<-as.factor(paste(d$V1,d$V2,d$V4,sep="-"))


d<-droplevels(d[!d$method %in% c("SMOTE-non-SMOTE.40","SMOTE-non-ADASYN.40"),])
levels(d$method)<-list("no augmentation"="no-augmentation-non-non",
                       "downsampling"="downsampling-non-non-non-non",
                       "ADASYN"="SMOTE-non-ADASYN.5",
                       "SMOTE"="SMOTE-non-SMOTE.5",
                       "TADA-TVSV-m"="beta-binomial-phylogeny-100-0.5-0",
                       "TADA-SV"="binomial-phylogeny-non-non-non"
                      )

# d<-droplevels(d[d$method != "Ind-Downsampling",])
# dw<-droplevels(dw[dw$method != "Ind-Downsampling",])

d$col<-"No Augmentation"
d[d$method == "Downsampling" | d$V5=="0" ,"col"]<-"Balance"
d[d$V5 %in% c(50,20),"col"]<-"Balance++"
d$col<-as.factor(d$col)

ggplot(d[d$V8=="AUC" &  d$V5 != 20 &   d$V13 %in% c("10-vs-90",  "20-vs-80",  "33-vs-66"),], 
       aes(x=method,y=V14, color = method, shape=col))+stat_summary(alpha=0.8)+
  facet_grid(V10~V12,scales="free")+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),text=element_text(size=14),
        legend.position = "bottom", legend.box = 'vertical')+
scale_color_manual(name="",values = c("#d95f02","#1b9e77","#7570b3","#7570b3","#e7298a","#e7298a"))+
  scale_shape_discrete(labels=c("Balance","Balance++","No Augmentation"))
ggsave('figures/balance-unbiased-auc-bmi.pdf', 6.6, height=6.6)



ggplot(d[d$V8=="accuracy" &  d$V10=="RF" & d$V5 != 20 &   d$V13 %in% c("10-vs-90",  "20-vs-80",  "33-vs-66"),], 
       aes(x=method,y=V14, color=col))+stat_summary(alpha=0.8)+
  facet_grid(V10~V12,scales="free")+
  theme_bw() +xlab("Augmentation method")+ylab("Accuracy")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  scale_color_brewer(name="",palette = "Dark2")+
  scale_x_discrete(labels=c("no augmentation","downsampling","TADA-SV","TADA-TVSV-m"))
ggsave('figures/balance-unbiased-acc-bmi.pdf', width= 6.6, height=3.8)




ggplot(d[d$V8=="AUC"& d$V14 == "original" & 
           d$V5 != 20 & 
           d$method %in% c("no augmentation","Ind-Downsampling","Phy-BetaBinomial-Samp","Phy-Binomial-Samp") &
           d$V13 %in% c("10-vs-90", 
                        "20-vs-80", 
                        "33-vs-66"),], 
       aes(x=method,y=V15, color=V5))+stat_summary(alpha=0.8)+
  facet_grid(V10~V12,scales="free")+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  scale_x_discrete(labels=c("no augmentation","downsampling","TADA-TVSV-m","TADA-SV"))+
  scale_color_brewer(name="",palette = "Dark2",labels=c("Balance","Balance++","No augmentation"))
ggsave('figures/balance-unbiased.pdf', width= 6.6, height=5.6)


ggplot(d[d$V8=="accuracy"& d$V14 == "original" & 
           d$V5 != 20 & d$V10 == "RF" &
           d$method %in% c("no augmentation","Phy-BetaBinomial-Samp", "Phy-Binomial-Samp") &
           d$V13 %in% c("10-vs-90", 
                        "20-vs-80", 
                        "33-vs-66"),], 
       aes(x=method,y=V15, color=V5))+stat_summary(alpha=0.8,size=0.3)+
  facet_wrap(V12~.)+
  theme_bw() +xlab("")+ylab("Acc.")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  scale_x_discrete(labels=c("no augmentation","TADA-TVSV-m","TADA-SV"))+
  scale_color_brewer(name="",palette = "Dark2",labels=c("Balance","Balance++","No augmentation"))
ggsave('figures/balance-unbiased-acc.pdf', width= 6.6, height=3.8)



ggplot(d[d$V8=="AUC"& d$V14 == "original" & 
           d$V5 != 20 & d$method %in% c("no augmentation","Ind-Downsampling","Phy-BetaBinomial-Samp","Phy-Binomial-Samp")& 
           d$V13 %in% c("80-vs-20", 
                        "20-vs-80", 
                        "50-vs-50"),],
       aes(x=method,y=V15, color=V5))+stat_summary(alpha=0.8)+
  facet_grid(V10~V13,scales="free")+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  scale_x_discrete(labels=c("no augmentation","downsampling","TADA-TVSV-m","TADA-SV"))+
  scale_color_brewer(name="",palette = "Dark2",labels=c("Balance","Balance++","No augmentation"))
ggsave('figures/balance-biased.pdf', width= 6.6, height=5.6)



