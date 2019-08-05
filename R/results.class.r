require('ggplot2')
require('reshape2')
setwd('/Users/erfan/Main/oasis/test_augmentation/results/')
d1<-read.csv('data/final_results/results.class.csv',sep=' ', header=F)
d<-read.csv('data/results.Jan.22.txt',sep=' ', header=F)

d<-rbind(d,d1)
d<-droplevels(d[d$V3%in%c("IBD", "bmi"),])

levels(d$V9)<-list( "RF"="rf","NN"="nn")
levels(d$V3)<-list("IBD"="IBD","BMI category"="bmi")
d<-droplevels(d[d$V2%in%c("non","phylogeny","class","class-4"),])
d<-droplevels(d[!d$V1=="beta",])
d<-droplevels(d[d$V4 %in% c("100-0.5-1", "100-0.5-0", "non-non-non"),])
d<-droplevels(d[!d$V4 %in% c(100-0.5-0.2),])
d<-droplevels(d[!d$V1 %in% c("beta"),])
d<-droplevels(d[!(d$V1 %in% c("binomial") & d$V2 %in% c("from_data")),])
d<-droplevels(d[!(d$V1 %in% c("beta_binomial") & d$V2 %in% c("from_data")),])
dw<-reshape(d, idvar = c("V1","V2","V3","V4","V5","V6","V7","V9","V10"), timevar = "V8", direction = "wide")
dw$diff<-dw$V11.augmented - dw$V11.original


# ggplot(data=d[d$V8 == 5 & d$V1 == "only_beta" & d$V10 =="AUC",  ],
#        aes(x=reorder(model,V15),y=V15,fill=V12))+geom_boxplot()+facet_wrap(V4~.,scales="free_y")+
#   theme_bw()+theme(axis.text.x = element_text(angle=45,hjust=1))

d1<-droplevels(d[d$V8=="original"  & d$V5==5 & d$V1 == "beta-binomial" & d$V4== "100-0.5-0" & d$V2 == "phylogeny",])

d3<-droplevels(d1)
levels(d3$V1)[1]<-"no-augmentation"
levels(d3$V2)[1]<-"non"
levels(d3$V4)[1]<-"non-non-non"
levels(d3$V5)[1]<-"non"
d<-droplevels(d[d$V8 != "original",])
d<-rbind(d,d3)
dw$method<-as.factor(paste(dw$V1,dw$V2,dw$V4,sep="-"))
d$method<-as.factor(paste(d$V1,d$V2,d$V4,sep="-"))


levels(dw$method)<-list("no augmentation"="no-augmentation-non-non-non-non",
                        "Resampling"="resampling-non-non-non-non",
                        "TADA-SV"="binomial-phylogeny-non-non-non",
                        "TADA-TVSV-1"="beta-binomial-clustering-1-phylogeny-100-0.5-1",
                        "TADA-TVSV-4"="beta-binomial-clustering-4-phylogeny-100-0.5-1",
                        "TADA-TVSV-8"="beta-binomial-clustering-8-phylogeny-100-0.5-1",
                        "TADA-TVSV-40"="beta-binomial-clustering-40-phylogeny-100-0.5-1",
                        "TADA-TVSV-m"="beta-binomial-phylogeny-100-0.5-0",
                        "TADA-TVSV*-1"="beta-binomial-class-non-non-non",
                        "TADA-TVSV*-4"="beta-binomial-class-4-non-non-non")

levels(d$method)<-list("no augmentation"="no-augmentation-non-non-non-non",
                        "Resampling"="resampling-non-non-non-non",
                        "TADA-SV"="binomial-phylogeny-non-non-non",
                        "TADA-TVSV-1"="beta-binomial-clustering-1-phylogeny-100-0.5-1",
                        "TADA-TVSV-4"="beta-binomial-clustering-4-phylogeny-100-0.5-1",
                        "TADA-TVSV-8"="beta-binomial-clustering-8-phylogeny-100-0.5-1",
                        "TADA-TVSV-40"="beta-binomial-clustering-40-phylogeny-100-0.5-1",
                        "TADA-TVSV-m"="beta-binomial-phylogeny-100-0.5-0",
                        "TADA-TVSV*-1"="beta-binomial-class-non-non-non",
                        "TADA-TVSV*-4"="beta-binomial-class-4-non-non-non")

d$V5<-as.factor(d$V5)

dw$V5<-as.factor(dw$V5)

d$col<-"no augmentation"
d[d$method != "no augmentation" & d$V5 == "5","col"]<-"5x"
d[d$method != "no augmentation" & d$V5 == "20", "col"]<-"20x"
d[d$method != "no augmentation" & d$V5 == "50", "col"]<-"50x"

d$col<-as.factor(d$col)

dw$col<-"no augmentation"
dw[grepl("TVSV",dw$method),15]<-"TVSV"
dw[dw$method=="SV", 15]<-"SV"
dw$col<-as.factor(dw$col)
d[d$method=="no augmentation", "V5"]<-"50"
d$col2<-"no augmentation"
d[d$method %in% c("TADA-TVSV*-1","TADA-TVSV*-4"),"col2"]<-"from data"
d[d$method %in% c("TADA-TVSV-m"),"col2"]<-"from phylogeny"
d$col2<-as.factor(d$col2)

ggplot(data=d[d$V7=="AUC" & (d$V5 == "50" | d$V5 == "non") & d$method %in% c("no augmentation", "TADA-TVSV-m", "TADA-TVSV*-1","TADA-TVSV*-4"), ],
       aes(x=method,y=V11,colour=col2))+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  stat_summary(position=position_dodge(0.5))+facet_grid(V3~V9,scales = "free_y")+
  scale_color_brewer(name="",palette = "Dark2")
ggsave('figures/compare-class-methods-AUC-final.pdf', width= 5, height=6.3)


d<-droplevels(d[!((d$method == "SV" | d$method == "TVSV-m") & d$V6 == "0"), ])

ggplot(data=d[(d$V5 == "5" | d$V5 == "non") & d$V7=="AUC" & !d$method == "Resampling" , ],
       aes(x=method,y=V11,colour=col))+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  stat_summary()+facet_grid(V3~V9,scales = "free_y")+
  scale_color_brewer(name="",palette = "Dark2")
ggsave('figures/compare-methods-AUC-final.pdf', width= 5, height=6.3)




d$col2<-"no augmentation"
d[d$V5=="5" & d$method != "no augmentation","col2"]<-"5"
d[d$V5=="20" & d$method != "no augmentation","col2"]<-"20"
d[d$V5=="50" & d$method != "no augmentation","col2"]<-"50"

d$col2<-as.factor(d$col2)
d$col2<-factor(d$col2,levels=c("no augmentation","5","20","50"))
ggplot(data=d[d$V7=="AUC" & !d$method == "Resampling" & d$method %in% c("no augmentation", "SV","TVSV-m"), ],
       aes(x=method,y=V11,color=col2))+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  stat_summary(position=position_dodge(width=0.4))+facet_grid(V3~V9,scales = "free_y")+
  scale_color_brewer(name="",palette = "Dark2")
ggsave('figures/compare-methods-supplementary-AUC-final.pdf', width= 5, height=6.3)

### p-value for IBD RF
t.test(dw[dw$method =="TVSV-m" & dw$V7=="AUC" & !dw$method == "Resampling" & dw$V9 == "RF" & dw$V5 %in% c("5") & dw$V3== "IBD",]$diff)
nrow(d[d$method =="TVSV-m" & d$V7=="AUC" & !d$method == "Resampling" & d$V9 == "RF" & d$V5 %in% c("5") & d$V3== "IBD",])

mean(d[d$method =="TVSV-m" & d$V7=="AUC"& d$V9 == "RF" & d$V5 %in% c("5") & d$V3== "BMI category",]$V11)
nrow(d[d$method =="TVSV-m" & d$V7=="AUC"& d$V9 == "RF" & d$V5 %in% c("5") & d$V3== "BMI category",])


dcast(dw[dw$V7=="AUC" & dw$V5== "5" & dw$method !="Resampling",],method+V9+V5+V3~.,value.var="diff",fun.aggregate = mean)

t.test(dw[dw$method =="TVSV-m" & dw$V7=="AUC"& dw$V9 == "RF" & dw$V5 %in% c("5") & dw$V3== "BMI category",]$diff)
dcast(dw[(dw$V5 == "5" ) & dw$V7=="AUC" & dw$method == "TVSV-m" , ],method+col+V3+V9~.,value.var="diff",fun.aggregate = mean)
dcast(d[d$V5 == "5"  & d$V7=="AUC" & d$method == "TVSV-m" , ],method+col+V3+V9~.,value.var="V11",fun.aggregate = mean)
