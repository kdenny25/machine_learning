par(mfrow=c(1,4))
  for(i in 1:4){
    boxplot(x[,i], main=names(iris)[i])
  }