import pandas as pd
score = [[34,67,87],[68,98,58],[75,73,86],[94,59,81]]
name = ['小明','小红','小李']
course = ['语文','数学','英语','政治']
mydata1 = pd.DataFrame(data=score,columns=name,index=course)#指定行名（index）和列名（columns）
# print(mydata1)
mydata2 = pd.DataFrame(score)#不指定行列名，默认使用0,1,2……
# print(mydata2)

print(mydata1.loc['语文',:])