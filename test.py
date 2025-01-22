import pandas as pd   

df=pd.read_csv("data/titanic.csv")

from helper_modules.agent import query

output=query("what is the data",df)
for c in output:
    print(c)