import pandas as pd
from datetime import datetime
import time

df = pd.read_csv("ping_exp.csv")
print(df.info())
df['Time']=pd.to_datetime(df['Time'],unit='s')
print(df)
df.to_csv("nuovo_ping.csv")

