import pandas as pd
import matplotlib.pyplot as plt

# Load the three CSV files
csv_files = ["input_uni.csv", "first_notequal.csv"]
df = [pd.read_csv(file) for file in csv_files]

#DO NOT TOUCH 21
#21 = RMSE OR MSE
#24 = THEIL
#31 = NCD 

plt.figure(figsize=(10, 6))
df[0] = df[0].iloc[1000:1140]
df[1] = df[1].iloc[1000:1140]


plt.plot(df[1]['pitch'], label=f'Input')
plt.plot(df[0]['pitch'], label=f'ARIMA')


plt.xlabel('Index')
plt.ylabel('Pitch')
plt.title('Pitch values of \'Ave Maria\' by Bach/')
plt.legend()
plt.show()