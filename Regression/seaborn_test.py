from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
url = "https://raw.githubusercontent.com/jonasbauer192/CODECADEMY-Capstone-Project-Netflix-Data/main/NFLX.csv"
netflix_stocks = pd.read_csv(url)

# rename columns
netflix_stocks.rename(columns = {'Adj Close': 'Price'}, inplace = True)

print(netflix_stocks.info())

# visualizing the netflix quarterly data
ax = plt.subplot()
sns.violinplot(data = netflix_stocks, x = "Date", y = "Price")
plt.xlabel("Business Quarters in 2017")
plt.ylabel("Closing Stock Price")

plt.savefig("Distribution of 2017 Netflix Stock Prices by Quarter.png")
plt.show()

plot_loss(history)

