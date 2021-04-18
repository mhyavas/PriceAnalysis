import finnhub
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import config
api=config.api
finnhub_client=finnhub.Client(api_key=api)

time=int(time.time())
res = finnhub_client.forex_candles('OANDA:EUR_USD',60,time-94694400, time)

tablo=pd.DataFrame().from_dict(res)
features=['c','t']
closing=tablo[features]
"""
plt.style.use('ggplot')
closing.plot(y='c',x='t',color='b',figsize=(10,5))
plt.show()

"""
closing['s_10']=closing['c'].shift(1).rolling(window=10).mean()
closing['corr']=closing['c'].rolling(window=10).corr(closing['s_10'])
closing=closing.dropna()
X=closing[['s_10','corr']]
#print(X.head())

y=closing['c']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3,random_state=42)
linear=LinearRegression().fit(train_X,train_y)

predict_price=linear.predict(test_X)
predict_price=pd.DataFrame(predict_price,index=test_y.index)



predict_price.plot()
test_y.plot()
plt.legend(['predicted_price','actual_price'])
plt.ylabel('price')
plt.show()

