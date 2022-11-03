#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[15]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re
from scipy import stats
from sklearn.linear_model import LinearRegression


# #  LinearRegression

# In[42]:


house=pd.read_csv('UK historical house prices and salary/Average_UK_houseprices_and_salary.csv')
house.drop('Unnamed: 3', axis=1, inplace=True)
house.columns = [x.rsplit(' ', 4)[0] for x in house.columns]
house.columns = [re.sub(r'\s+', '_', x).lower() for x in house.columns]
house.dropna(inplace=True)
house


# In[43]:


# Figure size
plt.figure(figsize=(14,6))

plt.scatter(house.median_salary, house.average_house_price)

plt.title('Average House Prices in UK')
plt.ylabel('Average price adjusted by inflation (£)')
plt.xlabel('Median salary adjusted by inflation (£)')


# In[44]:


y = house[['average_house_price']]
x = house[['median_salary']]
lr = LinearRegression()
lr.fit(x,y)


# In[45]:


y_pred = lr.predict(x)


# In[50]:


plt.scatter(house.median_salary, house.average_house_price,label='train_data')
plt.title('LinearRegression for Average price and Median salary')
plt.ylabel('Average price adjusted by inflation (£)')
plt.xlabel('Median salary adjusted by inflation (£)')
plt.plot(x.median_salary,y_pred, color='red')
plt.show()


# #  Median salary 

# In[51]:


# Figure size
plt.figure(figsize=(14,6))

sns.regplot(x='Year', y='Median Salary adj. by inflation (pounds)', data=data_df, 
            marker='x', scatter_kws={"s": 60, 'color':'green'}, line_kws={"color": "red"}, ci=False)

plt.title('Median yearly earnings in UK (adjusted by inflation) \n')
plt.ylabel('Median salary adjusted by inflation (£)')
plt.xlabel('Year(1980-2022)')


# #  Average house prices

# In[53]:


# Figure size
plt.figure(figsize=(14,6))

sns.regplot(x='Year', y='Average house price adj. by inflation (pounds)', data=data_df, 
            marker='x', scatter_kws={"s": 70, 'color':'green'}, line_kws={"color": "red"}, ci=False)

plt.title('Average House Prices in UK (adjusted by inflation) \n')
plt.ylabel('Average price adjusted by inflation (£)')
plt.xlabel('Year(1980-2022)')


# In[ ]:




