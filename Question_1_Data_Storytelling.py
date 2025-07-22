#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncovering Stories from Singapore’s HDB Resale Market


# In[ ]:


# Over the past decade, Singapore’s HDB resale market has undergone notable changes — from cooling measures and COVID-19 disruptions to the emergence of million-dollar flats. But while much attention is paid to price trends and housing supply, the story of the agent is often overlooked.

# Using open datasets from CEA (agent transactions) and HDB (resale flat prices), we explored:

# Where agents are most active

# What types of flats they focus on

# How activity and pricing evolved through the pandemic

# Let’s explore what the data reveals.


# In[ ]:


import requests
import time
import pandas as pd
from io import StringIO
import plotly.graph_objects as go
import warnings
warnings.filterwarnings(action="ignore")


CEA_DATASET_ID = "d_ee7e46d3c57f7865790704632b0aef71"
RESALE_DATASET_ID='d_8b84c4ee58e3cfc0ece0d773c8ca6abc'
max_tries=3

def load_dataset_as_dataframe(DATASET_ID):
    INITIATE_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download"
    POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"
    init_resp = requests.get(INITIATE_URL)
    init_resp.raise_for_status()

    for _ in range(max_tries):
        time.sleep(2)
        poll_resp = requests.get(POLL_URL)
        poll_resp.raise_for_status()
        download_url = poll_resp.json().get("data", {}).get("url")
        if download_url:
            break
    else:
        raise TimeoutError("Timed out waiting for dataset download URL.")

    csv_resp = requests.get(download_url)
    csv_resp.raise_for_status()
    df = pd.read_csv(StringIO(csv_resp.text))
    return df

cea_df = load_dataset_as_dataframe(CEA_DATASET_ID)
resale_df = load_dataset_as_dataframe(RESALE_DATASET_ID)


# In[ ]:


cea_df['transaction_type'].value_counts()


# In[ ]:


cea_df = cea_df[cea_df['transaction_type'] == 'RESALE']


# In[ ]:


cea_df['month'] = pd.to_datetime(cea_df['transaction_date'], format='%b-%Y')


# In[ ]:


cea_df


# In[ ]:


cea_df.describe()


# In[ ]:


resale_df


# In[ ]:


cea_df['transaction_date'] = pd.to_datetime(cea_df['transaction_date'], format="%b-%Y")
cea_df['year'] = cea_df['transaction_date'].dt.year


# In[ ]:


# The Resilient Agent: Post-COVID Recovery
# Even as lockdowns froze much of the economy in 2020, resale activity recovered strongly in 2021 and beyond.
# Agent transactions — which dipped due to viewing restrictions and uncertainty — surged back as buyers returned and prices began to climb.



# In[ ]:


cea_resale = cea_df[cea_df['transaction_type'] == 'RESALE']

agent_counts = cea_resale.groupby('year').size().reset_index(name='agent_transactions')

resale_avg = resale_df.groupby('year')['resale_price'].mean().reset_index(name='avg_resale_price')

merged = pd.merge(agent_counts, resale_avg, on='year')

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Bar(x=merged['year'], y=merged['agent_transactions'], name="Agent Transactions", marker_color='skyblue'))
fig.add_trace(go.Scatter(x=merged['year'], y=merged['avg_resale_price'], name="Avg Resale Price", mode='lines+markers', yaxis='y2'))
fig.update_layout(
    title=" Agent Recovery in the Resale Market (Post-COVID)",
    xaxis_title="Year",
    yaxis=dict(title="Agent Transactions"),
    yaxis2=dict(title="Average Resale Price (SGD)", overlaying='y', side='right'),
    template="plotly_white", height=500
)
fig.show()


# In[ ]:


#  Hidden Battlegrounds: Agent Activity vs Affordability
# While towns like Bishan and Queenstown dominate headlines with million-dollar flats, we found surprising hotspots of agent activity in more affordable towns
# like Yishun and Woodlands & Sengkang.

# These areas show:

# High agent transaction volume
# Moderate resale prices
# High flat turnover


# In[ ]:


# Agent activity by town
town_agents = cea_resale.groupby('town').size().reset_index(name='agent_deals')

# Average resale price by town
town_prices = resale_df.groupby('town')['resale_price'].mean().reset_index(name='avg_price')

# Volume by town
town_volume = resale_df.groupby('town').size().reset_index(name='flats_sold')

# Merge
merged_town = town_agents.merge(town_prices, on='town').merge(town_volume, on='town')

# Plot
import plotly.express as px
fig = px.scatter(
    merged_town, x='avg_price', y='agent_deals', size='flats_sold', color='town',
    title="Agent Activity vs Affordability",
    labels={'avg_price': 'Avg Resale Price (SGD)', 'agent_deals': 'Agent Transactions'},
    size_max=60
)
fig.show()


# In[ ]:


# The Flat Type Focus: What’s Keeping Agents Busy?
# Agents tend to specialize based on demand patterns. The 4-room flat emerges as the clear leader in transaction volume — striking a balance between size and affordability for families.

# Meanwhile, 5-room and Executive flats, while pricier, attract fewer deals — likely due to a smaller buyer pool or lower availability.


# In[ ]:


flat_group = resale_df.groupby('flat_type').size().reset_index(name='resale_transactions')
flat_price = resale_df.groupby('flat_type')['resale_price'].mean().reset_index(name='avg_price')
flat_merged = pd.merge(flat_group, flat_price, on='flat_type')

import plotly.express as px
fig = px.bar(flat_merged.sort_values(by='resale_transactions', ascending=False),
    x='flat_type', y='resale_transactions', color='avg_price',
    title="Flat Types with Most Resale Transactions",
    labels={'resale_transactions': 'Resale Transactions', 'avg_price': 'Avg Resale Price'},
    color_continuous_scale='Blues'
)
fig.show()

