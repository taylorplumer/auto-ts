# auto-ts
Build a Streamlit web app diagnostic tool for ARIMA time series analysis

![](https://github.com/taylorplumer/auto-ts/blob/master/assets/auto-ts-logo.png)
### Summary
The repository contains working code for preparing time series data, building an ARIMA model, and deploying to a Streamlit app locally. Instructions are below.

A demo deployed to Heroku is available for viewing at the following address (it may take a few seconds to load for the Heroku dyno to spin up): http://autots-dash.herokuapp.com/

The demo data is the often used AirPassengers.csv dataset; it contains monthly passenger totals from 1949 to 1960.

### Instructions:
1. Run the following command to run the web app.
    `streamlit run app.py`

3. Go to http://127.0.0.1:8501/


###  Installation
This project utilizes default packages within the Anaconda distribution of Python. Streamlit and pmdarima were additionally installed.

After creating a virtual environment (recommended), you can install the dependencies with the following command:

```
pip install -r requirements.txt
```
