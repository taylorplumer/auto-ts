from helpers import *

class Dashboard():
    def __init__(self, **kwargs):
        self.df = kwargs.get('df')
        self.image = kwargs.get('image')
        self.title = kwargs.get('title')
        self.m = kwargs.get('m')
        self.tabs = ["Time Series Decomposition", "Model Selection", "Model Evaluation"]

    def preprocess(self):
        self.train, self.test = self.df.iloc[:100].copy(), self.df[100:].copy()
        self.additive_decomposition = seasonal_decompose(self.df, model='additive')
        self.multiplicative_decomposition = seasonal_decompose(self.df, model='multiplicative')
        self.adf_test = pm.arima.ADFTest(alpha=0.05)

        self.Arima_models = run_auto_arima(self.train, self.m)
        self.grid_search_df = pd.DataFrame([model.to_dict() for model in self.Arima_models])[
            ['order', 'seasonal_order', 'aic', 'aicc', 'bic']]
        self.best_result_df = best_result_df(self.grid_search_df)
        self.test['preds'] = make_predictions(self.Arima_models, self.best_result_df.index[0], 44)
        self.test['error'] = calc_errors(self.test)

    def save_output(self, filepath):
        pass


    def main(self):
        if self.image != None:
            st.sidebar.image(self.image, width=200)
        tab = st.sidebar.radio("", self.tabs)
        if tab =="Time Series Decomposition":
            self.preprocess()
            st.write('ADF Test Result: {}'.format(self.adf_test.should_diff(self.df)))
            decomposition_selection = st.radio(
                "Select Decomposition Model",
                ('Additive', 'Multiplicative')
            )
            if decomposition_selection == 'Additive':
                display_plots(self.df, self.additive_decomposition)
            else:
                display_plots(self.df, self.multiplicative_decomposition)

        elif tab == "Model Selection":
            st.write("Model Selection")
            st.write("Auto-Arima Grid Search Results")
            st.dataframe(self.grid_search_df, width=800)
            st.write("Best Fit Model")
            st.dataframe(self.best_result_df)

        elif tab == "Model Evaluation":
            st.plotly_chart(train_test_predict_plot(self.train, self.test, self.title), use_container_width=True)
            st.dataframe(metrics_table(self.test))
            st.plotly_chart(prediction_error_plot(self.test), use_container_width=True)
            st.plotly_chart(probability_plot(self.test), use_container_width=True)
