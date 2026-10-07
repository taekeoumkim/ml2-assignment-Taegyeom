
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

st.title('Bike Sharing 시계열 예측')
st.write('월별 자전거 대여량을 확인하고 미래 값을 예측합니다.')
uploaded_file = st.file_uploader('Bike_Sharing_Demand.csv 업로드', type=['csv'])
if uploaded_file is not None:
    try:
        bike = pd.read_csv(uploaded_file)
        if not {'datetime', 'count'}.issubset(bike.columns):
            st.error('datetime과 count 컬럼이 필요합니다.')
            st.stop()
        st.write('원본 데이터')
        st.dataframe(bike)
        bike['datetime'] = pd.to_datetime(bike['datetime'], errors='raise')
        bike['count'] = pd.to_numeric(bike['count'], errors='raise')
        if bike[['datetime', 'count']].isna().any().any():
            st.error('datetime과 count의 결측치를 확인하세요.')
            st.stop()
        bike = bike.sort_values('datetime').reset_index(drop=True)
        bike['month'] = bike['datetime'].dt.to_period('M').dt.to_timestamp()
        monthly = bike.groupby('month')[['count']].sum().asfreq('MS')
        if len(monthly) < 6 or monthly['count'].isna().any():
            st.error('최소 6개월의 연속 월별 관측 데이터가 필요합니다. 누락 월을 확인하세요.')
            st.stop()
        st.write('월별 관측 대여량 합계')
        st.dataframe(monthly)
        st.line_chart(monthly['count'])
        st.write('이 CSV의 합계는 관측된 날짜 기준입니다. 매월 일부 날짜만 있으면 전체 월의 총수요가 아닙니다.')
        forecast_period = st.selectbox('예측 기간(개월)', [1, 3, 6])
        model_fit = ARIMA(monthly['count'], order=(2, 1, 1)).fit()
        forecast = model_fit.forecast(steps=forecast_period)
        future_dates = pd.date_range(start=monthly.index[-1] + pd.offsets.MonthBegin(1),
                                     periods=forecast_period, freq='MS')
        forecast_df = pd.DataFrame({'predicted': forecast.to_numpy()}, index=future_dates)
        forecast_df.index.name = 'month'
        st.write('미래 예측값')
        st.dataframe(forecast_df)
        chart_df = pd.concat([monthly.rename(columns={'count': 'actual'}), forecast_df], axis=1)
        st.line_chart(chart_df)
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f'CSV 내용을 확인하세요: {error}')
