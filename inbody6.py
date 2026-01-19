import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Inbody Data", layout="wide")
st.title("Inbody Data")

# 複数CSVアップローダ
uploaded_files = st.file_uploader("CSVファイルをアップロードしてください（複数選択可）", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    try:
        dfs = []
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            df = df[["1. ID", "5. Test Date / Time", "6. Weight", "24. FFM (Fat Free Mass)"]].copy()
            df.columns = ["ID", "DateTime", "Weight", "FFM"]
            df["Date"] = pd.to_datetime(df["DateTime"].str.split().str[0], format="%Y.%m.%d")
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # ユニークなIDのリスト
        unique_ids = df["ID"].unique()
        selected_id = st.selectbox("選手IDを選択", unique_ids)

        # 日付範囲の選択
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        date_range = st.date_input("表示する期間を選択", [min_date, max_date], min_value=min_date, max_value=max_date)

        # 日付型を統一
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        # データフィルタリング
        df_filtered = df[(df["ID"] == selected_id) & (df["Date"] >= start_date) & (df["Date"] <= end_date)]
        df_filtered = df_filtered.sort_values("Date")

        # 移動平均線・回帰直線の表示オプション
        show_moving_avg = st.checkbox("移動平均線を表示", value=True)
        show_regression = st.checkbox("回帰直線を表示", value=False)

        # ツールチップ用の割合計算
        df_filtered["FFM_Ratio"] = df_filtered["FFM"] / df_filtered["Weight"] * 100

        # 共通のグラフ設定
        layout_common = dict(
            xaxis_title="日付",
            yaxis_title="kg",
            margin=dict(l=30, r=30, t=30, b=30),
            height=350,
            hovermode="x unified",
            font=dict(size=14, color="black"),
            legend=dict(font=dict(size=14)),
            hoverlabel=dict(font=dict(size=14, color="white"), bgcolor="black")
        )

        # --- 除脂肪体重グラフ ---
        ffm_fig = go.Figure()
        ffm_fig.add_trace(go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["FFM"],
            mode="lines+markers",
            name="除脂肪体重 (FFM)",
            marker=dict(size=10, color='blue'),
            line=dict(width=3, color='blue'),
            hovertemplate="日付: %{x|%Y年%m月%d日}<br>除脂肪体重: %{y:.1f} kg<br>割合: %{customdata:.1f} %",
            customdata=np.round(df_filtered["FFM_Ratio"].values.reshape(-1, 1), 1)
        ))
        if show_moving_avg:
            ffm_fig.add_trace(go.Scatter(
                x=df_filtered["Date"],
                y=df_filtered["FFM"].rolling(window=3, min_periods=1).mean(),
                mode="lines",
                name="FFM移動平均",
                line=dict(dash="dash", width=2, color='darkblue')
            ))
        if show_regression and len(df_filtered) >= 2:
            x_num = df_filtered["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            model = LinearRegression().fit(x_num, df_filtered["FFM"])
            y_pred = model.predict(x_num)
            ffm_fig.add_trace(go.Scatter(
                x=df_filtered["Date"],
                y=y_pred,
                mode="lines",
                name="FFM回帰直線",
                line=dict(dash="dot", width=2, color='gray')
            ))
        if (end_date - start_date).days > 60:
            ffm_fig.update_xaxes(dtick="M1", tickformat="%Y年%m月")
        else:
            ffm_fig.update_xaxes(tickformat="%Y年%m月%d日")
        ffm_fig.update_layout(title="除脂肪体重の推移", **layout_common)

        # --- 体重グラフ ---
        weight_fig = go.Figure()
        weight_fig.add_trace(go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["Weight"],
            mode="lines+markers",
            name="体重",
            marker=dict(size=10, color='green'),
            line=dict(width=3, color='green'),
            hovertemplate="日付: %{x|%Y年%m月%d日}<br>体重: %{y:.1f} kg"
        ))
        if show_moving_avg:
            weight_fig.add_trace(go.Scatter(
                x=df_filtered["Date"],
                y=df_filtered["Weight"].rolling(window=3, min_periods=1).mean(),
                mode="lines",
                name="体重移動平均",
                line=dict(dash="dash", width=2, color='darkgreen')
            ))
        if show_regression and len(df_filtered) >= 2:
            x_num = df_filtered["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            model = LinearRegression().fit(x_num, df_filtered["Weight"])
            y_pred = model.predict(x_num)
            weight_fig.add_trace(go.Scatter(
                x=df_filtered["Date"],
                y=y_pred,
                mode="lines",
                name="体重回帰直線",
                line=dict(dash="dot", width=2, color='gray')
            ))
        if (end_date - start_date).days > 60:
            weight_fig.update_xaxes(dtick="M1", tickformat="%Y年%m月")
        else:
            weight_fig.update_xaxes(tickformat="%Y年%m月%d日")
        weight_fig.update_layout(title="体重の推移", **layout_common)

        st.plotly_chart(ffm_fig, use_container_width=True)
        st.plotly_chart(weight_fig, use_container_width=True)

    except Exception as e:
        st.error("❌ CSVの読み込みまたは処理中にエラーが発生しました。形式を確認してください。")
        st.exception(e)
