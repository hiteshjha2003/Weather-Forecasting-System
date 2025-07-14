import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
from datetime import datetime
import logging
from data_loading import load_data
from data_preprocesssing import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data_info(df):
    """Display dataset information, missing values, data types, and summary statistics."""
    logging.info("Generating dataset information")
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    print("\nDataset Shape:")
    print(df.shape)
    print("\nUnique Values in Categorical Columns:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].unique()}")
    print("\nSummary Statistics of Numerical Variables:")
    print(df.describe().transpose())
    return df

def save_plot(fig, filename, output_dir, html_path=None):
    """Save a Plotly figure as PNG (and optionally HTML) with error handling."""
    try:
        fig.write_image(os.path.join(output_dir, filename))
        logging.info(f"Saved plot: {filename}")
        if html_path:
            fig.write_html(os.path.join(output_dir, html_path))
            logging.info(f"Saved HTML: {html_path}")
    except Exception as e:
        logging.error(f"Failed to save {filename}: {e}")

def perform_eda(df, output_dir=None):
    """Perform enhanced EDA with comprehensive visualizations and robust error handling."""
    # Create output directory with timestamp
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f"eda_plots"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Drop unnecessary column if present
    if 'Daily Summary' in df.columns:
        df = df.drop('Daily Summary', axis=1)
        logging.info("Dropped 'Daily Summary' column")

    # Ensure datetime parsing and feature extraction
    if 'Formatted Date' in df.columns:
        try:
            df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True, errors='coerce')
            logging.info("Parsed 'Formatted Date' as datetime")
        except Exception as e:
            logging.warning(f"Failed to parse 'Formatted Date': {e}")

    # Derive Month
    if 'Formatted Date' in df.columns:
        df['Month'] = df['Formatted Date'].dt.month

    # Derive UTC offset if missing
    if 'UTC offset' not in df.columns and 'Formatted Date' in df.columns:
        try:
            df['UTC offset'] = df['Formatted Date'].dt.tz.utcoffset().dt.total_seconds() / 3600
            logging.info("Derived 'UTC offset' from 'Formatted Date'")
        except Exception as e:
            df['UTC offset'] = np.nan
            logging.warning(f"Failed to derive UTC offset: {e}")

    # Separate numerical/categorical
    df_num = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(include=['object'])
    logging.info(f"Numerical columns: {list(df_num.columns)}")
    logging.info(f"Categorical columns: {list(df_cat.columns)}")

    # Fill missing numeric values
    df_num = df_num.fillna(df_num.mean())
    df[df_num.columns] = df_num
    logging.info("Filled missing numerical values with mean")

    # 1. Missing Values Heatmap
    if df.isnull().any().any():
        missing_data = df.isnull().astype(int)
        fig = ff.create_annotated_heatmap(
            z=missing_data.values.T,
            x=list(missing_data.index),
            y=list(missing_data.columns),
            colorscale='Greys', showscale=True
        )
        fig.update_layout(title='Missing Values Heatmap', width=1000, height=400)
        save_plot(fig, "missing_values_heatmap.png", output_dir)

    # 2–4. Outlier & Distribution Plots
    for col in df_num.columns:
        fig = px.box(df, y=col, title=f'Box Plot of {col}')
        fig.update_layout(showlegend=False)
        save_plot(fig, f"{col}_boxplot.png", output_dir)

        fig = px.violin(df, y=col, title=f'Violin Plot of {col}', box=True, points='outliers')
        fig.update_layout(showlegend=False)
        save_plot(fig, f"{col}_violinplot.png", output_dir)

        fig = ff.create_distplot([df_num[col].dropna()], [col], show_hist=False, colors=['#1f77b4'])
        fig.update_layout(title=f'KDE Plot of {col}', xaxis_title=col, yaxis_title='Density')
        save_plot(fig, f"{col}_kdeplot.png", output_dir)

    # 5. Skewness & Kurtosis
    print("\nSkewness and Kurtosis of Numerical Features:")
    for col in df_num.columns:
        skewness, kurtosis = df_num[col].skew(), df_num[col].kurt()
        print(f"{col} - Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
        if abs(skewness) > 1:
            logging.info(f"{col} is highly skewed: {skewness:.2f}")
        if kurtosis > 3:
            logging.info(f"{col} has high kurtosis: {kurtosis:.2f}")

    # 6–7. Categorical Distribution Bars
    if 'Summary' in df_cat.columns:
        summary_counts = df['Summary'].value_counts().reset_index()
        summary_counts.columns = ['Summary', 'Count']
        fig = px.bar(summary_counts, x='Summary', y='Count', title='Summary of Weather',
                     color='Count', color_continuous_scale='Rainbow')
        fig.update_layout(xaxis_title='Weather Summary', yaxis_title='Count', xaxis_tickangle=45)
        save_plot(fig, "summary_barplot.png", output_dir)

    if 'Precip Type' in df_cat.columns:
        precip_counts = df['Precip Type'].value_counts().reset_index()
        precip_counts.columns = ['Precip Type', 'Count']
        fig = px.bar(precip_counts, x='Precip Type', y='Count', title='Summary of Precip Type',
                     color='Count', color_continuous_scale='Rainbow')
        fig.update_layout(xaxis_title='Precipitation Type', yaxis_title='Count')
        save_plot(fig, "precip_type_barplot.png", output_dir)

    # 8. Stacked Bar: Precip vs Summary
    if 'Precip Type' in df.columns and 'Summary' in df.columns:
        stacked_data = df.groupby(['Summary', 'Precip Type']).size().unstack(fill_value=0).reset_index()
        fig = go.Figure(data=[
            go.Bar(name=precip, x=stacked_data['Summary'], y=stacked_data[precip])
            for precip in stacked_data.columns[1:]
        ])
        fig.update_layout(barmode='stack', title='Precip Type Distribution by Summary',
                         xaxis_title='Weather Summary', yaxis_title='Count', xaxis_tickangle=45)
        save_plot(fig, "precip_vs_summary_stacked_bar.png", output_dir)

    # 9–10. Category vs Numeric boxplots
    if 'Precip Type' in df.columns:
        for col in df_num.columns:
            fig = px.box(df, x='Precip Type', y=col, title=f'Precip Type vs {col}',
                         color='Precip Type')
            save_plot(fig, f"precip_vs_{col}_boxplot.png", output_dir)

    if 'Summary' in df.columns:
        for col in df_num.columns:
            fig = px.box(df, x='Summary', y=col, title=f'Summary vs {col}',
                         color='Summary')
            save_plot(fig, f"summary_vs_{col}_boxplot.png", output_dir)

    # 11. Correlation Heatmap
    corr = df_num.corr()
    fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                                      colorscale='Viridis', annotation_text=np.round(corr.values,2))
    fig.update_layout(title='Correlation Heatmap', width=800, height=600)
    save_plot(fig, "correlation_heatmap.png", output_dir)

    # 12. Pair Plot
    if 'Precip Type' in df.columns and len(df_num.columns) > 1:
        fig = px.scatter_matrix(df, dimensions=df_num.columns, color='Precip Type',
                                title='Pair Plot by Precip Type')
        fig.update_layout(width=1200, height=1200)
        save_plot(fig, "pair_plot.png", output_dir)

    # 13. 3D Scatter
    if all(c in df.columns for c in ['Humidity', 'Temperature (C)', 'Wind Speed (km/h)']):
        fig = px.scatter_3d(df, x='Humidity', y='Temperature (C)', z='Wind Speed (km/h)',
                            color='Precip Type' if 'Precip Type' in df.columns else None,
                            title='3D Scatter: Humidity vs Temp vs Wind')
        save_plot(fig, "3d_scatter_plot.png", output_dir)

    # 14. KDE Contour
    if all(c in df.columns for c in ['Humidity', 'Temperature (C)', 'UTC offset']):
        fig = px.density_contour(df, x='Humidity', y='Temperature (C)', color='UTC offset',
                                 title='KDE: Humidity vs Temp by UTC Offset')
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        save_plot(fig, "humidity_temperature_kde.png", output_dir)

    # 15. Temperature Trends by Month
    if 'Month' in df.columns:
        monthly_temp = df.groupby('Month')['Temperature (C)'].mean().reset_index()
        fig = px.line(monthly_temp, x='Month', y='Temperature (C)', title='Average Temp by Month', markers=True)
        save_plot(fig, "temperature_by_month.png", output_dir)

        fig = px.box(df, x='Month', y='Temperature (C)', title='Temperature Distribution by Month')
        save_plot(fig, "temperature_by_month_boxplot.png", output_dir)

    # 16. Time Series Plot
    if 'Formatted Date' in df.columns:
        try:
            df_sorted = df.sort_values('Formatted Date')
            fig = px.line(df_sorted, x='Formatted Date', y='Temperature (C)', title='Temperature Time Series')
            save_plot(fig, "temperature_time_series.png", output_dir)
        except Exception as e:
            logging.warning(f"Failed time series plot: {e}")

    # Final Insights Summary
    print("\nKey EDA Insights:")
    total_missing = df.isnull().sum().sum()
    print(f"- Total missing values after imputation: {total_missing}")
    for col in df_num.columns:
        q1, q3 = df_num[col].quantile([0.25,0.75])
        iqr = q3 - q1
        outliers = ((df_num[col] < q1-1.5*iqr) | (df_num[col] > q3+1.5*iqr)).sum()
        print(f"- {col}: {outliers} outliers (IQR method)")
    if 'Summary' in df.columns:
        top4 = df['Summary'].value_counts().head(4).index.tolist()
        print(f"- Common weather summaries: {', '.join(top4)}")
    if 'Precip Type' in df.columns:
        precip_dist = df['Precip Type'].value_counts(normalize=True) * 100
        print(f"- Precipitation breakdown: Rain {precip_dist.get('rain',0):.1f}%, Snow {precip_dist.get('snow',0):.1f}%")
    high_corr = corr.abs().stack().loc[lambda x: (x>0.7)&(x<1)]
    if not high_corr.empty:
        print("- Strong correlations (>0.7):")
        for (c1,c2), val in high_corr.items():
            print(f"  • {c1} vs {c2}: {val:.2f}")
    if 'Month' in df.columns:
        monthly_means = df.groupby('Month')['Temperature (C)'].mean()
        print(f"- Warmest month: {monthly_means.idxmax()} ({monthly_means.max():.1f}°C), coldest: {monthly_means.idxmin()} ({monthly_means.min():.1f}°C).")
    logging.info("EDA completed successfully")

if __name__ == "__main__":
    file_path = "weatherHistory.csv"
    try:
        df = load_data(file_path)
        df_processed, _ = preprocess_data(df)
        df = get_data_info(df_processed)
        perform_eda(df)
    except Exception as e:
        logging.error(f"Failed to execute EDA: {e}")
