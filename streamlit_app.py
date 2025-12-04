# /workspaces/DC_LAB2/data/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set the title and favicon
st.set_page_config(
    page_title='Spotify Music Dashboard',
    page_icon='🎵',
    layout='wide'
)

# Title
st.title('🎵 Spotify Music Dashboard')
st.markdown('Explore Spotify music data with interactive visualizations')

# -------------------------------------------------------------------------
# 数据加载函数
# -------------------------------------------------------------------------

@st.cache_data
def load_spotify_data():
    """加载Spotify数据，尝试多种可能的文件路径"""
    
    # 可能的文件路径列表（从最可能到最不可能）
    possible_paths = [
        # 1. 当前目录下的文件
        Path(__file__).parent / 'spotify_data clean.csv',
        Path(__file__).parent / 'spotify_data_clean.csv',
        Path(__file__).parent / 'spotify.csv',
        
        # 2. 父目录中的文件
        Path(__file__).parent.parent / 'spotify_data clean.csv',
        Path(__file__).parent.parent / 'spotify_data_clean.csv',
        
        # 3. 数据目录中的文件
        Path(__file__).parent / 'data' / 'spotify_data clean.csv',
        Path(__file__).parent.parent / 'data' / 'spotify_data clean.csv',
    ]
    
    # 尝试每个路径
    for file_path in possible_paths:
        if os.path.exists(file_path):
            try:
                st.sidebar.success(f"✅ 找到数据文件: {file_path}")
                df = pd.read_csv(file_path)
                # 清理列名（移除前后空格）
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.sidebar.error(f"❌ 读取文件时出错 {file_path}: {e}")
                continue
    
    # 如果找不到文件，返回空DataFrame
    return pd.DataFrame()

# -------------------------------------------------------------------------
# 加载数据
# -------------------------------------------------------------------------

# 显示加载状态
with st.spinner('正在加载数据...'):
    df = load_spotify_data()

# -------------------------------------------------------------------------
# 主界面
# -------------------------------------------------------------------------

if df.empty:
    st.warning("""
    ## ⚠️ 未找到Spotify数据文件
    
    请确保 `spotify_data clean.csv` 文件在以下位置之一：
    
    1. **当前目录** (`/workspaces/DC_LAB2/data/`)
    2. **上级目录** (`/workspaces/DC_LAB2/`)
    3. **数据子目录** (`/workspaces/DC_LAB2/data/data/`)
    
    或者上传你的数据文件：
    """)
    
    # 文件上传选项
    uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.success(f"✅ 成功加载 {uploaded_file.name}")
            # 重新运行以显示数据
            st.rerun()
        except Exception as e:
            st.error(f"❌ 读取文件时出错: {e}")
else:
    # 显示数据概览
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 数据概览")
    st.sidebar.write(f"**行数:** {df.shape[0]:,}")
    st.sidebar.write(f"**列数:** {df.shape[1]}")
    
    # 显示前几列的名称
    st.sidebar.write("**列名:**")
    for i, col in enumerate(df.columns[:10]):
        st.sidebar.write(f"- {col}")
    if len(df.columns) > 10:
        st.sidebar.write(f"... 还有 {len(df.columns) - 10} 列")
    
    # 显示内存使用
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    st.sidebar.write(f"**内存使用:** {memory_mb:.2f} MB")
    
    # -----------------------------------------------------------------
    # 主内容区
    # -----------------------------------------------------------------
    
    # 选项卡布局
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 数据概览", 
        "🔍 数据分析", 
        "📈 可视化", 
        "💾 导出数据"
    ])
    
    # 选项卡1: 数据概览
    with tab1:
        st.header("数据预览")
        
        # 显示前几行数据
        st.subheader("前10行数据")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 显示数据信息
        st.subheader("数据信息")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总行数", f"{df.shape[0]:,}")
        
        with col2:
            st.metric("总列数", df.shape[1])
        
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("数值列", len(numeric_cols))
        
        # 显示列详情
        with st.expander("查看列详情"):
            col_info = pd.DataFrame({
                '列名': df.columns,
                '数据类型': df.dtypes.astype(str),
                '非空值数量': df.count().values,
                '空值数量': df.isnull().sum().values,
                '唯一值数量': df.nunique().values
            })
            st.dataframe(col_info, use_container_width=True)
        
        # 显示数据统计摘要
        st.subheader("数值列统计摘要")
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("没有数值列")
    
    # 选项卡2: 数据分析
    with tab2:
        st.header("数据分析")
        
        # 选择分析列
        if len(df.columns) > 0:
            selected_column = st.selectbox(
                "选择要分析的列",
                options=df.columns
            )
            
            if selected_column:
                st.subheader(f"列: {selected_column}")
                
                # 显示列信息
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("数据类型", str(df[selected_column].dtype))
                
                with col2:
                    st.metric("非空值数", df[selected_column].count())
                
                with col3:
                    st.metric("空值数", df[selected_column].isnull().sum())
                
                with col4:
                    st.metric("唯一值数", df[selected_column].nunique())
                
                # 如果是数值列，显示统计信息
                if pd.api.types.is_numeric_dtype(df[selected_column]):
                    st.subheader("数值统计")
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("平均值", f"{df[selected_column].mean():.2f}")
                    
                    with stat_col2:
                        st.metric("中位数", f"{df[selected_column].median():.2f}")
                    
                    with stat_col3:
                        st.metric("最小值", f"{df[selected_column].min():.2f}")
                    
                    with stat_col4:
                        st.metric("最大值", f"{df[selected_column].max():.2f}")
                    
                    # 显示分布
                    st.subheader("值分布")
                    
                    # 创建直方图数据
                    hist_data = df[selected_column].dropna()
                    if len(hist_data) > 0:
                        # 创建频率分布
                        hist, bin_edges = np.histogram(hist_data, bins=20)
                        
                        # 创建数据框用于显示
                        hist_df = pd.DataFrame({
                            '区间起始': bin_edges[:-1],
                            '区间结束': bin_edges[1:],
                            '频数': hist,
                            '频率(%)': (hist / len(hist_data) * 100).round(2)
                        })
                        
                        st.dataframe(hist_df, use_container_width=True)
                        
                        # 简单的条形图表示
                        st.bar_chart(pd.DataFrame({selected_column: hist}))
                
                # 如果是分类列，显示频率分布
                else:
                    st.subheader("类别分布")
                    
                    # 计算频率
                    value_counts = df[selected_column].value_counts().head(20)  # 显示前20个
                    
                    # 显示频率表
                    freq_df = pd.DataFrame({
                        '值': value_counts.index,
                        '频数': value_counts.values,
                        '频率(%)': (value_counts.values / len(df) * 100).round(2)
                    })
                    
                    st.dataframe(freq_df, use_container_width=True)
                    
                    # 简单的条形图
                    st.bar_chart(value_counts)
        
        # 相关性分析（如果有多个数值列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("相关性分析")
            
            # 选择要分析的相关性列
            corr_cols = st.multiselect(
                "选择列进行相关性分析",
                options=numeric_cols.tolist(),
                default=numeric_cols[:min(5, len(numeric_cols))].tolist()
            )
            
            if len(corr_cols) > 1:
                # 计算相关性矩阵
                corr_matrix = df[corr_cols].corr().round(3)
                
                # 显示相关性矩阵
                st.write("相关性矩阵 (Pearson相关系数):")
                
                # 使用样式突出显示
                def highlight_correlations(val):
                    color = ''
                    if val > 0.7:
                        color = 'background-color: #90EE90'  # 浅绿色
                    elif val < -0.7:
                        color = 'background-color: #FFB6C1'  # 浅红色
                    elif val > 0.3:
                        color = 'background-color: #F0F8FF'  # 浅蓝色
                    elif val < -0.3:
                        color = 'background-color: #FFF0F5'  # 浅粉色
                    return color
                
                st.dataframe(
                    corr_matrix.style.applymap(highlight_correlations),
                    use_container_width=True
                )
                
                # 显示强相关性
                strong_corrs = []
                for i in range(len(corr_cols)):
                    for j in range(i+1, len(corr_cols)):
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.7:
                            strong_corrs.append((corr_cols[i], corr_cols[j], corr))
                
                if strong_corrs:
                    st.write("**强相关性 (> |0.7|):**")
                    for col1, col2, corr in strong_corrs:
                        st.write(f"- {col1} 和 {col2}: {corr:.3f}")
    
    # 选项卡3: 可视化
    with tab3:
        st.header("数据可视化")
        
        # 选择可视化类型
        viz_type = st.selectbox(
            "选择可视化类型",
            ["折线图", "条形图", "面积图", "散点图"]
        )
        
        # 选择X轴和Y轴
        col1, col2 = st.columns(2)
        
        with col1:
            # X轴选择
            x_axis = st.selectbox(
                "选择X轴",
                options=df.columns.tolist()
            )
        
        with col2:
            # Y轴选择（如果是折线图、条形图、面积图）
            if viz_type != "散点图":
                y_axis = st.selectbox(
                    "选择Y轴",
                    options=df.select_dtypes(include=[np.number]).columns.tolist()
                )
            else:
                # 散点图需要第二个数值列
                y_axis = st.selectbox(
                    "选择Y轴",
                    options=df.select_dtypes(include=[np.number]).columns.tolist()
                )
        
        # 添加分组选项（如果数据中有分类列）
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            group_by = st.selectbox(
                "按此列分组（可选）",
                options=["无"] + categorical_cols.tolist()
            )
        else:
            group_by = "无"
        
        # 显示图表
        if viz_type == "折线图" and x_axis and y_axis:
            st.subheader(f"{y_axis} 随时间/顺序变化")
            
            if group_by != "无" and group_by in df.columns:
                # 按分组显示多条线
                groups = df[group_by].unique()[:10]  # 限制前10个组
                for group in groups:
                    group_data = df[df[group_by] == group]
                    if len(group_data) > 0:
                        st.line_chart(group_data.set_index(x_axis)[y_axis])
            else:
                # 显示单条线
                st.line_chart(df.set_index(x_axis)[y_axis])
        
        elif viz_type == "条形图" and x_axis and y_axis:
            st.subheader(f"{y_axis} 按 {x_axis} 分组")
            
            # 如果是分类X轴，显示条形图
            if not pd.api.types.is_numeric_dtype(df[x_axis]):
                # 按X轴分组计算Y轴平均值
                bar_data = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False).head(20)
                st.bar_chart(bar_data)
            else:
                st.info("X轴应为分类变量以显示条形图")
        
        elif viz_type == "面积图" and x_axis and y_axis:
            st.subheader(f"{y_axis} 面积图")
            st.area_chart(df.set_index(x_axis)[y_axis])
        
        elif viz_type == "散点图" and x_axis and y_axis:
            st.subheader(f"{y_axis} 与 {x_axis} 的关系")
            
            # 创建散点图数据
            scatter_data = df[[x_axis, y_axis]].dropna()
            if len(scatter_data) > 0:
                # 使用Streamlit的折线图模拟散点图
                st.line_chart(scatter_data.set_index(x_axis)[y_axis])
            else:
                st.warning("没有足够的数据显示散点图")
    
    # 选项卡4: 导出数据
    with tab4:
        st.header("数据导出")
        
        # 数据筛选选项
        st.subheader("筛选要导出的数据")
        
        # 选择列
        columns_to_export = st.multiselect(
            "选择要导出的列",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:min(10, len(df.columns))]
        )
        
        # 选择行数
        rows_to_export = st.slider(
            "选择要导出的行数",
            min_value=1,
            max_value=len(df),
            value=min(1000, len(df))
        )
        
        if columns_to_export:
            # 创建要导出的数据
            export_df = df[columns_to_export].head(rows_to_export)
            
            # 显示预览
            st.subheader("导出数据预览")
            st.dataframe(export_df.head(10), use_container_width=True)
            
            # 导出选项
            st.subheader("导出选项")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 导出为CSV
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载为CSV",
                    data=csv_data,
                    file_name="spotify_data_export.csv",
                    mime="text/csv"
                )
            
            with col2:
                # 导出为JSON
                json_data = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 下载为JSON",
                    data=json_data,
                    file_name="spotify_data_export.json",
                    mime="application/json"
                )
        else:
            st.info("请选择要导出的列")

# -------------------------------------------------------------------------
# 页脚
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🎵 Spotify Music Dashboard | 使用Streamlit构建</p>
        <p>数据来源: Spotify全球音乐数据集</p>
    </div>
    """,
    unsafe_allow_html=True
)
