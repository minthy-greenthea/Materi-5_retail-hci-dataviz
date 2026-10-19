import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs(".streamlit", exist_ok=True)
config_path = ".streamlit/config.toml"
config_content = """[theme]
base = "light"
primaryColor = "#3A5A40"
backgroundColor = "#F4F3EB"
secondaryBackgroundColor = "#9CAF88"
textColor = "#1A241B"
font = "sans serif"
"""
if not os.path.exists(config_path) or open(config_path).read() != config_content:
    with open(config_path, "w") as f:
        f.write(config_content)

st.set_page_config(
    page_title="RetailSense • HCI & AI Data Visualization",
    layout="wide"
)

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/remixicon@4.3.0/fonts/remixicon.css" rel="stylesheet">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #F4F3EB !important;
        color: #1A241B !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background-color: #F4F3EB !important;
        border-bottom: 1px solid #D5DDD0 !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #9CAF88 0%, #869D72 100%) !important;
        border-right: 3px solid #C5A869 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #162218 !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FCFBF7 !important;
        border: 1.5px solid #3A5A40 !important;
        border-radius: 8px !important;
        color: #1A241B !important;
    }

    div[data-baseweb="popover"] div,
    ul[data-baseweb="menu"] li {
        background-color: #FCFBF7 !important;
        color: #1A241B !important;
    }

    span[data-baseweb="tag"] {
        background-color: #2D4739 !important;
        border: 1px solid #C5A869 !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
    }

    span[data-baseweb="tag"] span,
    span[data-baseweb="tag"] svg {
        color: #F9F9F6 !important;
        fill: #F9F9F6 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
        background-color: #2D4739 !important;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #C5A869 !important;
        border: 2px solid #2D4739 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
    }

    div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
    div[data-testid="stSlider"] div[data-testid="stTickBarMax"],
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #162218 !important;
        font-weight: 700 !important;
    }

    .genshin-banner {
        background: linear-gradient(135deg, #223528 0%, #35523B 100%);
        border: 2px solid #C5A869;
        outline: 1px solid #C5A869;
        outline-offset: -7px;
        border-radius: 12px;
        padding: 26px 32px;
        text-align: center;
        color: #F9F9F6;
        box-shadow: 0 10px 24px rgba(34, 53, 40, 0.14);
        margin-bottom: 24px;
    }

    .genshin-kicker {
        font-size: 0.78rem;
        color: #DEC484;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .genshin-banner h1 {
        font-family: 'Cinzel', serif !important;
        color: #FAF8F2 !important;
        font-size: 1.85rem !important;
        letter-spacing: 1.2px;
        margin: 0;
    }

    .genshin-divider {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin: 10px 0 12px 0;
        color: #DEC484;
    }

    .genshin-line {
        height: 1px;
        width: 90px;
        background: linear-gradient(90deg, transparent, #DEC484, transparent);
    }

    .genshin-subtitle {
        font-size: 0.92rem;
        color: #E6ECE2;
        max-width: 820px;
        margin: 0 auto;
        line-height: 1.55;
    }

    .section-ribbon {
        background: linear-gradient(90deg, #7B9267 0%, #9CAF88 70%, rgba(244,243,235,0) 100%);
        color: #FFFFFF !important;
        padding: 9px 18px;
        border-left: 5px solid #223528;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1.02rem;
        letter-spacing: 0.4px;
        margin: 18px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .kpi-card {
        background-color: #FCFBF7;
        border: 1.5px solid #C5CEB9;
        border-top: 4px solid #3A5A40;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(26, 36, 27, 0.05);
        height: 130px;
    }

    .kpi-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #3D4F40;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .kpi-title i {
        color: #3A5A40;
        font-size: 1.05rem;
    }

    .kpi-value {
        font-family: 'Cinzel', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1A241B;
        margin-bottom: 6px;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #3D4F40;
        font-weight: 500;
    }

    .kpi-badge-sage {
        display: inline-block;
        background-color: #E6EFE3;
        color: #26402B;
        border: 1px solid #AEC4A7;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 600;
    }

    .kpi-badge-gold {
        display: inline-block;
        background-color: #F6EFE0;
        color: #6E5223;
        border: 1px solid #D8C08E;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 600;
    }

    div[data-testid="stPlotlyChart"] {
        background-color: #FCFBF7;
        border: 1.5px solid #CFD8C7;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 14px rgba(26, 36, 27, 0.04);
    }

    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
    }

    button[data-baseweb="tab"] {
        background-color: #E3E8DC !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 22px !important;
        border: 1.5px solid #BFC9B4 !important;
        border-bottom: none !important;
    }

    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] span {
        color: #1E2E21 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2D4739 !important;
        border-color: #2D4739 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] div,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #FCFBF7 !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #C5A869 !important;
        height: 3px !important;
    }

    div[data-baseweb="tab-border"] {
        background-color: #C5CEB9 !important;
    }

    .sidebar-header-box {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Cinzel', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #162218;
        padding-bottom: 6px;
        border-bottom: 1.5px solid #687D56;
        margin-bottom: 10px;
    }

    .sidebar-desc {
        font-size: 0.83rem;
        color: #1E2D20;
        line-height: 1.45;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="genshin-banner">
    <div class="genshin-kicker">
        <i class="ri-compass-3-line"></i> Informatics Practitioner Lecture Series • HCI & Data Visualization
    </div>
    <h1>RetailSense: Chronicle of Customer Intelligence</h1>
    <div class="genshin-divider">
        <div class="genshin-line"></div>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="#DEC484">
            <path d="M12 1L15.5 8.5L23 12L15.5 15.5L12 23L8.5 15.5L1 12L8.5 8.5L12 1Z"/>
        </svg>
        <div class="genshin-line"></div>
    </div>
    <div class="genshin-subtitle">
        Mengintegrasikan estetika <b>Human-Computer Interaction (HCI)</b>, <b>Visualisasi Data Interaktif</b>, dan 
        <b>Machine Learning (K-Means Clustering)</b> untuk mengubah data transaksi mentah menjadi keputusan strategis yang intuitif.
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("retail_customers.csv")
    
    fitur_rfm = ['Recency_Days', 'Frequency_Orders', 'Monetary_Spend']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df[fitur_rfm])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(rfm_scaled)
    
    profil_cluster = df.groupby('Cluster_ID').agg({
        'Recency_Days': 'mean',
        'Frequency_Orders': 'mean',
        'Monetary_Spend': 'mean'
    })
    
    cluster_sultan = profil_cluster['Monetary_Spend'].idxmax()
    sisa_1 = profil_cluster.drop(index=cluster_sultan)
    
    cluster_kabur = sisa_1['Recency_Days'].idxmax()
    sisa_2 = sisa_1.drop(index=cluster_kabur)
    
    cluster_setia = sisa_2['Monetary_Spend'].idxmax()
    cluster_potensial = sisa_2.drop(index=cluster_setia).index[0]
    
    pemetaan_segmen = {
        cluster_sultan: "Champions (Pelanggan Utama)",
        cluster_setia: "Loyal (Pelanggan Setia)",
        cluster_potensial: "Potential (Pelanggan Potensial)",
        cluster_kabur: "At-Risk (Perlu Retensi)"
    }
    
    df['AI_Segment'] = df['Cluster_ID'].map(pemetaan_segmen)
    return df

df = load_and_process_data()

WARNA_SEGMEN = {
    "Champions (Pelanggan Utama)": "#2D5A43",
    "Loyal (Pelanggan Setia)": "#6B8E63",
    "Potential (Pelanggan Potensial)": "#C59B48",
    "At-Risk (Perlu Retensi)": "#526D82"
}

def rapikan_grafik(fig, tinggi=420):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FCFBF7",
        plot_bgcolor="#F7F6F0",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1A241B", size=12),
        title=dict(font=dict(family="Cinzel, serif", size=15, color="#1A241B")),
        margin=dict(l=25, r=25, t=55, b=25),
        height=tinggi,
        legend=dict(
            bgcolor="rgba(252, 251, 247, 0.95)",
            bordercolor="#C5CEB9",
            borderwidth=1,
            font=dict(color="#1A241B", size=11),
            title_font=dict(color="#1A241B", size=12)
        )
    )
    fig.update_xaxes(
        gridcolor="#E2E6DC",
        linecolor="#BFC9B4",
        tickfont=dict(color="#1A241B", size=11),
        title_font=dict(color="#1A241B", size=12),
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor="#E2E6DC",
        linecolor="#BFC9B4",
        tickfont=dict(color="#1A241B", size=11),
        title_font=dict(color="#1A241B", size=12),
        zeroline=False
    )
    return fig

st.sidebar.markdown("""
<div class="sidebar-header-box">
    <i class="ri-equalizer-2-line"></i> Panel Filter & Navigasi
</div>
<div class="sidebar-desc">
    Saring data pelanggan secara dinamis sesuai prinsip <i>Visual Information-Seeking Mantra</i>.
</div>
""", unsafe_allow_html=True)

pilihan_kota = st.sidebar.multiselect(
    "Pilih Kota Pelanggan:",
    options=df['City'].unique(),
    default=df['City'].unique()
)

pilihan_kategori = st.sidebar.multiselect(
    "Pilih Kategori Produk:",
    options=df['Product_Category'].unique(),
    default=df['Product_Category'].unique()
)

rentang_usia = st.sidebar.slider(
    "Rentang Usia Pelanggan:",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

df_filtered = df[
    (df['City'].isin(pilihan_kota)) &
    (df['Product_Category'].isin(pilihan_kategori)) &
    (df['Age'].between(rentang_usia[0], rentang_usia[1]))
]

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="background-color: #FCFBF7; padding: 14px; border-radius: 10px; border: 1.5px solid #2D4739; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.06);">
    <div style="font-size: 0.78rem; color: #3D4F40; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
        <i class="ri-database-2-line"></i> Status Data Terfilter
    </div>
    <div style="font-family: 'Cinzel', serif; font-size: 1.25rem; font-weight: 700; color: #1A241B; margin-top: 4px;">
        {len(df_filtered)} / {len(df)} Pelanggan
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-ribbon"><i class="ri-dashboard-3-line"></i> Ringkasan Eksekutif (Overview Metrics)</div>', unsafe_allow_html=True)

total_pelanggan = len(df_filtered)
total_pendapatan = df_filtered['Monetary_Spend'].sum()
pendapatan_miliar = total_pendapatan / 1_000_000_000
rata_kepuasan = df_filtered['Satisfaction_Score'].mean() if total_pelanggan > 0 else 0
jumlah_at_risk = len(df_filtered[df_filtered['AI_Segment'] == "At-Risk (Perlu Retensi)"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><i class="ri-team-line"></i> Total Pelanggan Aktif</div>
        <div class="kpi-value">{total_pelanggan} Orang</div>
        <div class="kpi-sub"><span class="kpi-badge-sage"><i class="ri-checkbox-circle-line"></i> Aktif Terpantau</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><i class="ri-coins-line"></i> Total Nilai Transaksi</div>
        <div class="kpi-value">Rp {pendapatan_miliar:.2f} Miliar</div>
        <div class="kpi-sub">Akumulasi: Rp {total_pendapatan:,.0f}</div>
    </div>
    """.replace(",", "."), unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><i class="ri-award-line"></i> Indeks Kepuasan</div>
        <div class="kpi-value">{rata_kepuasan:.2f} / 5.00</div>
        <div class="kpi-sub"><span class="kpi-badge-sage"><i class="ri-star-smile-line"></i> Skor Ulasan Pelanggan</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color: #C5A869;">
        <div class="kpi-title"><i class="ri-shield-user-line" style="color: #8C6D3F;"></i> Segmen Perlu Retensi</div>
        <div class="kpi-value" style="color: #6E5223;">{jumlah_at_risk} Orang</div>
        <div class="kpi-sub"><span class="kpi-badge-gold"><i class="ri-focus-3-line"></i> Prioritas Tindak Lanjut</span></div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3 = st.tabs([
    "1. Peta Segmen AI (Data Visualization)",
    "2. Ruang Klaster 3D & Diagnostik",
    "3. Simulator Prediksi AI & Arsip Data"
])

with tab1:
    st.markdown('<div class="section-ribbon"><i class="ri-bubble-chart-line"></i> Pemetaan Perilaku Belanja Pelanggan (K-Means Clustering)</div>', unsafe_allow_html=True)
    col_kiri, col_kanan = st.columns([6, 4])
    
    with col_kiri:
        fig_scatter = px.scatter(
            df_filtered,
            x="Frequency_Orders",
            y="Monetary_Spend",
            color="AI_Segment",
            size="Satisfaction_Score",
            size_max=13,
            hover_data=["Customer_ID", "City", "Product_Category", "Recency_Days"],
            color_discrete_map=WARNA_SEGMEN,
            title="Korelasi Frekuensi Transaksi vs Total Pengeluaran",
            labels={
                "Frequency_Orders": "Frekuensi Belanja (Kali)",
                "Monetary_Spend": "Total Pengeluaran (Rp)",
                "AI_Segment": "Segmen AI"
            }
        )
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color='#FCFBF7'), opacity=0.9))
        st.plotly_chart(rapikan_grafik(fig_scatter, 430), use_container_width=True, theme=None)
        
    with col_kanan:
        fig_donut = px.pie(
            df_filtered,
            names="AI_Segment",
            hole=0.5,
            color="AI_Segment",
            color_discrete_map=WARNA_SEGMEN,
            title="Proporsi Segmen Pelanggan"
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent',
            textfont=dict(color='#FFFFFF', size=12),
            marker=dict(line=dict(color='#FCFBF7', width=2))
        )
        st.plotly_chart(rapikan_grafik(fig_donut, 430), use_container_width=True, theme=None)

    df_kategori = df_filtered.groupby(['Product_Category', 'AI_Segment'])['Monetary_Spend'].sum().reset_index()
    fig_bar = px.bar(
        df_kategori,
        x="Product_Category",
        y="Monetary_Spend",
        color="AI_Segment",
        barmode="group",
        color_discrete_map=WARNA_SEGMEN,
        title="Distribusi Nilai Transaksi Berdasarkan Kategori Produk & Segmen AI",
        labels={
            "Product_Category": "Kategori Produk",
            "Monetary_Spend": "Total Nilai Transaksi (Rp)",
            "AI_Segment": "Segmen AI"
        }
    )
    st.plotly_chart(rapikan_grafik(fig_bar, 410), use_container_width=True, theme=None)

with tab2:
    st.markdown('<div class="section-ribbon"><i class="ri-compass-discover-line"></i> Eksplorasi Ruang 3 Dimensi RFM & Analisis Kepuasan</div>', unsafe_allow_html=True)
    col_3d, col_box = st.columns([6, 4])
    
    with col_3d:
        fig_3d = px.scatter_3d(
            df_filtered,
            x="Recency_Days",
            y="Frequency_Orders",
            z="Monetary_Spend",
            color="AI_Segment",
            color_discrete_map=WARNA_SEGMEN,
            hover_name="Customer_ID",
            title="Peta Klaster 3D RFM (Interaktif)",
            labels={
                "Recency_Days": "Recency (Hari)",
                "Frequency_Orders": "Frequency (Order)",
                "Monetary_Spend": "Monetary (Rp)"
            }
        )
        fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color='#FCFBF7')))
        st.plotly_chart(rapikan_grafik(fig_3d, 520), use_container_width=True, theme=None)
        
    with col_box:
        fig_box = px.box(
            df_filtered,
            x="AI_Segment",
            y="Satisfaction_Score",
            color="AI_Segment",
            color_discrete_map=WARNA_SEGMEN,
            title="Distribusi Skor Kepuasan per Segmen",
            labels={
                "AI_Segment": "Segmen AI",
                "Satisfaction_Score": "Skor Kepuasan (1-5)"
            }
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(rapikan_grafik(fig_box, 520), use_container_width=True, theme=None)

with tab3:
    st.markdown('<div class="section-ribbon"><i class="ri-Cpu-line"></i> Simulator Prediksi AI Interaktif (Prescriptive Analytics)</div>', unsafe_allow_html=True)
    
    sim_col1, sim_col2 = st.columns([5, 5])
    
    with sim_col1:
        st.markdown('<div style="font-weight: 700; color: #1A241B; margin-bottom: 10px;"><i class="ri-sliders-line"></i> Atur Parameter Profil Pelanggan:</div>', unsafe_allow_html=True)
        input_recency = st.slider("1. Hari Sejak Transaksi Terakhir (Recency):", 1, 240, 20)
        input_frequency = st.slider("2. Total Frekuensi Belanja (Frequency):", 1, 35, 15)
        input_monetary = st.slider("3. Total Nilai Belanja (Monetary - Rp):", 250000, 15000000, 5000000, step=250000)
        
    with sim_col2:
        fitur_rfm = ['Recency_Days', 'Frequency_Orders', 'Monetary_Spend']
        scaler_sim = StandardScaler()
        rfm_scaled_all = scaler_sim.fit_transform(df[fitur_rfm])
        kmeans_sim = KMeans(n_clusters=4, random_state=42, n_init=10).fit(rfm_scaled_all)
        
        peta_id_ke_nama = df[['Cluster_ID', 'AI_Segment']].drop_duplicates().set_index('Cluster_ID')['AI_Segment'].to_dict()
        data_uji = pd.DataFrame([[input_recency, input_frequency, input_monetary]], columns=fitur_rfm)
        klaster_prediksi = kmeans_sim.predict(scaler_sim.transform(data_uji))[0]
        hasil_segmen = peta_id_ke_nama[klaster_prediksi]
        
        if "Champions" in hasil_segmen:
            warna_kartu, ikon_ri = "#2D5A43", "ri-vip-crown-2-line"
            rekomendasi = "Berikan akses eksklusif produk baru (VIP Early Access), reward loyalitas khusus, dan undangan program member prioritas."
        elif "Loyal" in hasil_segmen:
            warna_kartu, ikon_ri = "#6B8E63", "ri-medal-line"
            rekomendasi = "Tawarkan paket bundling (upselling) dan bonus poin belanja agar pelanggan ini naik kelas menjadi segmen Champions."
        elif "Potential" in hasil_segmen:
            warna_kartu, ikon_ri = "#C59B48", "ri-seedling-line"
            rekomendasi = "Kirimkan voucher diskon pembelian berikutnya dan rekomendasi produk personal untuk meningkatkan frekuensi transaksi."
        else:
            warna_kartu, ikon_ri = "#526D82", "ri-Restart-line"
            rekomendasi = "Pelanggan mulai jarang bertransaksi. Segera kirimkan kampanye re-engagement (penawaran khusus kembali belanja) dan survei layanan."

        st.markdown(f"""
        <div style="background-color: #FCFBF7; border: 2px solid {warna_kartu}; border-left: 8px solid {warna_kartu}; border-radius: 10px; padding: 22px; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
            <div style="font-size: 0.82rem; color: #3D4F40; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;">
                <i class="ri-sparkling-2-line"></i> Hasil Klasifikasi AI Real-Time
            </div>
            <div style="font-family: 'Cinzel', serif; font-size: 1.4rem; font-weight: 700; color: {warna_kartu}; margin: 8px 0 12px 0; display: flex; align-items: center; gap: 8px;">
                <i class="{ikon_ri}"></i> {hasil_segmen}
            </div>
            <div style="font-size: 0.92rem; color: #1A241B; line-height: 1.55;">
                <b>Rekomendasi Keputusan (Prescriptive Analytics):</b><br>{rekomendasi}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-ribbon"><i class="ri-file-list-3-line"></i> Arsip Data Pelanggan Terfilter (Details on Demand)</div>', unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True)