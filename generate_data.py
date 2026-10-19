import numpy as np
import pandas as pd

np.random.seed(42)

jumlah_pelanggan = 500

recency = np.concatenate([
    np.random.randint(1, 20, 125),    
    np.random.randint(15, 60, 150),    
    np.random.randint(10, 45, 125),    
    np.random.randint(90, 240, 100)    
])

frequency = np.concatenate([
    np.random.randint(18, 35, 125),    
    np.random.randint(8, 18, 150),     
    np.random.randint(1, 7, 125),      
    np.random.randint(1, 6, 100)       
])

monetary = np.concatenate([
    np.random.randint(7500000, 15000000, 125), 
    np.random.randint(3000000, 7500000, 150),  
    np.random.randint(500000, 3000000, 125),   
    np.random.randint(250000, 2000000, 100)    
])

satisfaction = np.concatenate([
    np.round(np.random.uniform(4.2, 5.0, 125), 1),
    np.round(np.random.uniform(3.6, 4.6, 150), 1),
    np.round(np.random.uniform(3.2, 4.5, 125), 1),
    np.round(np.random.uniform(1.5, 3.2, 100), 1)
])

indeks_acak = np.random.permutation(jumlah_pelanggan)

kota_pilihan = ['Bengkulu', 'Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Yogyakarta']
kategori_produk = ['Elektronik', 'Fashion & Apparel', 'Kebutuhan Rumah Tangga', 'Kesehatan & Kecantikan', 'Makanan & Minuman']
metode_bayar = ['E-Wallet', 'Transfer Bank', 'Kartu Kredit', 'QRIS', 'COD']

df = pd.DataFrame({
    'Customer_ID': [f'CUST-{i:04d}' for i in range(1, jumlah_pelanggan + 1)],
    'Gender': np.random.choice(['Wanita', 'Pria'], jumlah_pelanggan, p=[0.55, 0.45]),
    'Age': np.random.randint(19, 58, jumlah_pelanggan),
    'City': np.random.choice(kota_pilihan, jumlah_pelanggan, p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10]),
    'Product_Category': np.random.choice(kategori_produk, jumlah_pelanggan),
    'Payment_Method': np.random.choice(metode_bayar, jumlah_pelanggan),
    'Recency_Days': recency[indeks_acak],
    'Frequency_Orders': frequency[indeks_acak],
    'Monetary_Spend': monetary[indeks_acak],
    'Satisfaction_Score': satisfaction[indeks_acak]
})

nama_file = 'retail_customers.csv'
df.to_csv(nama_file, index=False)
print(f"Berhasil, File '{nama_file}' telah dibuat dengan {len(df)} baris data pelanggan.")