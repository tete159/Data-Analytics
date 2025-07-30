import pandas as pd

ruta_df = '/content/drive/MyDrive/2. Segmento_Datos completos todo re todo.csv'
df = pd.read_csv(ruta_df, sep=';', encoding='utf-8')
df.columns = df.columns.str.strip()

# --- Preparar DataFrame de secuencia ---
df_seq = df[['Nro Documento', 'DESTINO', 'FECHA_INICIO_VIGENCIA']].copy()
df_seq['FECHA_INICIO_VIGENCIA'] = pd.to_datetime(df_seq['FECHA_INICIO_VIGENCIA'], errors='coerce')
df_seq = df_seq.dropna(subset=['Nro Documento', 'FECHA_INICIO_VIGENCIA'])

# --- Ordenar cronológicamente ---
df_seq = df_seq.sort_values(by=['Nro Documento', 'FECHA_INICIO_VIGENCIA'])
df_seq['ORDEN_VIAJE'] = df_seq.groupby('Nro Documento').cumcount() + 1

# --- Crear secuencia por documento ---
secuencias = df_seq.groupby('Nro Documento')['DESTINO'].apply(list).reset_index(name='SECUENCIA')

# --- Filtrar personas con al menos 3 viajes ---
secuencias_top = secuencias[secuencias['SECUENCIA'].apply(len) > 2].copy()

# --- Extraer D1, D2, D3 ---
secuencias_top['D1'] = secuencias_top['SECUENCIA'].apply(lambda x: x[0])
secuencias_top['D2'] = secuencias_top['SECUENCIA'].apply(lambda x: x[1])
secuencias_top['D3'] = secuencias_top['SECUENCIA'].apply(lambda x: x[2])

# --- Contar combinaciones ---
transiciones_d3 = (
    secuencias_top
    .groupby(['D1', 'D2', 'D3'])
    .size()
    .reset_index(name='Cantidad')
)

# --- Calcular % relativo dentro de cada grupo D1 → D2 ---
transiciones_d3['Total_D1_D2'] = transiciones_d3.groupby(['D1', 'D2'])['Cantidad'].transform('sum')
transiciones_d3['Porcentaje'] = (transiciones_d3['Cantidad'] / transiciones_d3['Total_D1_D2'] * 100).round(2)

# --- Mostrar resultados como tabla legible ---
print("D1         → D2         → D3             | Cantidad | Porcentaje")
print("-" * 70)
for _, row in transiciones_d3.iterrows():
    print(f"{row['D1']:<10} → {row['D2']:<10} → {row['D3']:<15} | {row['Cantidad']:>8} | {row['Porcentaje']:>9.2f}%")
