# --- MONTAJE DRIVE Y LIBRERÍAS ---
from google.colab import drive, files
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# --- CARGA DE DATOS ---
ruta_df = '/content/drive/MyDrive/2. Segmento_Datos completos todo re todo.csv'
ruta_costos = '/content/drive/MyDrive/Costos Prestacionales.csv'

df = pd.read_csv(ruta_df, sep=';', encoding='utf-8')
df_costos = pd.read_csv(ruta_costos, sep=';', encoding='utf-8')

# --- LIMPIEZA Y MERGE ---
df.columns = df.columns.str.strip()
df_costos.columns = df_costos.columns.str.strip()
df_costos['Total cost'] = (
    df_costos['Total cost']
    .astype(str)
    .str.replace(',', '.', regex=False)
    .str.replace(' ', '', regex=False)
)
df_costos['Total cost'] = pd.to_numeric(df_costos['Total cost'], errors='coerce')

df['clave_merge'] = df['NUM_VOUCHER'].astype(str).str.strip() + '___' + df['Nro Documento'].astype(str).str.strip()
df_costos['clave_merge'] = df_costos['Nro Voucher'].astype(str).str.strip() + '___' + df_costos['Nro Doc'].astype(str).str.strip()
df = df.merge(df_costos[['clave_merge', 'Total cost']], on='clave_merge', how='left')

# --- TRANSFORMACIONES ---
df['FECHA_EMISION'] = pd.to_datetime(df['FECHA_EMISION'], errors='coerce', dayfirst=True)
df['FECHA_INICIO_VIGENCIA'] = pd.to_datetime(df['FECHA_INICIO_VIGENCIA'], errors='coerce', dayfirst=True)
df['Fin Vigencia'] = pd.to_datetime(df['Fin Vigencia'], errors='coerce', dayfirst=True)

condiciones_edad = [
    df['Edad Actual'] >= 70,
    (df['Edad Actual'] >= 60) & (df['Edad Actual'] < 70),
    (df['Edad Actual'] >= 35) & (df['Edad Actual'] < 60),
    (df['Edad Actual'] >= 21) & (df['Edad Actual'] < 35),
    df['Edad Actual'] < 21
]
valores_edad = ["Adultos Mayores", "Senior", "Adultos", "Jóvenes", "Menor"]
df['Clasificacion edad'] = np.select(condiciones_edad, valores_edad, default="Sin clasificar")


df['PRODUCTO'] = df['PRODUCTO'].astype(str).str.lower()
df['Duracion_viaje_dias'] = (df['Fin Vigencia'] - df['FECHA_INICIO_VIGENCIA']).dt.days

df['tipo_estadia'] = np.select(
    [df['PRODUCTO'].str.contains("anual"), df['Duracion_viaje_dias'] > 60],
    ["Anual", "Larga estadía"],
    default="Módulo"
)
df['sufijo_amvr'] = np.where(df['PRODUCTO'].str.contains("reno"), " en AMVR", "")
df['TIPO_VENTA_FINAL'] = df['tipo_estadia'] + df['sufijo_amvr']

df['Mes_Inicio_Viaje'] = df['FECHA_INICIO_VIGENCIA'].dt.month

#Filtros
df = df[
    (df['PAIS_EMISION'] == 'ARGENTINA') &
    (df['FECHA_EMISION'].dt.year >= 2022) &
    (df['TIPO_VENTA_FINAL'] == 'Módulo')
].copy()

def obtener_top_meses_por_cluster(df, df_subset, n_clusters):
    resultado = []
    for cluster in range(1, n_clusters + 1):
        indices = df_subset[df_subset[f'cluster_nuevo_{n_clusters}'] == cluster].index
        meses = df.loc[indices, 'FECHA_INICIO_VIGENCIA'].dt.month
        conteo = meses.value_counts(normalize=True).round(3).head(3) * 100
        meses_str = ', '.join([f"{pd.to_datetime(f'2025-{m}-01').strftime('%b')} ({int(p)}%)" for m, p in conteo.items()])
        resultado.append(meses_str)
    return resultado

# ---------------------- FUNCIÓN DE RESUMEN ----------------------

def generar_resumen_por_pais(df, df_subset, pais=None):
    if pais:
        mask = df['PAIS_EMISION'] == pais
        df_pais = df[mask].copy()
        df_subset_pais = df_subset.loc[df[mask].index.intersection(df_subset.index)].copy()
    else:
        df_pais = df.copy()
        df_subset_pais = df_subset.copy()

    if df_subset_pais.empty:
        return pd.DataFrame({'Aviso': [f'No hay datos para {pais or "TOTAL"} en df_subset']})

    resumen = []
    gross_numerico = []

    for cluster in sorted(df_subset_pais['cluster_nuevo'].unique()):
        dfc = df_pais.loc[df_subset_pais[df_subset_pais['cluster_nuevo'] == cluster].index].copy()

        clientes = len(dfc)
        porcentaje = round(100 * clientes / len(df_subset_pais), 1)

        edades = dfc['Rango de edades'].value_counts(normalize=True).round(3) * 100
        top_edades = [f'{k} ({int(v)}%)' for k, v in edades.head(4).items()]

        actividad = dfc['Clasificación Actividad'].value_counts(normalize=True).round(3)
        estado = ', '.join([f'{k} {int(v*100)}%' for k, v in actividad.items()])

        tope = dfc['Tope de cobertura'].value_counts(normalize=True).round(3).head(3)
        tope_str = ', '.join([f'{k} {int(v*100)}%' for k, v in tope.items()])

        pax = dfc['Clasificación tipo PAX'].value_counts(normalize=True).round(3).head(3)
        pax_str = ', '.join([f'{k} {int(v*100)}%' for k, v in pax.items()])

        anticipacion = round(dfc['Anticipacion de compra'].astype(str).str.replace(',', '.', regex=False).astype(float).mean(), 1)
        compras = round(dfc['Cant. Compras'].astype(str).str.replace(',', '.', regex=False).astype(float).mean(), 2)

        canal = dfc['Canal Venta'].value_counts(normalize=True).round(3).head(3)
        canal_str = ', '.join([f'{k} {int(v*100)}%' for k, v in canal.items()])

        tipo = dfc['TIPO_VENTA_FINAL'].value_counts(normalize=True).round(3).head(4)
        tipo_str = ', '.join([f'{k} {int(v*100)}%' for k, v in tipo.items()])

        destinos = dfc['DESTINO'].value_counts(normalize=True).round(3).head(5)
        destino_str = ', '.join([f'{k} {int(v*100)}%' for k, v in destinos.items()])
        destino_nac = int((dfc['DESTINO'] == 'Territorio Nacional').mean() * 100)
        destino_int = 100 - destino_nac

        precio = dfc['Precio Unitario USD'].astype(str).str.replace(',', '.', regex=False).astype(float)
        pax_cant = dfc['Cantidad PAX por vouchers'].astype(str).str.replace(',', '.', regex=False).astype(float)
        ticket_medio = round(precio.mean(), 1)
        ticket_pax = round((precio / pax_cant).mean(), 1)
        gross = round(ticket_pax * clientes, 0)
        gross_str = f'USD {int(gross):,}'.replace(",", ".")
        gross_numerico.append(gross)

        # Costo total del cluster
        costo_cluster = dfc['Total cost'].sum()
        costo_cluster_str = f'USD {int(round(costo_cluster, 0)):,}'.replace(",", ".")
                # Costo % respecto al gross del mismo cluster
        costo_pct = round(100 * costo_cluster / gross, 1) if gross else 0




                # Estado de cliente
        estado_cliente = dfc['ESTADO'].value_counts(normalize=True).round(3).head(3)
        estado_cliente_str = ', '.join([f'{k} {int(v*100)}%' for k, v in estado_cliente.items()])

        resumen.append({
            'Segmento': f'Cluster {cluster}',
            'Clientes': clientes,
            '% Clientes': f'{porcentaje}%',
            'Edad 1': top_edades[0] if len(top_edades) > 0 else '',
            'Edad 2': top_edades[1] if len(top_edades) > 1 else '',
            'Edad 3': top_edades[2] if len(top_edades) > 2 else '',
            'Edad 4': top_edades[3] if len(top_edades) > 3 else '',
            'Estado Cliente': estado,
            'Tope cobertura': tope_str,
            'PAX por voucher': pax_str,
            'Anticipación compra': anticipacion,
            'Compras promedio': compras,
            'Canal de compra': canal_str,
            'Tipo de producto': tipo_str,
            'Destino nacional': f'{destino_nac}%',
            'Destino internacional': f'{destino_int}%',
            'Destino': destino_str,
            'Ticket medio': f'USD {ticket_medio}',
            'Ticket por PAX': f'USD {ticket_pax}',
            'Gross': gross_str,      
            'Estado final': estado_cliente_str,
            'Costo Total': costo_cluster_str,
            'Cantidad PAX por vouchers': round(pax_cant.mean(), 2),
            'Costo %': f'{costo_pct}%',




        })

    total_gross = sum(gross_numerico)
    for i in range(len(resumen)):
        porcentaje = round(100 * gross_numerico[i] / total_gross, 1)
        resumen[i]['Gross %'] = f'{porcentaje}%'
        costo_i = df_pais.loc[df_subset_pais[df_subset_pais["cluster_nuevo"] == i].index]["Total cost"].sum()
        

    df_resumen = pd.DataFrame(resumen)
    df_resumen.index = [f'Cluster {i}' for i in sorted(df_subset_pais['cluster_nuevo'].unique())]
    return df_resumen.transpose()

# ---------------------- EXPORTACIÓN EXCEL ----------------------



# Columnas seleccionadas para clustering
columnas_cluster = [
    'TIPO_VENTA_FINAL', 'Tiene Menores', 'Canal Venta', 'Anticipacion de compra',
    'Cant. Compras', 'Clasificacion edad', 'Clasificación tipo PAX',
    'DESTINO', 'Gasto promedio por PAX', 'Cantidad PAX por vouchers',
    'R', 'F', 'M', 'Score RFM', 'Tope de cobertura', 'Precio Unitario USD',
    'Clasificación Actividad', 'Mes_Inicio_Viaje', 'Rango de edades',
    'ESTADO', 'BBDD Health'
]


df_subset = df[columnas_cluster].copy()

# Columnas numéricas
numericas = [
    'Anticipacion de compra', 'Cant. Compras', 'Gasto promedio por PAX',
    'Cantidad PAX por vouchers', 'R', 'F', 'M', 'Score RFM', 'Precio Unitario USD'
]

for col in numericas:
    df_subset[col] = df_subset[col].astype(str).str.replace(',', '.', regex=False)
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

df_subset = df_subset.dropna()
numericas_final = [col for col in df_subset.columns if df_subset[col].dtype != 'object']
categoricas_final = [col for col in df_subset.columns if df_subset[col].dtype == 'object']

# Pipeline clustering
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numericas_final),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categoricas_final)
])


resumenes_consolidados = []

for n_clusters in range(1, 4):  # de 1 a 3 inclusive
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
    ])
    pipeline.fit(df_subset)

    df_subset[f'cluster_nuevo_{n_clusters}'] = pipeline.named_steps['kmeans'].labels_ + 1

    # Copia temporal con columna fija para análisis
    df_subset_temp = df_subset.copy()
    df_subset_temp['cluster_nuevo'] = df_subset_temp[f'cluster_nuevo_{n_clusters}']

    # Generar resumen
    resumen = generar_resumen_por_pais(df, df_subset_temp)
    resumen.insert(0, 'Indicador', resumen.index)
    resumen.reset_index(drop=True, inplace=True)

        # Fila adicional con top 3 meses de inicio por cluster
    top_meses = obtener_top_meses_por_cluster(df, df_subset, n_clusters)
    fila_meses = pd.DataFrame([["Top 3 meses de inicio"] + top_meses + [''] * (resumen.shape[1] - 1 - n_clusters)],
                              columns=resumen.columns)



    # Agregar encabezado antes del bloque
    separador = pd.DataFrame([[f'--- CLUSTERS {n_clusters} ---'] + [''] * (resumen.shape[1] - 1)],
                             columns=resumen.columns)
    resumenes_consolidados.append(separador)
    resumenes_consolidados.append(resumen)
    resumenes_consolidados.append(fila_meses)
    resumenes_consolidados.append(pd.DataFrame([[''] * resumen.shape[1]], columns=resumen.columns))  # fila vacía

# Concatenar todo
resumen_final = pd.concat(resumenes_consolidados, ignore_index=True)



resumenes = {}  # inicializá el diccionario vacío, sin llamar la función todavía


# Agregar filas adicionales al Excel
for clave in resumenes:
    df_pais = df if clave == 'TOTAL' else df[df['PAIS_EMISION'] == clave]
    df_subset_pais = df_subset.loc[df_pais.index.intersection(df_subset.index)].copy()

    clusters_ordenados = sorted(df_subset_pais['cluster_nuevo'].unique())
    promedio_pax_por_cluster = []
    top_meses_cluster = []

    for cluster in clusters_ordenados:
        indices_cluster = df_subset_pais[df_subset_pais['cluster_nuevo'] == cluster].index

        # PAX promedio
        pax_vals = df_pais.loc[indices_cluster, 'Cantidad PAX por vouchers'].astype(str).str.replace(',', '.', regex=False).astype(float)
        promedio_pax = round(pax_vals.mean(), 2)
        promedio_pax_por_cluster.append(promedio_pax)
        # Top 3 meses con porcentaje
        meses = df_pais.loc[indices_cluster, 'FECHA_INICIO_VIGENCIA'].dt.month
        conteo = meses.value_counts(normalize=True).round(3).head(4) * 100
        nombres_meses_pct = ', '.join([
            f"{pd.to_datetime(f'2025-{mes}-01').strftime('%b')} ({int(pct)}%)"
            for mes, pct in conteo.items()
        ])
        top_meses_cluster.append(nombres_meses_pct)


    fila_pax = pd.DataFrame([["Cantidad PAX por vouchers"] + promedio_pax_por_cluster],
                            columns=['Indicador'] + [f'Cluster {i}' for i in clusters_ordenados],
                            index=['PAX Promedio'])

    fila_meses = pd.DataFrame([["Top 3 meses de inicio"] + top_meses_cluster],
                              columns=['Indicador'] + [f'Cluster {i}' for i in clusters_ordenados],
                              index=['Meses Inicio'])

    resumenes[clave] = pd.concat([resumenes[clave], fila_pax.set_index('Indicador')])
    resumenes[clave] = pd.concat([resumenes[clave], fila_meses.set_index('Indicador')])

# Guardar en Excel
output_path = 'resumen_clusters_1_a_3.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    resumen_final.to_excel(writer, sheet_name='Clusters 1 a 3', index=False, header=False, startrow=1)
    workbook = writer.book
    worksheet = writer.sheets['Clusters 1 a 3']

    formato_titulo = workbook.add_format({'bold': True, 'align': 'center', 'border': 1, 'bg_color': '#DDEBF7'})
    formato_celda = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})

    # Títulos de la fila 0
    for col_num, value in enumerate(resumen_final.columns):
        worksheet.write(0, col_num, value, formato_titulo)

    worksheet.freeze_panes(1, 1)
    for i in range(len(resumen_final.columns)):
        worksheet.set_column(i, i, 25)

files.download(output_path)
