import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv')  # Carrega o arquivo CSV com os dados médicos

df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)  
# Cria a coluna 'overweight' para identificar sobrepeso com base no IMC (> 25)

df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)  
# Reclassifica colesterol: 0 para normal, 1 para anormal

df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)  
# Reclassifica glicose: 0 para normal, 1 para anormal

def draw_cat_plot():
    df_cat = pd.melt(df, 
                     id_vars=['cardio'],  
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])  
    # Converte as colunas selecionadas para formato longo, mantendo 'cardio' como variável identificadora

    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().reset_index(name='total')  
    # Agrupa e conta as combinações de 'cardio', 'variable' e 'value'

    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig  
    # Cria um gráfico categórico de barras, dividido pela variável 'cardio'

    fig.savefig('catplot.png')  # Salva o gráfico como 'catplot.png'
    return fig  # Retorna a figura

def draw_heat_map():
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]  
    # Filtra os dados para remover outliers e garantir que 'ap_lo' <= 'ap_hi'

    corr = df_heat.corr()  # Calcula a matriz de correlação das variáveis filtradas

    mask = np.triu(np.ones_like(corr, dtype=bool))  
    # Cria uma máscara para ocultar a metade superior da matriz de correlação

    fig, ax = plt.subplots(figsize=(8, 12))  # Define o tamanho da figura

    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, cmap='coolwarm', cbar_kws={'shrink': .5})  
    # Cria o heatmap com anotações das correlações e estilo visual

    fig.savefig('heatmap.png')  # Salva o heatmap como 'heatmap.png'
    return fig  # Retorna a figura