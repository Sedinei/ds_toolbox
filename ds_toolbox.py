# Bibliotecas necessárias para as funções
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as met

# Funções para análises do dataset
def count_missing_values(df):
    '''
    Função para informar a situação do dataset com relação a quantidade de valores faltantes (NaN).
    Parâmetro:
        - df: DataFrame do Pandas.
    Retorno:
        - None
        - Imprime um sumário com as seguintes informações sobre o dataset:
            * total de linhas do dataset;
            * total de linhas que contém ao menos um NaN (percentual em relação ao total de linhas do dataset);
            * quantidade de NaN por feature (percentual em relação ao total de linhas do dataset).
    '''
    total = df.shape[0]
    total_com_nan = total - df.dropna().shape[0]
    percent = round((total_com_nan/total)*100, 3)
    print(f'Número total de linhas: {total}')
    print(f'Total de linhas que tem ao menos algum dado faltante: {total_com_nan} ({percent}%)')
    print('Quantidade de dados faltantes por feature:')
    df_faltantes = pd.DataFrame(total - df.count(), columns=['missing'])
    df_faltantes['%'] = ((df_faltantes['missing'] / total) * 100).round(3)
    df_faltantes = df_faltantes
    print(df_faltantes)


def show_correlations(df, target, min_corr=0.01, style='dark'):
    '''
    Mostra um gráfico de correlações entre as features de um DataFrame e a lista das correlações entre a
    variável alvo e as features na ordem crescente dos valores de correlação absoluta.
    Parâmetros:
        - df: DataFrame do Pandas;
        - target: string com o nome da variável alvo;
        - min_corr: valor mínimo de correlação entre a feature e a variável (default=0.01)
        - style: estilo do Seaborn a ser aplicado (default='dark')
    Retorno:
        - lista das features cuja correlação com 'target' é menor que 'min_corr';
        - Mostra um gráfico do tipo heatmap do Seaborn com as correlações entre os campos do DataFrame 'df';
        - Mostra features em ordem crescente dos valores de correlação absoluta com 'target'.
    '''
    sns.set(style=style, palette='coolwarm')
    fig, ax = plt.subplots(1, 1, figsize=(19,15))
    ax.set_title('Correlation Matrix', fontsize=16)
    #sns.heatmap(df.corr(), center=0, cmap='vlag', ax=ax) Essa linha deixa espelhado o gráfico
    sns.heatmap(df.corr().abs().where(np.tril(np.ones(df.corr().shape), k = 0).astype(np.bool)), center=0, cmap='vlag', ax=ax)
    plt.show()
    print('-'*200)
    print(f'Lista ordenada das correlações das features com {target}')
    correlations = df.corr().abs()[target].sort_values()
    print(correlations)
    return list((correlations[correlations < min_corr]).index)


# Funções para feature engineering
def transf_feat_bin(df, features, values_1):
    '''
    Função para criar dummies de variáveis binárias.
    Parâmetros:
        - df: DataFrame do Pandas;
        - features: lista com os nomes das features que serão transformadas em valores 1 e 0;
        - values_1: lista do mesmo tamanho de 'features' que contém os valores que serão considerados 1.
    Retorno:
        - None
        - O DataFrame 'df' é alterado 'inplace' sendo que as colunas 'features' são convertidas em dummies, nas quais
          os valores indicados em 'values_1' serão convertidos para 1 e o outro valor da feture será convertido
          para 0.
    '''
    for feature, value_1 in zip(features, values_1):
        df[feature] = df[feature].map(lambda x: 1 if x==value_1 else 0)


def tranf_feat_func(df, features, funcao):
    '''
    Função que transforma os valores de uma relação de features de um DataFrame conforme a função passada.
    Parâmetros:
        - df: DataFrame do Pandas;
        - features: lista com os nomes das features que serão transformadas;
        - funcao: função que será aplicada em todas as features para a transformação.
    Retorno:
        - None
        - O DataFrame 'df' é alterado 'inplace' sendo que as colunas 'features' tem seus valores alterados
          pela aplicação da 'funcao' sobre eles.
    '''
    for feature in features:
        df[feature] = df[feature].map(funcao)


# Funções para avaliação de modelos
def gridsearchcv_valiation(gs_fited, better='max'):
    '''
    Função para mostrar de forma mais adequada os resultados das iterações de um GriedSearchCV do sklearn.
    Parâmetro:
        - gs_fited: objeto GriedSearch após o fit.
        - better: string que indica se o score é melhor quanto maior ('max') ou menor ('min') (default='max')
    Retorno:
        - None
        - Mostra, para cada iteração, em ordem decrescente do valor da métrica de avaliação, os parâmetros
          que foram utilizados. Indica, com '****', os parâmetros que obtiveram os melhores resultados.
    '''
    gs_keys = pd.Series(gs_fited.cv_results_.keys())
    num_splits = gs_keys.str.count(r'^split').sum()
    lst_parameters = gs_fited.cv_results_['params']
    for i in range(num_splits):
        print(f'Scores do split {i}:')
        scores = pd.Series(gs_fited.cv_results_[f'split{i}_test_score'])
        if better == 'max': best_score = scores.max()
        else: best_score = scores.min()
        print(f'Melhor score: {best_score}')
        df_scores = pd.DataFrame({'scores': scores, 'parameters': lst_parameters})
        df_scores.sort_values(by='scores', ascending=False, inplace=True, ignore_index=True)
        df_scores['best'] = ''
        df_scores.loc[df_scores['scores']==best_score, 'best'] = '****'
        print(df_scores)
        print('-'*150)


def show_confusion_matrix(y_true, y_pred):
    '''
    Mostra a visualização da matriz de confusão para uma relação de valores reais da variável alvo e dos
    valores preditos por algum modelo de classificação.
    Parâmetros:
        - y_true: Series do Pandas com os valores reais da variável alvo
        - y_pred: Series do Pandas com os valores preditos da variável alvo
    Retorno:
        - None
        - Mostra o gráfico da matriz de confusão.
    '''
    cf_mat = met.confusion_matrix(y_true, y_pred)
    columns = [f'Predict {i}' for i in range(cf_mat.shape[0])]
    rows = [f'True {i}' for i in range(cf_mat.shape[0])]
    cf_mat_graf = sns.heatmap(cf_mat, annot=True, fmt='g', cmap='Blues', xticklabels=columns, yticklabels=rows)
    cf_mat_graf.set_title('Confusion Matrix')
    plt.show()