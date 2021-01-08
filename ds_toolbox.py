# Bibliotecas necessárias para as funções
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as met
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


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

def seleciona_colunas_relevantes(modelo, X_treinamento, X_teste, threshold = 0.05):
    # Cria um seletor para selecionar as COLUNAS com importância > threshold
    sfm = SelectFromModel(modelo, threshold)
    
    # Treina o seletor
    sfm.fit(X_treinamento, y_treinamento)

    # Mostra o indice das COLUNAS mais importantes
    print(f'\n********** COLUNAS Relevantes ******')
    print(sfm.get_support(indices=True))

    # Seleciona somente as COLUNAS relevantes
    X_treinamento_I = sfm.transform(X_treinamento)
    X_teste_I = sfm.transform(X_teste)
    return X_treinamento_I, X_teste_I

def mostra_feature_importances(clf, X_treinamento, y_treinamento=None, 
                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):
    '''
    plot feature importances of a tree-based sklearn estimator
    
    Note: X_treinamento and y_treinamento are pandas DataFrames
    
    Note: Scikit-plot is a lovely package but I sometimes have issues
              1. flexibility/extendibility
              2. complicated models/datasets
          But for many situations Scikit-plot is the way to go
          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html
    
    Parameters
    ----------
        clf         (sklearn estimator) if not fitted, this routine will fit it
        
        X_treinamento     (pandas DataFrame)
        
        y_treinamento     (pandas DataFrame)  optional
                                        required only if clf has not already been fitted 
        
        top_n       (int)               Plot the top_n most-important features
                                        Default: 10
                                        
        figsize     ((int,int))         The physical size of the plot
                                        Default: (8,8)
        
        print_table (boolean)           If True, print out the table of feature importances
                                        Default: False
        
    Returns
    -------
        the pandas dataframe with the features and their importance
        
    Author
    ------
        George Fisher
    '''
    
    __name__ = "mostra_feature_importances"
    
    import pandas as pd
    import numpy  as np
    import matplotlib.pyplot as plt
    
    from xgboost.core     import XGBoostError
    from lightgbm.sklearn import LightGBMError
    
    try: 
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_treinamento.values, y_treinamento.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                    format(clf.__class__.__name__))
                
    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_treinamento.values, y_treinamento.values.ravel())
            
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_treinamento.columns
    feat_imp.sort_values(by ='importance', ascending = False, inplace = True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace = True)
    feat_imp = feat_imp.set_index('feature', drop = True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by = 'importance', ascending = False))
        
    return feat_imp


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

def mostra_confusion_matrix(cf, 
                            group_names = None, 
                            categories = 'auto', 
                            count = True, 
                            percent = True, 
                            cbar = True, 
                            xyticks = False, 
                            xyplotlabels = True, 
                            sum_stats = True, 
                            figsize = (8, 8), 
                            cmap = 'Blues'):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

def do_cross_val_score(modelo, X_treinamento, y_treinamento, CV):
    '''
    MELHORAR!!!!
    * Possibilitar que se peça os scores que se quer que calcule
    * Melhorar a documentação

    Realiza o cross validation, retornando um array com os scores de de cada iteração
    e imprimindo a média e o desvio padrão desses scores.
    '''
    a_scores_CV = cross_val_score(modelo, X_treinamento, y_treinamento, cv = CV)
    print(f'Média das Acurácias calculadas pelo CV....: {100*round(a_scores_CV.mean(),4)}')
    print(f'std médio das Acurácias calculadas pelo CV: {100*round(a_scores_CV.std(),4)}')
    return a_scores_CV

def GridSearchOptimizer(modelo, ml_Opt, d_hiperparametros, X_treinamento, y_treinamento, X_teste, y_teste, i_CV, l_colunas):
    ml_GridSearchCV = GridSearchCV(modelo, d_hiperparametros, cv = i_CV, n_jobs = -1, verbose= 10, scoring = 'accuracy')
    start = time()
    ml_GridSearchCV.fit(X_treinamento, y_treinamento)
    tempo_elapsed = time()-start
    print(f"\nGridSearchCV levou {tempo_elapsed:.2f} segundos.")

    # Hiperparâmetros que otimizam a classificação:
    print(f'\nHiperparâmetros otimizados: {ml_GridSearchCV.best_params_}')
    
    if ml_Opt == 'ml_DT2':
        print(f'\nDecisionTreeClassifier *********************************************************************************************************')
        ml_Opt = DecisionTreeClassifier(criterion= ml_GridSearchCV.best_params_['criterion'], 
                                        max_depth= ml_GridSearchCV.best_params_['max_depth'],
                                        max_leaf_nodes= ml_GridSearchCV.best_params_['max_leaf_nodes'],
                                        min_samples_split= ml_GridSearchCV.best_params_['min_samples_leaf'],
                                        min_samples_leaf= ml_GridSearchCV.best_params_['min_samples_split'], 
                                        random_state= i_Seed)
        
    elif ml_Opt == 'ml_RF2':
        print(f'\nRandomForestClassifier *********************************************************************************************************')
        ml_Opt = RandomForestClassifier(bootstrap= ml_GridSearchCV.best_params_['bootstrap'], 
                                        max_depth= ml_GridSearchCV.best_params_['max_depth'],
                                        max_features= ml_GridSearchCV.best_params_['max_features'],
                                        min_samples_leaf= ml_GridSearchCV.best_params_['min_samples_leaf'],
                                        min_samples_split= ml_GridSearchCV.best_params_['min_samples_split'],
                                        n_estimators= ml_GridSearchCV.best_params_['n_estimators'],
                                        random_state= i_Seed)
        
    elif ml_Opt == 'ml_AB2':
        print(f'\nAdaBoostClassifier *********************************************************************************************************')
        ml_Opt = AdaBoostClassifier(algorithm='SAMME.R', 
                                    base_estimator=RandomForestClassifier(bootstrap = False, 
                                                                          max_depth = 10, 
                                                                          max_features = 'auto', 
                                                                          min_samples_leaf = 1, 
                                                                          min_samples_split = 2, 
                                                                          n_estimators = 400), 
                                    learning_rate = ml_GridSearchCV.best_params_['learning_rate'], 
                                    n_estimators = ml_GridSearchCV.best_params_['n_estimators'], 
                                    random_state = i_Seed)
        
    elif ml_Opt == 'ml_GB2':
        print(f'\nGradientBoostingClassifier *********************************************************************************************************')
        ml_Opt = GradientBoostingClassifier(learning_rate = ml_GridSearchCV.best_params_['learning_rate'], 
                                            n_estimators = ml_GridSearchCV.best_params_['n_estimators'], 
                                            max_depth = ml_GridSearchCV.best_params_['max_depth'], 
                                            min_samples_split = ml_GridSearchCV.best_params_['min_samples_split'], 
                                            min_samples_leaf = ml_GridSearchCV.best_params_['min_samples_leaf'], 
                                            max_features = ml_GridSearchCV.best_params_['max_features'])
        
    elif ml_Opt == 'ml_XGB2':
        print(f'\nXGBoostingClassifier *********************************************************************************************************')
        ml_Opt = XGBoostingClassifier(learning_rate= ml_GridSearchCV.best_params_['learning_rate'], 
                                      max_depth= ml_GridSearchCV.best_params_['max_depth'], 
                                      colsample_bytree= ml_GridSearchCV.best_params_['colsample_bytree'], 
                                      subsample= ml_GridSearchCV.best_params_['subsample'], 
                                      gamma= ml_GridSearchCV.best_params_['gamma'], 
                                      min_child_weight= ml_GridSearchCV.best_params_['min_child_weight'])
        
    # Treina novamente usando os hiperparâmetros otimizados...
    ml_Opt.fit(X_treinamento, y_treinamento)

    # Cross-Validation com 10 folds
    print(f'\n********* CROSS-VALIDATION ***********')
    a_scores_CV = funcao_cross_val_score(ml_Opt, X_treinamento, y_treinamento, i_CV)

    # Faz predições com os hiperparâmetros otimizados...
    y_pred = ml_Opt.predict(X_teste)
  
    # Importância das COLUNAS
    print(f'\n********* IMPORTÂNCIA DAS COLUNAS ***********')
    df_importancia_variaveis = pd.DataFrame(zip(l_colunas, ml_Opt.feature_importances_), columns= ['coluna', 'importancia'])
    df_importancia_variaveis = df_importancia_variaveis.sort_values(by= ['importancia'], ascending=False)
    print(df_importancia_variaveis)

    # Matriz de Confusão
    print(f'\n********* CONFUSION MATRIX - PARAMETER TUNNING ***********')
    cf_matrix = confusion_matrix(y_teste, y_pred)
    cf_labels = ['True_Negative', 'False_Positive', 'False_Negative', 'True_Positive']
    cf_categories = ['Zero', 'One']
    mostra_confusion_matrix(cf_matrix, group_names = cf_labels, categories = cf_categories)

    return ml_Opt, ml_GridSearchCV.best_params_

