## Librerías
import time
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, QuantileRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import d2_absolute_error_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, median_absolute_error, max_error

from xgboost import XGBRegressor

import optuna


def calculo_all_metrics_regression(y_true, y_pred):

    d2mae = d2_absolute_error_score(y_true, y_pred)
    # D2 score - MAE coeficiente de determinación basado en la desviación
    # Mide DESVIACIÓN
    # D2MAE: Qué tan bueno es el modelo en comparación a no tener modelo (modelo que siempre predice la mediana de los datos).
    # Resultados: -inf, 1. Mejor resultado es 1. El modelo es igual a decir que todos tienen el valor de la mediana: 0. Resultados negativos: Modelo peor que el promedio, ruido.
    # Penalización homogena a todos los errores (ya sea que tengan alta o baja desviación)
    rmse = root_mean_squared_error(y_true, y_pred)
    # RMSE: Cuánto es la equivocación promedio en las unidades de datos, penalizando altas desviaciones.
    # Penaliza fuertemente desviaciones grandes (porque los errores se elevan al cuadrado, despúes se promedian y finalmente se saca raiz)
    # Es de interes (pero no es central como en MSE) el desempeño del modelo durante la predicción de outliers
    # Si target tiene muchos outliers (que van a generar altos errores), estos pueden distorcionar la métrica (especialmente a MSE)
    # Si RMSE = MAE se tienen que los errores siguen una distribución semejante. Si RMSE > MAE entonces modelo tiene errores de magnitudes (desviaciones) muy variadas.
    mae = mean_absolute_error(y_true, y_pred)
    # MAE: Cuánto es la equivocación promedio en las unidades de datos.
    # No dice si ese error es "bueno" o "malo". Un error de 5 es excelente si los valores van de 0 a 1000, pero es pésimo si van de 0 a 10.
    # Penalización homogena a todos los errores (ya sea que tengan alta o baja desviación)
    # No hay mucho interes en cómo desviaciones muy grandes afectan el desempeño general del modelo
    mape = mean_absolute_percentage_error(y_true, y_pred)
    # MAPE: Indica el error medio en términos relativos. ¿Qué tan grande es el error comparado el valor real?
    # No se ve afectado por diferentes escalas de los datos (algunos pueden estar en miles y otros en millones).
    # Sirve para comparar modelos entrenados en datasets de diferentes escalas (0-1, 0-100)
    # No dice el valor porcentual (porque se divide por 100). Un error de 200% se reporta como 2
    # Penalización mayor cuando error real está en menores escalas (mayores penalizaciones en variables de 0-1 que de 50-100)
    # Prefiere modelos que subestiman (predicen valores más bajos de lo real) porque la penalización porcentual es menor matemáticamente
    r2 = r2_score(y_true, y_pred)
    # Proporción de varianza que variables independientes logran explicar
    # Mide VARIANZA
    # Ajuste del modelo y cuán bien se pueden predecir muestras no vistas utilizando varianza explicada del modelo
    # Resultados -inf, 1. Mejor resultado: 1. Modelo que da a todos el promedio indica resultado = 0. Resultados negativos son modelos ruidosos
    # Parte del supuesto que los errores siguen una distribucón normal
    medianae = median_absolute_error(y_true, y_pred)
    # MEDIAN: Informa el valor de la mediana de la desviación en la distribución de errores.
    maxae = max_error(y_true, y_pred)
    # MAX: Informa la máxima desviación en la distribución de errores.
    return [mae, d2mae, rmse, mape, r2, medianae, maxae]

def performance_statistics_regression(list_performance, tipo):# [[d2mae, rmse, mae, mape, r2, medianae, maxae],[d2mae, rmse, mae, mape, r2, medianae, maxae],...]

    arr = np.array(list_performance)
    resultados = {}

    n = arr.shape[0]  # número de muestras
    for i, metrica in enumerate(metricas):
        valores = arr[:, i]
        mean = np.mean(valores)
        std = np.std(valores, ddof=1)  # desviación estándar muestral
        # error estándar
        sem = stats.sem(valores)  
        # valor crítico t para 95% de confianza
        t_crit = stats.t.ppf(1 - 0.025, df=n-1)
        sample_error = t_crit * sem

        resultados[f"{metrica}_mean_%s"%tipo] = mean
        resultados[f"{metrica}_std_%s"%tipo] = std
        resultados[f"{metrica}_limsup_%s"%tipo] = mean + sample_error
        resultados[f"{metrica}_liminf_%s"%tipo] = mean - sample_error
        resultados[f"{metrica}_samperror_%s"%tipo] = sample_error

    return resultados

def get_regressor(trial, regressor_name):
    is_svc_non_linear = 0

    if regressor_name == "LinearRegression":
        clf = LinearRegression()

    elif regressor_name == "Ridge":
        alpha = trial.suggest_float("ridge_alpha", 1e-1, 1)
        solver = trial.suggest_categorical("ridge_solver", ["auto", "svd", "lsqr", "sparse_cg"])
        clf = Ridge(alpha=alpha, random_state=42, solver=solver)

    elif regressor_name == 'ElasticNet':
        alpha = trial.suggest_float("elastic_alpha", 1e-1, 1)
        l1_ratio = trial.suggest_categorical("elastic_l1ratio", [0, 0.5, 1])
        clf = ElasticNet(random_state=42, alpha=alpha, l1_ratio=l1_ratio)
    
    elif regressor_name == 'QuantileRegressor':
        alpha = trial.suggest_float("quantile_alpha", 1e-1, 1)
        clf = QuantileRegressor(quantile=0.5, alpha=alpha, solver='highs')

    elif regressor_name == "SVR":
        C = trial.suggest_float("svr_C", 1e-3, 100, log=True)
        epsilon = trial.suggest_float("svr_epsilon", 1e-3, 100, log=True)
        kernel = trial.suggest_categorical("svr_kernel", ["linear", "rbf", "poly"])
        degree = trial.suggest_int("svr_degree", 2, 5) if kernel == "poly" else 3
        clf = SVR(C=C, kernel=kernel, degree=degree, epsilon=epsilon)
        if kernel != "linear": is_svc_non_linear = 1

    elif regressor_name == "RandomForestRegressor":
        n_estimators = trial.suggest_int("rf_n_estimators", 10, 200)
        max_depth = trial.suggest_int("rf_max_depth", 2, 8)
        min_samples_split = trial.suggest_int("rf_min_sample", 2, 8)
        clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    
    elif regressor_name == "ExtraTreesRegressor":
        n_estimators = trial.suggest_int("et_n_estimators", 50, 500)
        max_depth = trial.suggest_int("et_max_depth", 2, 20)
        min_samples_split = trial.suggest_int("et_min_samples_split", 2, 8)
        clf = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, criterion='friedman_mse', random_state=42)

    elif regressor_name == "BaggingRegressor":
        n_estimators = trial.suggest_int("br_n_estimators", 50, 500)
        clf = BaggingRegressor(n_estimators=n_estimators, random_state=42)

    elif regressor_name == "StackingRegressor":
        estimators = [('lr', Ridge()),('svr', SVR())]
        base_estimator = trial.suggest_categorical("str_final_estimator", ["RandomForestRegressor", "ExtraTreesRegressor", "Ridge", "LinearRegression"])
        if base_estimator == "RandomForestRegressor":
            final_estimator = RandomForestRegressor(random_state=42)
        elif base_estimator == "ExtraTreesRegressor":
            final_estimator = ExtraTreesRegressor(random_state=42)
        elif base_estimator == "Ridge":
            final_estimator = Ridge(random_state=42)
        elif base_estimator == "LinearRegression":
            final_estimator = LinearRegression()
        clf = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=None)

    elif regressor_name == "GaussianProcessRegressor":
        alpha = trial.suggest_float("gaussian_alpha", 1e-10, 1e-1)
        clf = GaussianProcessRegressor(alpha=alpha, random_state=42)

    elif regressor_name == "KNeighborsRegressor":
        n_neighbors = trial.suggest_int("knn_n_neighbors", 1, 20)
        weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("knn_metric", ["euclidean","manhattan","minkowski"])
        clf = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
    
    elif regressor_name == "DecisionTreeRegressor":
        max_depth = trial.suggest_int("dt_max_depth", 2, 20)
        min_samples_split = trial.suggest_int("dt_min_samples_split", 2, 20)
        clf = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    
    elif regressor_name == "XGBRegressor":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 500)
        max_depth = trial.suggest_int("xgb_max_depth", 2, 10)
        learning_rate = trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True)
        subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0, step=0.1)
        colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0, step=0.1)
        reg_alpha = trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True)
        booster = trial.suggest_categorical("xgb_booster", ["gbtree", "dart", "gblinear"])
        clf = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           subsample=subsample, colsample_bytree=colsample_bytree,
                           reg_alpha=reg_alpha, reg_lambda=reg_lambda, booster=booster,
                           random_state=42)
    else:
        raise ValueError(f"Regressor {regressor_name} no reconocido.")

    return clf, is_svc_non_linear

def objective_particiones_regresion(trial, X_bd, name_optimizacion, name_file_particion):

    start_time = time.time()  # ⏱ Inicia contador de tiempo

    # 1. Seleccionar tipo de clasificador
    regressor_name = trial.suggest_categorical("regressor", ["LinearRegression", "Ridge", "ElasticNet", "QuantileRegressor",
                                                             "SVR",
                                                             "RandomForestRegressor", "ExtraTreesRegressor", "BaggingRegressor", "StackingRegressor",
                                                             "GaussianProcessRegressor",
                                                             "KNeighborsRegressor",
                                                             "DecisionTreeRegressor",
                                                             "XGBRegressor"])
    
    # 2. Normalización
    use_scaler = trial.suggest_categorical("use_scaler", [True, False])

    # 3. Definir el regresor y sus hiperparámetros
    clf, is_svc_non_linear = get_regressor(trial, regressor_name)
    
    # 4. Construir pipeline
    steps = []
    steps.append(("imputer", SimpleImputer(strategy="mean"))) #### Por medidas de coherencia con WordEmbeddings
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", clf))
    pipeline = Pipeline(steps)

    # 5. Lectura de combinations (particiones de BD)
    df_particiones = lectura_combination(name_file_particion=name_file_particion)
    combinations = df_particiones['COMBINATION'].unique()

    performance_validation = []
    performance_testeo = []

    # 6. Validación particiones
    for step, combination in enumerate(combinations, start=1):

        # 6a. Definir muestras que pertencen a cada SET
        df_comb = df_particiones[df_particiones['COMBINATION'] == combination]

        train_cod = df_comb[df_comb['SET'] == 'TRAIN']['COD'].values
        X_train = X_bd.loc[X_bd['Cod'].isin(train_cod), [c for c in X_bd.columns if c not in ['Cod', 'Target']]]
        y_train = X_bd.loc[X_bd['Cod'].isin(train_cod), ['Target']]

        val_cod = df_comb[df_comb['SET'] == 'VALIDATION']['COD'].values
        X_val = X_bd.loc[X_bd['Cod'].isin(val_cod), [c for c in X_bd.columns if c not in ['Cod', 'Target']]]
        y_val = X_bd.loc[X_bd['Cod'].isin(val_cod), ['Target']]

        test_cod = df_comb[df_comb['SET'] == 'TEST']['COD'].values
        X_test = X_bd.loc[X_bd['Cod'].isin(test_cod), [c for c in X_bd.columns if c not in ['Cod', 'Target']]]
        y_test = X_bd.loc[X_bd['Cod'].isin(test_cod), ['Target']]
        
        # 6b. Entrenar modelo y hacer prediccion (val, test)
        pipeline.fit(X_train, y_train)
        pred_val = pipeline.predict(X_val)
        pred_test = pipeline.predict(X_test)
        
        # 6c. Calcular performance
        performance_validation.append(calculo_all_metrics_regression(y_true=y_val, y_pred=pred_val))
        performance_testeo.append(calculo_all_metrics_regression(y_true=y_test, y_pred=pred_test))

        # 6d. Evaluar si vale la pena prune el trial dependiendo de mse parcial: promedio de los folds ya corridos
        acc_partial = np.mean([m[0] for m in performance_validation])

        # Reportar a Optuna
        trial.report(acc_partial, step)

        # Si el pruner decide cortar aquí
        if trial.should_prune():
            trial.set_user_attr("termination_reason", "pruned")
            raise optuna.TrialPruned()

        # ⏱ Última verificación
        if time.time() - start_time > 480: # Lim 8 minutes
            trial.set_user_attr("termination_reason", "timeout")
            raise optuna.TrialPruned()
    
    # 6e. Calcular estadísticas de performance
    dict_valitacion_performance = performance_statistics_regression(list_performance=performance_validation, tipo='val')
    dict_testeo_performance = performance_statistics_regression(list_performance=performance_testeo, tipo='test')

    trial.set_user_attr("termination_reason", "completed")

    # 6f. Guardar todas las métricas en el trial
    for k, v in dict_valitacion_performance.items():
        trial.set_user_attr(k, v)
    for k, v in dict_testeo_performance.items():
        trial.set_user_attr(k, v)
    # 7. Optimizar sobre mse_mean de conjunto de val
    return dict_valitacion_performance[name_optimizacion]

def objective_particiones_rfe_regression(trial, X_bd, name_optimizacion, name_file_particion):

    start_time = time.time()  # ⏱ Inicia contador de tiempo

    # 1. Seleccionar tipo de clasificador
    regressor_name = trial.suggest_categorical("regressor", ["LinearRegression", "Ridge", "ElasticNet", "QuantileRegressor",
                                                             "SVR",
                                                             "RandomForestRegressor", "ExtraTreesRegressor", "BaggingRegressor", "StackingRegressor",
                                                             "GaussianProcessRegressor",
                                                             "KNeighborsRegressor",
                                                             "DecisionTreeRegressor",
                                                             "XGBRegressor"])
    
    # 2. Normalización
    use_scaler = trial.suggest_categorical("use_scaler", [True, False])

    # 3. Compatibilidad RFE según clasificador
    max_features = min(X_bd.shape[1] - 3, 100)
    n_features_to_select = trial.suggest_int("rfe_n_features", 2, max_features)
    
    # 4. Definir el regresor y sus hiperparámetros
    clf, is_svc_non_linear = get_regressor(trial, regressor_name)
    
    # Lista de modelos que NO funcionan con RFE directamente
    modelos_no_compatibles = ["KNeighborsRegressor", "GaussianProcessRegressor", "BaggingRegressor", "StackingRegressor"]
    
    if (regressor_name in modelos_no_compatibles) or (is_svc_non_linear == 1):
        # 1. Definimos un selector de respaldo (Proxy)
        rfe_proxy_type = trial.suggest_categorical("rfe_proxy_type", ["ExtraTrees", "LinearSVC"])
    
        if rfe_proxy_type == "ExtraTrees":
            rfe_n_est = trial.suggest_int("rfe_proxy_et_n_estimators", 20, 100, step=10)
            rfe_depth = trial.suggest_int("rfe_proxy_et_max_depth", 3, 10)
            rfe_min_samples = trial.suggest_int("et_min_samples_split", 2, 8)
            
            rfe_clf = ExtraTreesRegressor(n_estimators=rfe_n_est, max_depth=rfe_depth, min_samples_split=rfe_min_samples, criterion='friedman_mse', random_state=42)
            rfe_params = {
                "mode": "Proxy", 
                "type": "ExtraTrees", 
                "n_estimators": rfe_n_est, 
                "max_depth": rfe_depth,
                "min_samples_split": rfe_min_samples}
            
        else:
            rfe_c = trial.suggest_float("rfe_proxy_svc_C", 1e-3, 10.0, log=True)
            rfe_epsilon = trial.suggest_float("svr_epsilon", 1e-3, 100, log=True)
            
            rfe_clf = SVR(C=rfe_c, kernel="linear", epsilon=rfe_epsilon)
        
            rfe_params = {
                "mode": "Proxy", 
                "type": "LinearSVC", 
                "C": rfe_c,
                "epsilon": rfe_epsilon}
            
    else:
        # 2. El modelo original ES compatible
        rfe_clf = clf
        rfe_params = {"mode": "Original", "type":regressor_name}

    # Guardamos toda la configuración en los atributos del trial
    trial.set_user_attr("rfe_clf", rfe_params)
    
    # 5. RFE (fuera del ciclo)
    X_bd_copy = X_bd.copy()
    y_op = X_bd_copy['Target'].values
    X_bd_copy = X_bd_copy[[c for c in X_bd_copy.columns if c not in ['Cod', 'Target']]]
    X_proc = X_bd_copy.values  # usar directamente los datos como array numpy
    imputer = SimpleImputer(strategy="mean") ###### Por medidas de coherencia con WordEmbeddings
    X_proc = imputer.fit_transform(X_proc)

    if use_scaler:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    # Selección de características con RFE
    selector = RFE(estimator=rfe_clf, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X_proc, y_op)
    selected_mask = selector.support_

    # Guardar features seleccionadas
    selected_features = X_bd_copy.columns[selected_mask].tolist()
    trial.set_user_attr("rfe_selected_features", selected_features)
    trial.set_user_attr("rfe_n_features", n_features_to_select)

    # Reconstruir DataFrame con las columnas seleccionadas
    selected_columns = X_bd_copy.columns[selected_mask]
    X_selected = pd.DataFrame(X_proc[:, selected_mask], columns=selected_columns, index=X_bd.index)

    # Añadir Cod y Target garantizando orden original
    X_proc_final = pd.concat([X_bd[['Cod', 'Target']], X_selected], axis=1)

    # 6. Organizar pipeline final
    steps = []
    steps.append(("regressor", clf))
    pipeline = Pipeline(steps)

    # 7. Lectura de combinations (particiones de BD)
    df_particiones = lectura_combination(name_file_particion=name_file_particion)
    combinations = df_particiones['COMBINATION'].unique()

    performance_validation = []
    performance_testeo = []

    # 8. Validación Particiones
    for step, combination in enumerate(combinations, start=1):

        # 8a. Definir muestras que pertencen a cada SET
        df_comb = df_particiones[df_particiones['COMBINATION'] == combination]

        train_cod = df_comb[df_comb['SET'] == 'TRAIN']['COD'].values
        X_train = X_proc_final.loc[X_proc_final['Cod'].isin(train_cod), [c for c in X_proc_final.columns if c not in ['Cod', 'Target']]]
        y_train = X_proc_final.loc[X_proc_final['Cod'].isin(train_cod), ['Target']]

        val_cod = df_comb[df_comb['SET'] == 'VALIDATION']['COD'].values
        X_val = X_proc_final.loc[X_proc_final['Cod'].isin(val_cod), [c for c in X_proc_final.columns if c not in ['Cod', 'Target']]]
        y_val = X_proc_final.loc[X_proc_final['Cod'].isin(val_cod), ['Target']]

        test_cod = df_comb[df_comb['SET'] == 'TEST']['COD'].values
        X_test = X_proc_final.loc[X_proc_final['Cod'].isin(test_cod), [c for c in X_proc_final.columns if c not in ['Cod', 'Target']]]
        y_test = X_proc_final.loc[X_proc_final['Cod'].isin(test_cod), ['Target']]
        
        # 8b. Entrenar modelo y hacer prediccion (val, test)
        pipeline.fit(X_train, y_train)
        pred_val = pipeline.predict(X_val)
        pred_test = pipeline.predict(X_test)
        
        # 8c. Calcular performance
        performance_validation.append(calculo_all_metrics_regression(y_true=y_val, y_pred=pred_val))
        performance_testeo.append(calculo_all_metrics_regression(y_true=y_test, y_pred=pred_test))

        # 8d. Evaluar si vale la pena prune el trial
        acc_partial = np.mean([m[1] for m in performance_validation]) # En el artículo fue contra acc ::: np.mean([m[0] for m in performance_validation])

        # Reportar a Optuna
        trial.report(acc_partial, step)

        # Si el pruner decide cortar aquí
        if trial.should_prune():
            trial.set_user_attr("termination_reason", "pruned")
            raise optuna.TrialPruned()

        # ⏱ Última verificación
        if time.time() - start_time > 480: # Lim 8 minutes
            trial.set_user_attr("termination_reason", "timeout")
            raise optuna.TrialPruned()
    
    # 8e. Calcular estadísticas de performance
    dict_valitacion_performance = performance_statistics_regression(list_performance=performance_validation, tipo='val')
    dict_testeo_performance = performance_statistics_regression(list_performance=performance_testeo, tipo='test')

    trial.set_user_attr("termination_reason", "completed")

    # 8f. Guardar todas las métricas en el trial
    for k, v in dict_valitacion_performance.items():
        trial.set_user_attr(k, v)
    for k, v in dict_testeo_performance.items():
        trial.set_user_attr(k, v)

    # 9. Optimizar
    return dict_valitacion_performance[name_optimizacion]


name_optimizacion = 'mae_mean_val'
#%% Optimización hiperparámetros - Sin transformación
# Nombre y ubicación del archivo .db donde se guardarán los trials
name_experiment = '%s_v0'%(name_general)
db_path = ruta_prediccion + "/%s/optuna_trials_features_%s.db"%(name_folder, name_experiment)
storage_url = f"sqlite:///{db_path}"
study_name = "optimizacion_%s"%(name_experiment)

# Crea o carga el estudio
study_v0 = optuna.create_study(
    sampler=optuna.samplers.TPESampler(),
    study_name=study_name,
    storage=storage_url,
    direction="minimize",#"maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=15),
    load_if_exists=True  # ← importante para warmstart
    )

if name_window in ['w40'] and c in ['TOTAL_zscore']:
    n_trials = 1
else:
    n_trials = 300

# Ejecutar la optimización
study_v0.optimize(partial(fpredpart.objective_particiones_regresion, X_bd=df_op, name_optimizacion=name_optimizacion,
                            name_file_particion=name_file_particion), n_trials=n_trials)

# Resultados
best_trial = study_v0.best_trial
attr_name = name_optimizacion#"d2mae_mean_val"

print("Mejores parámetros - validation:")
print(study_v0.best_params)
print(f"Mejor metric mean - validation: {study_v0.best_value:.4f}")
print(f"{attr_name} de mejor  - validation: {best_trial.user_attrs[attr_name]:.4f}")

# Buscar el trial con el mejor valor de ese atributo
attr_name = "mae_mean_test"#"d2mae_mean_test"
best_trial_attr = max(
    study_v0.trials,
    key=lambda t: t.user_attrs.get(attr_name, float("-inf")))

# Publicar el valor y los parámetros asociados
print("Mejores parámetros - testeo:")
print(best_trial_attr.params)
print(f"Mejor {attr_name}: {best_trial_attr.user_attrs[attr_name]:.4f}")

df_v0 = study_v0.trials_dataframe()
df_v0 = fpredpart.procesar_db_regression(df_op=df_v0, name_optimizacion = name_optimizacion, order_df = order_df, list_rev=[])
df_v0.to_excel(ruta_prediccion + "/%s/optuna_trials_features_%s.xlsx"%(name_folder, name_experiment), index=False)

# %% Optimización hiperparámetros - RFE
# Nombre y ubicación del archivo .db donde se guardarán los trials
name_experiment = '%s_v2'%(name_general)
db_path = ruta_prediccion + "/%s/optuna_trials_features_%s.db"%(name_folder, name_experiment)
storage_url = f"sqlite:///{db_path}"
study_name = "optimizacion_%s"%(name_experiment)

# Crea o carga el estudio
study_v2 = optuna.create_study(
    sampler=optuna.samplers.TPESampler(),
    study_name=study_name,
    storage=storage_url,
    direction="minimize",#"maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=15),
    ### It stops unpromising trials early based on the intermediate results compared against the median of previous completed trials
    ### n_startup_trials (int) – Pruning is disabled until the given number of trials finish in the same study
    ### n_warmup_steps (int) – Pruning is disabled until the trial exceeds the given number of step. Note that this feature assumes that step starts at zero.
    load_if_exists=True  # ← importante para warmstart
)

# if name_window in ['w170']:
#     n_trials = 276
# else:
n_trials = 300

# Ejecutar la optimización
study_v2.optimize(partial(fpredpart.objective_particiones_rfe_regression, X_bd=df_op, name_optimizacion=name_optimizacion,
                            name_file_particion = name_file_particion), n_trials=n_trials)
# Resultados
best_trial = study_v2.best_trial
attr_name = name_optimizacion

print("Mejores parámetros - validation:")
print(study_v2.best_params)
print("No. de features en RFE:")
print(study_v2.best_trial.user_attrs['rfe_n_features'])
print(study_v2.best_trial.user_attrs['rfe_selected_features'])
print(f"Mejor metric mean - validation: {study_v2.best_value:.4f}")
print(f"{attr_name} de mejor  - validation: {best_trial.user_attrs[attr_name]:.4f}")

# Buscar el trial con el mejor valor de ese atributo
attr_name = "mae_mean_test"#"d2mae_mean_test"
best_trial_attr = max(
    study_v2.trials,
    key=lambda t: t.user_attrs.get(attr_name, float("-inf")))

# Publicar el valor y los parámetros asociados
print("Mejores parámetros - testeo:")
print(best_trial_attr.params)
print(f"Mejor {attr_name}: {best_trial_attr.user_attrs[attr_name]:.4f}")

df_v2 = study_v2.trials_dataframe()
df_v2 = fpredpart.procesar_db_regression(df_op=df_v2, name_optimizacion = name_optimizacion, order_df = order_df,
                                            list_rev=['rfe_n_features', 'rfe_selected_features', 'rfe_clf'])
df_v2.to_excel(ruta_prediccion + "/%s/optuna_trials_features_%s.xlsx"%(name_folder, name_experiment), index=False)