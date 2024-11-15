import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import chardet
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold
from collections import Counter
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.python.keras.optimizers import rmsprop_v2
from functools import partial

class ClasificadorDepresion:
    def __init__(self, archivo, codificacion:str = 'utf-8', tecnica_datos=0, reporte="reporte.txt"):
        self.archivo = archivo
        self.ok = True
        self.codificacion = codificacion
        self.tecnica_datos = tecnica_datos
        self.reporte = reporte
        try:
            self.df, self.X_train, self.X_test, self.y_train, self.y_test = self.crearConjuntos()
        except:
            self.ok = False
        self.resultado_dict = {}
        self.modelos = {}

    def crearConjuntos(self):
        file = open(self.reporte, "a")
        file.write("###############################################################################\n")
        file.write("## Conjunto: {} ##\n".format(self.archivo))
        df = pd.read_csv(self.archivo, encoding=self.codificacion)
        # Seleccionar las características y etiquetas
        X = df.drop(["Class"], axis=1)
        y = df["Class"]
        file.write("Muestras {}\n".format(Counter(y)))
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            # Normalizar los datos
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)    
            file.write("###############################################################################\n")
            file.close()    
            return df, X_train, X_test, y_train, y_test
        except Exception as e:
            raise ValueError("no se pudo crear el dataset") from e
    
    def clasificadorArboles(self):
        model = RandomForestClassifier()
        #'max_depth': None, 
        # 'min_samples_leaf': 4, 
        # 'min_samples_split': 10, 
        # 'n_estimators': 10,
        # 'random_state' : 11
        parametros = {
            'n_estimators': [200],          # Número de árboles en el bosque
            'max_depth': [None],             # Profundidad máxima de los árboles
            'min_samples_split': [2],             # Número mínimo de muestras requeridas para dividir un nodo interno
            'min_samples_leaf': [1],               # Número mínimo de muestras requeridas para estar en un nodo hoja
            'random_state': [11]                         # Semilla para la reproducibilidad
        }
        # Configurar GridSearchCV
        grid = GridSearchCV(model, parametros, scoring='accuracy', cv=3)
        grid_resultado = grid.fit(self.X_train, self.y_train)
        # Obtener el modelo con los mejores hiperparámetros
        mejor_modelo = grid_resultado.best_estimator_
        feature_importance = mejor_modelo.feature_importances_
        # Realiza predicciones en un conjunto de prueba
        y_pred = mejor_modelo.predict(self.X_test)
        # Calcula la precisión del modelo en el conjunto de prueba.
        accuracy = accuracy_score(self.y_test, y_pred)
        # Calcula la curva precisión-recuperación y su área (PR-AUC)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
        pr_auc = auc(recall, precision)
        # Imprime un informe detallado con otras métricas.
        report = classification_report(self.y_test, y_pred, zero_division=1)
        # Muestra la matriz de confusión.
        confusion = confusion_matrix(self.y_test, y_pred)
        self.imprimirReporte(
            "Mejores hiperparámetros:\n{}\n".format(grid_resultado.best_params_),
            "Importancia de las caracteristicas:\n {}\n".format(str(feature_importance)),
            "PR-AUC: {}\n".format(pr_auc),
            accuracy,
            report,
            confusion,
            mejor_modelo,
            "RF en {}".format(self.archivo))
        return mejor_modelo

    def clasificadorSVM(self):
        # Especificar los pesos de clase
        todo = len(self.df)
        class_weights = {
            0 : (self.df['Class'].value_counts()[0]) / todo,
            1 : (self.df['Class'].value_counts()[1]) / todo,
        }
        
        parameters = {
            'C': [0.1],
            'degree': [4],
            'gamma': [0.1],
            'coef0': [1]
        }

        classifier = svm.SVC(kernel='poly', class_weight=class_weights)
        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=5)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        accuracy = best_model.score(self.X_test, self.y_test)
        # Calcula la curva precisión-recuperación y su área (PR-AUC)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
        pr_auc = auc(recall, precision)
        # Generate the classification report
        report = classification_report(self.y_test, y_pred, zero_division=1)
        # Generate the confusion matrix
        confusion = confusion_matrix(self.y_test, y_pred)
        self.imprimirReporte(
            "Mejores hiperparámetros:\n{}\n".format(grid_search.best_params_),
            "",
            "PR-AUC: {}\n".format(pr_auc),
            accuracy,
            report,
            confusion,
            best_model,
            "SVM en {}".format(self.archivo))
        
    def clasificadorGradientBoosting(self):
        # Definir parámetros para búsqueda en cuadrícula
        parameters = {
            'learning_rate': [0.1],
            'n_estimators': [200],
            'max_depth': [3],
            'min_samples_split': [3]
        }

        # Inicializar el clasificador GradientBoostingClassifier
        classifier = GradientBoostingClassifier(random_state=42)
        # Realizar búsqueda en cuadrícula para encontrar los mejores hiperparámetros
        grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=5)
        grid_search.fit(self.X_train, self.y_train)
        # Evaluar el modelo con los mejores hiperparámetros en el conjunto de prueba
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        # Calcula la curva precisión-recuperación y su área (PR-AUC)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
        pr_auc = auc(recall, precision)
        # Imprime un informe detallado con otras métricas.
        report = classification_report(self.y_test, y_pred, zero_division=1)
        # Muestra la matriz de confusión.
        confusion = confusion_matrix(self.y_test, y_pred)

        self.imprimirReporte(
            "Mejores hiperparámetros:\n{}\n".format(grid_search.best_params_),
            "",
            "PR-AUC: {}\n".format(pr_auc),
            accuracy,
            report,
            confusion,
            best_model,
            "GradientBoosting en {}".format(self.archivo))

    def modeloNN(
            self,
            units_capa1=64,
            units_capa2=32,
            units_capa3=16,
            units_capa4=8,
            optimizer='rmsprop'
        ):
        model = Sequential()
        model.add(Dense(units_capa1, input_dim=self.X_train.shape[1], activation='relu', kernel_initializer="glorot_uniform"))
        model.add(Dropout(0.25))
        model.add(Dense(units_capa2, activation='relu', kernel_initializer="glorot_uniform"))
        model.add(Dropout(0.25))
        model.add(Dense(units_capa3, activation='relu', kernel_initializer="glorot_uniform"))
        model.add(Dropout(0.25))
        model.add(Dense(units_capa4, activation='relu', kernel_initializer="glorot_uniform"))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))

        # Configurar el optimizador con tasa de aprendizaje específica
        optimizer_instance = rmsprop_v2(learning_rate=0.001) if optimizer == 'rmsprop' else optimizer
        model.compile(optimizer=optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def clasificadorRedNeuronal(self):
        # Configurar KerasClassifier usando self.modeloNN
        modelo = KerasClassifier(model=partial(self.modeloNN, optimizer='rmsprop'), verbose=0)

        # Definir grid de hiperparámetros
        parametros = {
            'epochs': [30],
            'batch_size': [16]
        }

        grid = GridSearchCV(
            estimator=modelo, 
            param_grid=parametros, 
            scoring='accuracy', 
            cv=3
        )
        grid_resultado = grid.fit(self.X_train, self.y_train)

        # Obtener el modelo con los mejores hiperparámetros
        mejor_modelo = grid_resultado.best_estimator_
        # Hacer predicciones con el conjunto de prueba
        y_pred = mejor_modelo.predict(self.X_test)
        # Evaluar el modelo en el conjunto de prueba
        accuracy = grid_resultado.best_score_
        # Calcula la curva precisión-recuperación y su área (PR-AUC)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
        pr_auc = auc(recall, precision)
        # Reporte de clasificación
        report = classification_report(self.y_test, y_pred, zero_division=1)
        # Matriz de Confusión
        confusion = confusion_matrix(self.y_test, y_pred)

        self.imprimirReporte(
            "Mejores hiperparámetros:\n{}\n".format(grid_resultado.best_params_),
            "",
            "PR-AUC: {}\n".format(pr_auc),
            accuracy,
            report,
            confusion,
            mejor_modelo,
            "MLP en {}".format(self.archivo))

    def imprimirReporte(self, extra, extra2, extra3, accuracy, classification_report, confusion_matrix, modelo, titulo= "Reporte de resultados"):
        fecha_y_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        file = open(self.reporte, "a")  
        file.write("----- Inicio del reporte {} -----\n".format(fecha_y_hora))
        file.write("---------------------\n")
        file.write("{}\n".format(titulo))
        file.write("---------------------\n")
        file.write(extra)
        file.write("\n")
        file.write(extra2)
        file.write("\n")
        file.write(extra3)
        file.write("\n")
        file.write("Precisión: {}\n".format(accuracy))
        file.write("\nReporte de clasificación\n")
        file.write(classification_report)
        file.write("\nMatriz de confusión:\n")
        file.write(str(confusion_matrix))
        file.write("Validación cruzada:\n")
        file.write(self.validaciónCruzada(modelo))
        file.write("\n----- Fin del reporte -----\n\n")
        file.close()

        dataset = self.archivo
        algoritmo = titulo.split(" ")[0]
        tecnica = "ninguna"

        self.resultado_dict["{}_{}_{}_{}".format(fecha_y_hora,dataset,tecnica,algoritmo)] = classification_report

    def validaciónCruzada(self, model):
        # Configura la validación cruzada de 5 pliegues
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # Realiza la validación cruzada y obtén los puntajes de precisión
        scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='accuracy')
        mensaje = "Puntajes de precisión para cada pliegue:\n{}\n\nPrecisión media{}\nDesviación estándar de la precisión:{}\n".format(scores,scores.mean(),scores.std())
        return mensaje

    # Código a ejecutar
    def main(self):
        if(self.ok):
            with open(self.archivo, 'rb') as f:
                result = chardet.detect(f.read())
                self.codificacion = result['encoding']
            try:
                self.modelos["RF"] = self.clasificadorArboles()
                self.modelos["SVM"] = self.clasificadorSVM()
                self.modelos["GB"] = self.clasificadorGradientBoosting()
                self.modelos["MLP"] = self.clasificadorRedNeuronal()
            except Exception as e:
                print("error en algoritmos: {}".format(str(e)))
                pass 

            file = open("IEEE.txt", "a")
            for etiqueta, contenido in self.resultado_dict.items():
                file.write("{}:\n {}\n\n".format(etiqueta, contenido))
        else:
            print("error al crear el algoritmo, posiblemente hay muy pocas muestras en dataset")   


##Entrenamiento del modelo final en una instancia del objeto ClasificadorDepresion
objeto = ClasificadorDepresion(
                            archivo="selectedFeaturesDataset.csv",
                            reporte="reporteIEEE.txt"
                        )
objeto.main()
# Este es el modelo 
# objeto.modelos["SVM","GB","MLP","RF"]

