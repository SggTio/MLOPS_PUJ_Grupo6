"""
Módulo para entrenamiento y comparación de múltiples algoritmos de ML.

Este módulo implementa un sistema de experimentación que entrena múltiples
algoritmos simultáneamente, compara su rendimiento, y permite seleccionar
el mejor modelo o mantener múltiples modelos para diferentes casos de uso.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score
)
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from datetime import datetime
import joblib
import json
import warnings

# Suprimir warnings de convergencia para MLPs
warnings.filterwarnings('ignore', category=Warning)

# Configurar logging para este módulo
logger = logging.getLogger(__name__)


class MultiModelTrainer:
    """
    Clase para entrenar y comparar múltiples algoritmos de ML.
    
    Esta clase implementa un pipeline de experimentación que:
    1. Entrena múltiples algoritmos con hiperparámetros optimizados
    2. Evalúa cada modelo usando múltiples métricas
    3. Realiza validación cruzada para robustez
    4. Compara modelos y sugiere el mejor
    5. Permite guardar múltiples modelos para selección en runtime
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5):
        """
        Inicializar el entrenador multi-modelo.
        
        Args:
            test_size (float): Proporción de datos para testing
            random_state (int): Semilla para reproducibilidad
            cv_folds (int): Número de folds para validación cruzada
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # Componentes del pipeline
        self.scaler = None
        self.trained_models = {}
        self.model_metrics = {}
        self.feature_columns = []
        self.best_model_name = None
        
        # Configuración de modelos a entrenar
        self.model_configs = self._get_model_configurations()
        
        logger.info(f"MultiModelTrainer inicializado con {len(self.model_configs)} algoritmos")
    
    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Definir configuraciones de modelos con hiperparámetros optimizados.
        
        Returns:
            Dict[str, Dict[str, Any]]: Configuraciones de modelos
        """
        return {
            "logistic_regression": {
                "model": LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    multi_class='ovr',
                    solver='liblinear'
                ),
                "description": "Regresión Logística - Algoritmo lineal rápido y interpretable",
                "family": "linear"
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                "description": "Random Forest - Ensemble robusto con alta precisión",
                "family": "ensemble"
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.random_state
                ),
                "description": "Gradient Boosting - Ensemble secuencial de alta performance",
                "family": "ensemble"
            },
            "svm_rbf": {
                "model": SVC(
                    C=1.0,
                    kernel='rbf',
                    gamma='scale',
                    probability=True,  # Para predict_proba
                    random_state=self.random_state
                ),
                "description": "SVM con kernel RBF - Clasificador de márgenes máximos",
                "family": "kernel"
            },
            "neural_network": {
                "model": MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1
                ),
                "description": "Red Neural (MLP) - Aprendizaje no lineal profundo",
                "family": "neural"
            }
        }
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preparar datos para entrenamiento multi-modelo.
        
        Args:
            X (pd.DataFrame): Features del dataset
            y (pd.Series): Target variable
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                X_train_scaled, X_test_scaled, y_train, y_test
        """
        logger.info("Preparando datos para entrenamiento multi-modelo")
        
        # Guardar información sobre features
        self.feature_columns = X.columns.tolist()
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Escalado (importante para SVM y Neural Networks)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Datos preparados: {len(X_train)} train, {len(X_test)} test")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_single_model(self, model_name: str, model_config: Dict[str, Any], 
                          X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, float]:
        """
        Entrenar un modelo individual y medir tiempo.
        
        Args:
            model_name (str): Nombre del modelo
            model_config (Dict[str, Any]): Configuración del modelo
            X_train (np.ndarray): Features de entrenamiento
            y_train (np.ndarray): Target de entrenamiento
            
        Returns:
            Tuple[Any, float]: Modelo entrenado y tiempo de entrenamiento
        """
        logger.info(f"Entrenando {model_name}...")
        
        model = model_config["model"]
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            logger.info(f"✅ {model_name} entrenado en {training_time:.2f}s")
            return model, training_time
            
        except Exception as e:
            logger.error(f"❌ Error entrenando {model_name}: {e}")
            return None, 0.0
    
    def evaluate_model(self, model_name: str, model: Any, 
                      X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray,
                      training_time: float) -> Dict[str, Any]:
        """
        Evaluar un modelo usando múltiples métricas.
        
        Args:
            model_name (str): Nombre del modelo
            model: Modelo entrenado
            X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba
            training_time (float): Tiempo de entrenamiento en segundos
            
        Returns:
            Dict[str, Any]: Métricas completas del modelo
        """
        if model is None:
            return {"error": "Model training failed"}
        
        try:
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)
            
            # Métricas básicas
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Métricas detalladas (macro average para multiclase)
            precision = precision_score(y_test, y_pred_test, average='macro')
            recall = recall_score(y_test, y_pred_test, average='macro')
            f1 = f1_score(y_test, y_pred_test, average='macro')
            
            # Validación cruzada en datos de entrenamiento
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='accuracy')
            
            # Matriz de confusión
            conf_matrix = confusion_matrix(y_test, y_pred_test)
            
            # Reporte de clasificación
            class_report = classification_report(y_test, y_pred_test, output_dict=True)
            
            # Confianza de predicciones
            confidence_scores = np.max(y_proba_test, axis=1)
            
            metrics = {
                "model_name": model_name,
                "training_time_seconds": training_time,
                "accuracy": {
                    "train": float(train_accuracy),
                    "test": float(test_accuracy),
                    "cv_mean": float(cv_scores.mean()),
                    "cv_std": float(cv_scores.std())
                },
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_score_macro": float(f1),
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report,
                "confidence_stats": {
                    "mean": float(confidence_scores.mean()),
                    "std": float(confidence_scores.std()),
                    "min": float(confidence_scores.min()),
                    "max": float(confidence_scores.max())
                },
                "cross_validation": {
                    "scores": cv_scores.tolist(),
                    "mean": float(cv_scores.mean()),
                    "std": float(cv_scores.std())
                }
            }
            
            logger.info(f"📊 {model_name} - Test Accuracy: {test_accuracy:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluando {model_name}: {e}")
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Entrenar y evaluar todos los modelos configurados.
        
        Args:
            X (pd.DataFrame): Features del dataset
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, Any]: Resultados completos de todos los modelos
        """
        logger.info("=== Iniciando entrenamiento multi-modelo ===")
        
        # Preparar datos
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Entrenar cada modelo
        for model_name, model_config in self.model_configs.items():
            logger.info(f"🚀 Procesando {model_name}...")
            
            # Entrenar
            trained_model, training_time = self.train_single_model(
                model_name, model_config, X_train, y_train
            )
            
            if trained_model is not None:
                # Guardar modelo entrenado
                self.trained_models[model_name] = trained_model
                
                # Evaluar
                metrics = self.evaluate_model(
                    model_name, trained_model,
                    X_train, X_test, y_train, y_test,
                    training_time
                )
                
                # Agregar información de configuración
                metrics["config"] = {
                    "description": model_config["description"],
                    "family": model_config["family"],
                    "algorithm": type(trained_model).__name__
                }
                
                self.model_metrics[model_name] = metrics
        
        # Determinar el mejor modelo
        self._find_best_model()
        
        # Crear resumen comparativo
        comparison_results = self._create_model_comparison()
        
        logger.info("=== Entrenamiento multi-modelo completado ===")
        logger.info(f"🏆 Mejor modelo: {self.best_model_name}")
        
        return comparison_results
    
    def _find_best_model(self) -> None:
        """
        Determinar el mejor modelo basado en múltiples criterios.
        """
        if not self.model_metrics:
            return
        
        # Criterios de evaluación (pesos)
        criteria_weights = {
            "test_accuracy": 0.4,      # Accuracy en test
            "cv_mean": 0.3,            # Validación cruzada
            "cv_stability": 0.2,       # Estabilidad (1 - cv_std)
            "training_speed": 0.1      # Velocidad de entrenamiento
        }
        
        model_scores = {}
        
        for model_name, metrics in self.model_metrics.items():
            if "error" in metrics:
                continue
                
            # Normalizar métricas (0-1)
            test_acc = metrics["accuracy"]["test"]
            cv_mean = metrics["accuracy"]["cv_mean"] 
            cv_stability = 1 - metrics["accuracy"]["cv_std"]  # Menos std = más estable
            
            # Velocidad (invertida y normalizada)
            max_time = max([m["training_time_seconds"] for m in self.model_metrics.values() 
                           if "training_time_seconds" in m])
            training_speed = 1 - (metrics["training_time_seconds"] / max_time)
            
            # Calcular score compuesto
            composite_score = (
                test_acc * criteria_weights["test_accuracy"] +
                cv_mean * criteria_weights["cv_mean"] +
                cv_stability * criteria_weights["cv_stability"] +
                training_speed * criteria_weights["training_speed"]
            )
            
            model_scores[model_name] = composite_score
        
        # Encontrar el mejor
        if model_scores:
            self.best_model_name = max(model_scores, key=model_scores.get)
            logger.info(f"🏆 Mejor modelo determinado: {self.best_model_name} (score: {model_scores[self.best_model_name]:.4f})")
    
    def _create_model_comparison(self) -> Dict[str, Any]:
        """
        Crear resumen comparativo de todos los modelos.
        
        Returns:
            Dict[str, Any]: Comparación completa de modelos
        """
        comparison = {
            "experiment_info": {
                "total_models_trained": len(self.trained_models),
                "best_model": self.best_model_name,
                "timestamp": datetime.now().isoformat(),
                "cv_folds": self.cv_folds,
                "test_size": self.test_size,
                "random_state": self.random_state
            },
            "models_summary": {},
            "ranking": [],
            "detailed_metrics": self.model_metrics
        }
        
        # Crear resumen por modelo
        ranking_data = []
        
        for model_name, metrics in self.model_metrics.items():
            if "error" in metrics:
                continue
                
            summary = {
                "model_name": model_name,
                "algorithm": metrics["config"]["algorithm"],
                "family": metrics["config"]["family"],
                "test_accuracy": metrics["accuracy"]["test"],
                "cv_accuracy": metrics["accuracy"]["cv_mean"],
                "cv_std": metrics["accuracy"]["cv_std"],
                "precision": metrics["precision_macro"],
                "recall": metrics["recall_macro"],
                "f1_score": metrics["f1_score_macro"],
                "training_time": metrics["training_time_seconds"],
                "confidence_mean": metrics["confidence_stats"]["mean"],
                "is_best": model_name == self.best_model_name
            }
            
            comparison["models_summary"][model_name] = summary
            ranking_data.append((model_name, metrics["accuracy"]["test"]))
        
        # Crear ranking por accuracy
        ranking_data.sort(key=lambda x: x[1], reverse=True)
        comparison["ranking"] = [{"rank": i+1, "model": name, "accuracy": acc} 
                               for i, (name, acc) in enumerate(ranking_data)]
        
        return comparison
    
    def get_model_by_name(self, model_name: str) -> Optional[Any]:
        """
        Obtener un modelo entrenado por nombre.
        
        Args:
            model_name (str): Nombre del modelo
            
        Returns:
            Optional[Any]: Modelo entrenado o None si no existe
        """
        return self.trained_models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """
        Obtener lista de modelos disponibles.
        
        Returns:
            List[str]: Lista de nombres de modelos entrenados
        """
        return list(self.trained_models.keys())
    
    def get_model_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener métricas de un modelo específico.
        
        Args:
            model_name (str): Nombre del modelo
            
        Returns:
            Optional[Dict[str, Any]]: Métricas del modelo o None
        """
        return self.model_metrics.get(model_name)


def train_multiple_algorithms(X: pd.DataFrame, y: pd.Series, processing_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función de conveniencia para entrenar múltiples algoritmos.
    
    Args:
        X (pd.DataFrame): Features procesadas
        y (pd.Series): Target variable
        processing_info (Dict[str, Any]): Información del procesamiento
        
    Returns:
        Dict[str, Any]: Resultados completos del entrenamiento multi-modelo
    """
    trainer = MultiModelTrainer()
    results = trainer.train_all_models(X, y)
    
    # Agregar información del procesamiento
    results["data_processing_info"] = processing_info
    results["feature_info"] = {
        "feature_columns": trainer.feature_columns,
        "num_features": len(trainer.feature_columns)
    }
    
    # Agregar objetos entrenados para acceso posterior
    results["trained_models"] = trainer.trained_models
    results["scaler"] = trainer.scaler
    
    return results
