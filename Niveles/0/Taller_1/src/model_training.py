"""
Módulo responsable del entrenamiento y evaluación de modelos de machine learning.

Este módulo encapsula toda la lógica relacionada con el entrenamiento de modelos,
evaluación de performance, y preparación de artefactos para producción. La separación
de estas responsabilidades del procesamiento de datos permite que cada componente
evolucione independientemente y facilita la experimentación con diferentes algoritmos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict, Any, Optional
import logging
import json
from datetime import datetime

# Configurar logging para este módulo
logger = logging.getLogger(__name__)


class PenguinModelTrainer:
    """
    Clase responsable del entrenamiento completo de modelos para clasificación de pingüinos.
    
    Esta clase implementa el patrón Strategy, permitiendo que diferentes algoritmos
    de machine learning sean intercambiados fácilmente sin cambiar el resto del sistema.
    También incluye capacidades de logging y métricas que son esenciales para MLOps.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Inicializar el entrenador con configuración predeterminada.
        
        Args:
            test_size (float): Proporción de datos para testing (0.2 = 20%)
            random_state (int): Semilla para reproducibilidad
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # Componentes del pipeline que se inicializarán durante entrenamiento
        self.scaler = None
        self.model = None
        
        # Métricas y metadatos que se generarán durante entrenamiento
        self.training_metrics = {}
        self.feature_columns = []
        self.target_mapping = {}
        
        logger.info(f"ModelTrainer inicializado con test_size={test_size}, random_state={random_state}")
    
    def prepare_train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Dividir datos en conjuntos de entrenamiento y prueba, y aplicar escalado.
        
        Este método implementa las mejores prácticas de ML asegurando que:
        1. La división sea reproducible (random_state fijo)
        2. El escalado se ajuste solo en datos de entrenamiento 
        3. El mismo escalado se aplique a datos de prueba
        
        Args:
            X (pd.DataFrame): Features del dataset
            y (pd.Series): Target variable
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                X_train_scaled, X_test_scaled, y_train, y_test
        """
        logger.info("Preparando división train/test y escalado")
        
        # Guardar información sobre las features para uso posterior
        self.feature_columns = X.columns.tolist()
        logger.info(f"Features para entrenamiento: {len(self.feature_columns)} columnas")
        
        # Dividir datos manteniendo distribución de clases
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Mantener proporción de clases en ambos conjuntos
        )
        
        logger.info(f"División completada: {len(X_train)} entrenamiento, {len(X_test)} prueba")
        
        # Inicializar y ajustar el escalador solo con datos de entrenamiento
        # Esto es crucial para evitar data leakage
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)  # Solo transform, no fit
        
        logger.info("Escalado Z-score aplicado exitosamente")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Entrenar modelo de regresión logística con configuración optimizada.
        
        Args:
            X_train (np.ndarray): Features de entrenamiento escaladas
            y_train (np.ndarray): Target de entrenamiento
            
        Returns:
            LogisticRegression: Modelo entrenado
        """
        logger.info("Iniciando entrenamiento de regresión logística")
        
        # Configurar modelo con parámetros optimizados para el problema
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,  # Suficientes iteraciones para convergencia
            multi_class='ovr',  # One-vs-Rest para clasificación multiclase
            solver='liblinear'  # Solver robusto para datasets pequeños-medianos
        )
        
        # Entrenar el modelo
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluar el modelo entrenado usando múltiples métricas.
        
        Args:
            X_test (np.ndarray): Features de prueba
            y_test (np.ndarray): Target verdadero de prueba
            
        Returns:
            Dict[str, Any]: Métricas de evaluación completas
        """
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado. Ejecuta train_logistic_regression primero.")
        
        logger.info("Evaluando performance del modelo")
        
        # Generar predicciones
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calcular métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generar reporte de clasificación detallado
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Compilar métricas en estructura organizada
        metrics = {
            "accuracy": float(accuracy),
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),  # Convertir a lista para JSON serialization
            "prediction_distribution": {
                "y_test_unique": np.unique(y_test, return_counts=True)[1].tolist(),
                "y_pred_unique": np.unique(y_pred, return_counts=True)[1].tolist()
            },
            "model_confidence": {
                "mean_max_probability": float(np.mean(np.max(y_proba, axis=1))),
                "std_max_probability": float(np.std(np.max(y_proba, axis=1)))
            }
        }
        
        # Guardar métricas para uso posterior
        self.training_metrics = metrics
        
        logger.info(f"Evaluación completada - Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def generate_model_metadata(self, processing_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar metadata completa sobre el modelo y el proceso de entrenamiento.
        
        Args:
            processing_info (Dict[str, Any]): Información del procesamiento de datos
            
        Returns:
            Dict[str, Any]: Metadata completa del modelo
        """
        metadata = {
            "model_info": {
                "algorithm": "LogisticRegression",
                "library": "scikit-learn",
                "hyperparameters": {
                    "random_state": self.random_state,
                    "max_iter": 1000,
                    "multi_class": "ovr",
                    "solver": "liblinear"
                }
            },
            "training_config": {
                "test_size": self.test_size,
                "random_state": self.random_state,
                "scaling_method": "StandardScaler"
            },
            "data_info": processing_info,
            "feature_info": {
                "feature_columns": self.feature_columns,
                "num_features": len(self.feature_columns)
            },
            "performance_metrics": self.training_metrics,
            "training_timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
        
        return metadata
    
    def train_complete_pipeline(self, X: pd.DataFrame, y: pd.Series, processing_info: Dict[str, Any]) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
        """
        Ejecutar el pipeline completo de entrenamiento incluyendo evaluación.
        
        Args:
            X (pd.DataFrame): Features procesadas
            y (pd.Series): Target variable
            processing_info (Dict[str, Any]): Información del procesamiento de datos
            
        Returns:
            Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]: 
                Modelo entrenado, scaler, metadata completa
        """
        logger.info("=== Iniciando pipeline completo de entrenamiento ===")
        
        # Paso 1: Preparar división y escalado
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_train_test_split(X, y)
        
        # Paso 2: Entrenar modelo
        trained_model = self.train_logistic_regression(X_train_scaled, y_train)
        
        # Paso 3: Evaluar modelo
        metrics = self.evaluate_model(X_test_scaled, y_test)
        
        # Paso 4: Generar metadata completa
        model_metadata = self.generate_model_metadata(processing_info)
        
        logger.info("=== Pipeline de entrenamiento completado exitosamente ===")
        logger.info(f"Accuracy final: {metrics['accuracy']:.4f}")
        
        return trained_model, self.scaler, model_metadata


def train_penguin_classifier(X: pd.DataFrame, y: pd.Series, processing_info: Dict[str, Any]) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """
    Función de conveniencia para entrenar un clasificador de pingüinos.
    
    Esta función proporciona una interfaz simple para ejecutar todo el pipeline
    de entrenamiento sin necesidad de instanciar directamente la clase.
    
    Args:
        X (pd.DataFrame): Features procesadas
        y (pd.Series): Target variable  
        processing_info (Dict[str, Any]): Información del procesamiento
        
    Returns:
        Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]: 
            Modelo entrenado, scaler, metadata
    """
    trainer = PenguinModelTrainer()
    return trainer.train_complete_pipeline(X, y, processing_info)
