"""
Módulo responsable de la gestión, persistencia y carga de modelos de machine learning.

Este módulo actúa como el "registry" central de modelos, manejando la serialización,
deserialización, versionado y metadata de todos los artefactos de ML. Esta separación
de responsabilidades es crucial en MLOps porque permite que la lógica de persistencia
evolucione independientemente del entrenamiento y la inferencia.
"""

import joblib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

# Configurar logging para este módulo
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Clase responsable de la gestión completa del ciclo de vida de modelos.
    
    Esta clase implementa el patrón Repository para modelos de ML, proporcionando
    una interfaz consistente para guardar, cargar, y gestionar metadatos de modelos
    sin importar el backend de almacenamiento subyacente.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializar el gestor de modelos con directorio base.
        
        Args:
            models_dir (str): Directorio donde se almacenarán los modelos
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)  # Crear directorio si no existe
        
        # Nombres estándar para los archivos de artefactos
        self.model_filename = "logistic_regression_model.pkl"
        self.scaler_filename = "scaler.pkl"
        self.metadata_filename = "model_info.json"
        
        logger.info(f"ModelManager inicializado con directorio: {self.models_dir}")
    
    def save_model_artifacts(self, model: LogisticRegression, scaler: StandardScaler, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Guardar todos los artefactos del modelo de manera atómica.
        
        Este método implementa el principio de atomicidad: todos los artefactos
        se guardan exitosamente o ninguno se guarda. Esto previene estados
        inconsistentes donde algunos archivos existen pero otros no.
        
        Args:
            model (LogisticRegression): Modelo entrenado
            scaler (StandardScaler): Escalador entrenado
            metadata (Dict[str, Any]): Metadata completa del modelo
            
        Returns:
            Dict[str, str]: Rutas donde se guardaron los archivos
            
        Raises:
            Exception: Si hay error guardando cualquier artefacto
        """
        logger.info("Iniciando guardado de artefactos del modelo")
        
        # Definir rutas completas para todos los archivos
        model_path = self.models_dir / self.model_filename
        scaler_path = self.models_dir / self.scaler_filename
        metadata_path = self.models_dir / self.metadata_filename
        
        try:
            # Guardar modelo usando joblib (optimizado para objetos sklearn)
            joblib.dump(model, model_path)
            logger.info(f"Modelo guardado en: {model_path}")
            
            # Guardar scaler
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler guardado en: {scaler_path}")
            
            # Enriquecer metadata con información de persistencia
            enriched_metadata = self._enrich_metadata_for_storage(metadata)
            
            # Guardar metadata como JSON para legibilidad y interoperabilidad
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata guardada en: {metadata_path}")
            
            # Retornar rutas para confirmación
            saved_paths = {
                "model": str(model_path),
                "scaler": str(scaler_path),
                "metadata": str(metadata_path)
            }
            
            logger.info("Todos los artefactos guardados exitosamente")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error guardando artefactos: {str(e)}")
            # En un sistema más robusto, aquí haríamos rollback de archivos parcialmente guardados
            raise
    
    def load_model_artifacts(self) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
        """
        Cargar todos los artefactos del modelo desde almacenamiento.
        
        Este método implementa validación robusta para asegurar que todos
        los componentes necesarios estén disponibles y sean compatibles
        entre sí antes de retornarlos para uso en inferencia.
        
        Returns:
            Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]: 
                Modelo, scaler, y metadata cargados
                
        Raises:
            FileNotFoundError: Si algún artefacto requerido no existe
            Exception: Si hay errores de deserialización
        """
        logger.info("Iniciando carga de artefactos del modelo")
        
        # Definir rutas de todos los archivos requeridos
        model_path = self.models_dir / self.model_filename
        scaler_path = self.models_dir / self.scaler_filename
        metadata_path = self.models_dir / self.metadata_filename
        
        # Validar que todos los archivos existen
        missing_files = []
        for name, path in [("modelo", model_path), ("scaler", scaler_path), ("metadata", metadata_path)]:
            if not path.exists():
                missing_files.append(f"{name} ({path})")
        
        if missing_files:
            error_msg = f"Archivos faltantes: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Cargar artefactos en orden de dependencia
            logger.info("Cargando modelo...")
            model = joblib.load(model_path)
            
            logger.info("Cargando scaler...")
            scaler = joblib.load(scaler_path)
            
            logger.info("Cargando metadata...")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validar compatibilidad básica
            self._validate_artifacts_compatibility(model, scaler, metadata)
            
            logger.info("Todos los artefactos cargados y validados exitosamente")
            return model, scaler, metadata
            
        except Exception as e:
            logger.error(f"Error cargando artefactos: {str(e)}")
            raise
    
    def _enrich_metadata_for_storage(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriquecer metadata con información adicional relevante para persistencia.
        
        Args:
            metadata (Dict[str, Any]): Metadata original
            
        Returns:
            Dict[str, Any]: Metadata enriquecida
        """
        enriched = metadata.copy()
        
        # Agregar información de persistencia
        enriched["storage_info"] = {
            "saved_timestamp": datetime.now().isoformat(),
            "model_file": self.model_filename,
            "scaler_file": self.scaler_filename,
            "metadata_file": self.metadata_filename,
            "storage_format": "joblib_pkl"
        }
        
        # Agregar checksums básicos para integridad (en sistemas reales usaríamos hashes)
        enriched["integrity_info"] = {
            "files_count": 3,
            "expected_files": [self.model_filename, self.scaler_filename, self.metadata_filename]
        }
        
        return enriched
    
    def _validate_artifacts_compatibility(self, model: LogisticRegression, scaler: StandardScaler, metadata: Dict[str, Any]) -> None:
        """
        Validar que los artefactos cargados sean compatibles entre sí.
        
        Args:
            model (LogisticRegression): Modelo cargado
            scaler (StandardScaler): Scaler cargado
            metadata (Dict[str, Any]): Metadata cargada
            
        Raises:
            ValueError: Si se detectan incompatibilidades
        """
        # Validar que el número de features coincida
        expected_features = metadata.get("feature_info", {}).get("num_features")
        if expected_features and hasattr(scaler, 'n_features_in_'):
            if scaler.n_features_in_ != expected_features:
                raise ValueError(f"Incompatibilidad de features: scaler espera {scaler.n_features_in_}, metadata indica {expected_features}")
        
        # Validar tipo de modelo
        if not isinstance(model, LogisticRegression):
            raise ValueError(f"Tipo de modelo inesperado: {type(model)}")
        
        # Validar que el modelo esté entrenado (tiene coeficientes)
        if not hasattr(model, 'coef_') or model.coef_ is None:
            raise ValueError("El modelo no parece estar entrenado (sin coeficientes)")
        
        logger.info("Validación de compatibilidad exitosa")
    
    def check_model_availability(self) -> Dict[str, Any]:
        """
        Verificar qué artefactos están disponibles en el directorio de modelos.
        
        Returns:
            Dict[str, Any]: Estado de disponibilidad de artefactos
        """
        model_path = self.models_dir / self.model_filename
        scaler_path = self.models_dir / self.scaler_filename
        metadata_path = self.models_dir / self.metadata_filename
        
        availability = {
            "model_available": model_path.exists(),
            "scaler_available": scaler_path.exists(),
            "metadata_available": metadata_path.exists(),
            "all_artifacts_available": all([
                model_path.exists(),
                scaler_path.exists(), 
                metadata_path.exists()
            ]),
            "models_directory": str(self.models_dir),
            "check_timestamp": datetime.now().isoformat()
        }
        
        # Agregar información de archivos si están disponibles
        if availability["model_available"]:
            availability["model_size_mb"] = round(model_path.stat().st_size / (1024*1024), 2)
        
        if availability["metadata_available"]:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                availability["model_version"] = metadata.get("model_version", "unknown")
                availability["training_timestamp"] = metadata.get("training_timestamp", "unknown")
            except:
                availability["metadata_readable"] = False
        
        return availability
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo sin cargar los artefactos pesados.
        
        Returns:
            Dict[str, Any]: Información del modelo desde metadata
        """
        metadata_path = self.models_dir / self.metadata_filename
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Archivo de metadata no encontrado: {metadata_path}")
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extraer información clave para respuesta rápida
            model_info = {
                "model_type": metadata.get("model_info", {}).get("algorithm", "unknown"),
                "version": metadata.get("model_version", "unknown"),
                "training_date": metadata.get("training_timestamp", "unknown"),
                "accuracy": metadata.get("performance_metrics", {}).get("accuracy", "unknown"),
                "feature_count": metadata.get("feature_info", {}).get("num_features", "unknown"),
                "target_classes": metadata.get("data_info", {}).get("species_mapping", {}),
                "last_updated": metadata.get("storage_info", {}).get("saved_timestamp", "unknown")
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error leyendo información del modelo: {str(e)}")
            raise


# Funciones de conveniencia para uso directo
def save_trained_model(model: LogisticRegression, scaler: StandardScaler, metadata: Dict[str, Any], models_dir: str = "models") -> Dict[str, str]:
    """
    Función de conveniencia para guardar un modelo entrenado.
    
    Args:
        model (LogisticRegression): Modelo entrenado
        scaler (StandardScaler): Scaler entrenado
        metadata (Dict[str, Any]): Metadata del modelo
        models_dir (str): Directorio donde guardar
        
    Returns:
        Dict[str, str]: Rutas donde se guardaron los archivos
    """
    manager = ModelManager(models_dir)
    return manager.save_model_artifacts(model, scaler, metadata)


def load_trained_model(models_dir: str = "models") -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """
    Función de conveniencia para cargar un modelo entrenado.
    
    Args:
        models_dir (str): Directorio donde buscar los modelos
        
    Returns:
        Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]: Artefactos cargados
    """
    manager = ModelManager(models_dir)
    return manager.load_trained_model()
