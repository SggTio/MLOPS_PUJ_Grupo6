"""
Módulo responsable de la gestión, persistencia y carga de múltiples modelos de ML.

Este módulo actúa como el "registry" central de modelos, manejando múltiples algoritmos,
serialización, deserialización, versionado y metadata. Soporta selección dinámica
de modelos en tiempo de ejecución para experimentación y A/B testing.
"""

import joblib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

# Configurar logging para este módulo
logger = logging.getLogger(__name__)


class MultiModelManager:
    """
    Clase responsable de la gestión completa del ciclo de vida de múltiples modelos.
    
    Esta clase extiende el patrón Repository para manejar múltiples algoritmos,
    permitiendo guardar, cargar, y seleccionar entre diferentes modelos entrenados.
    Ideal para experimentación, A/B testing, y sistemas de ML en producción.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializar el gestor multi-modelo.
        
        Args:
            models_dir (str): Directorio base donde se almacenarán los modelos
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Archivos estándar
        self.scaler_filename = "scaler.pkl"
        self.models_registry_filename = "models_registry.json"
        self.comparison_filename = "models_comparison.json"
        self.active_model_filename = "active_model.txt"
        
        # Tipos de modelos soportados
        self.supported_models = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'SVC': SVC,
            'MLPClassifier': MLPClassifier
        }
        
        logger.info(f"MultiModelManager inicializado - Directorio: {self.models_dir}")
    
    def save_multiple_models(self, trained_models: Dict[str, Any], scaler: StandardScaler, 
                           comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Guardar múltiples modelos entrenados con sus metadatos.
        
        Args:
            trained_models (Dict[str, Any]): Diccionario de modelos entrenados
            scaler (StandardScaler): Escalador compartido
            comparison_results (Dict[str, Any]): Resultados de comparación
            
        Returns:
            Dict[str, Any]: Información de archivos guardados
        """
        logger.info(f"Guardando {len(trained_models)} modelos...")
        
        saved_files = {}
        
        try:
            # 1. Guardar escalador (compartido por todos los modelos)
            scaler_path = self.models_dir / self.scaler_filename
            joblib.dump(scaler, scaler_path)
            saved_files["scaler"] = str(scaler_path)
            logger.info(f"✅ Escalador guardado: {scaler_path}")
            
            # 2. Guardar cada modelo individual
            models_info = {}
            for model_name, model in trained_models.items():
                model_filename = f"{model_name}_model.pkl"
                model_path = self.models_dir / model_filename
                
                joblib.dump(model, model_path)
                saved_files[f"model_{model_name}"] = str(model_path)
                
                # Información del modelo para el registro
                models_info[model_name] = {
                    "filename": model_filename,
                    "algorithm": type(model).__name__,
                    "saved_timestamp": datetime.now().isoformat(),
                    "file_size_kb": round(model_path.stat().st_size / 1024, 2)
                }
                
                logger.info(f"✅ {model_name} guardado: {model_path}")
            
            # 3. Crear registro de modelos
            registry = {
                "models": models_info,
                "scaler_file": self.scaler_filename,
                "last_updated": datetime.now().isoformat(),
                "total_models": len(models_info)
            }
            
            registry_path = self.models_dir / self.models_registry_filename
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            saved_files["registry"] = str(registry_path)
            
            # 4. Guardar resultados de comparación
            comparison_path = self.models_dir / self.comparison_filename
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            saved_files["comparison"] = str(comparison_path)
            
            # 5. Establecer modelo activo (el mejor por defecto)
            best_model = comparison_results.get("experiment_info", {}).get("best_model")
            if best_model and best_model in trained_models:
                self.set_active_model(best_model)
                saved_files["active_model"] = str(self.models_dir / self.active_model_filename)
            
            logger.info(f"✅ Guardado completo: {len(saved_files)} archivos creados")
            
            return {
                "status": "success",
                "files_saved": saved_files,
                "models_count": len(trained_models),
                "best_model": best_model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")
            raise
    
    def load_model_by_name(self, model_name: str) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
        """
        Cargar un modelo específico por nombre.
        
        Args:
            model_name (str): Nombre del modelo a cargar
            
        Returns:
            Tuple[Any, StandardScaler, Dict[str, Any]]: Modelo, scaler, metadata
        """
        logger.info(f"Cargando modelo específico: {model_name}")
        
        # Cargar registro de modelos
        registry = self._load_models_registry()
        
        if model_name not in registry["models"]:
            available = list(registry["models"].keys())
            raise ValueError(f"Modelo '{model_name}' no encontrado. Disponibles: {available}")
        
        try:
            # Cargar scaler
            scaler_path = self.models_dir / registry["scaler_file"]
            scaler = joblib.load(scaler_path)
            
            # Cargar modelo específico
            model_info = registry["models"][model_name]
            model_path = self.models_dir / model_info["filename"]
            model = joblib.load(model_path)
            
            # Cargar metadata de comparación
            comparison_path = self.models_dir / self.comparison_filename
            with open(comparison_path, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
            
            # Extraer metadata específica del modelo
            model_metadata = comparison_data.get("detailed_metrics", {}).get(model_name, {})
            model_metadata["model_info"] = model_info
            model_metadata["registry_info"] = registry
            
            logger.info(f"✅ Modelo {model_name} cargado exitosamente")
            
            return model, scaler, model_metadata
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo {model_name}: {e}")
            raise
    
    def load_active_model(self) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
        """
        Cargar el modelo actualmente activo.
        
        Returns:
            Tuple[Any, StandardScaler, Dict[str, Any]]: Modelo activo, scaler, metadata
        """
        active_model_name = self.get_active_model_name()
        
        if not active_model_name:
            # Si no hay modelo activo, usar el mejor de la comparación
            comparison = self._load_comparison_results()
            active_model_name = comparison.get("experiment_info", {}).get("best_model")
            
            if not active_model_name:
                raise ValueError("No hay modelo activo definido y no se pudo determinar el mejor")
        
        return self.load_model_by_name(active_model_name)
    
    def set_active_model(self, model_name: str) -> bool:
        """
        Establecer qué modelo debe usarse como activo.
        
        Args:
            model_name (str): Nombre del modelo a activar
            
        Returns:
            bool: True si se estableció correctamente
        """
        try:
            # Verificar que el modelo existe
            registry = self._load_models_registry()
            
            if model_name not in registry["models"]:
                available = list(registry["models"].keys())
                raise ValueError(f"Modelo '{model_name}' no encontrado. Disponibles: {available}")
            
            # Guardar modelo activo
            active_path = self.models_dir / self.active_model_filename
            with open(active_path, 'w', encoding='utf-8') as f:
                f.write(model_name)
            
            logger.info(f"✅ Modelo activo establecido: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error estableciendo modelo activo {model_name}: {e}")
            return False
    
    def get_active_model_name(self) -> Optional[str]:
        """
        Obtener el nombre del modelo actualmente activo.
        
        Returns:
            Optional[str]: Nombre del modelo activo o None
        """
        active_path = self.models_dir / self.active_model_filename
        
        if not active_path.exists():
            return None
            
        try:
            with open(active_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            return None
    
    def list_available_models(self) -> Dict[str, Any]:
        """
        Listar todos los modelos disponibles con su información.
        
        Returns:
            Dict[str, Any]: Información de modelos disponibles
        """
        try:
            registry = self._load_models_registry()
            comparison = self._load_comparison_results()
            active_model = self.get_active_model_name()
            
            models_list = []
            
            for model_name, model_info in registry["models"].items():
                # Obtener métricas de la comparación
                metrics = comparison.get("detailed_metrics", {}).get(model_name, {})
                summary = comparison.get("models_summary", {}).get(model_name, {})
                
                model_entry = {
                    "name": model_name,
                    "algorithm": model_info.get("algorithm", "unknown"),
                    "is_active": model_name == active_model,
                    "is_best": summary.get("is_best", False),
                    "test_accuracy": summary.get("test_accuracy", 0.0),
                    "cv_accuracy": summary.get("cv_accuracy", 0.0),
                    "training_time": summary.get("training_time", 0.0),
                    "file_size_kb": model_info.get("file_size_kb", 0.0),
                    "saved_timestamp": model_info.get("saved_timestamp", "unknown")
                }
                
                models_list.append(model_entry)
            
            # Ordenar por accuracy de test (descendente)
            models_list.sort(key=lambda x: x["test_accuracy"], reverse=True)
            
            return {
                "total_models": len(models_list),
                "active_model": active_model,
                "best_model": comparison.get("experiment_info", {}).get("best_model"),
                "models": models_list,
                "last_updated": registry.get("last_updated", "unknown")
            }
            
        except Exception as e:
            logger.error(f"❌ Error listando modelos: {e}")
            return {"error": str(e), "models": []}
    
    def get_models_comparison(self) -> Dict[str, Any]:
        """
        Obtener resultados completos de comparación entre modelos.
        
        Returns:
            Dict[str, Any]: Comparación detallada de modelos
        """
        try:
            return self._load_comparison_results()
        except Exception as e:
            logger.error(f"❌ Error cargando comparación: {e}")
            return {"error": str(e)}
    
    def delete_model(self, model_name: str) -> bool:
        """
        Eliminar un modelo específico.
        
        Args:
            model_name (str): Nombre del modelo a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        try:
            registry = self._load_models_registry()
            
            if model_name not in registry["models"]:
                logger.warning(f"Modelo {model_name} no existe")
                return False
            
            # Eliminar archivo del modelo
            model_info = registry["models"][model_name]
            model_path = self.models_dir / model_info["filename"]
            
            if model_path.exists():
                model_path.unlink()
            
            # Actualizar registro
            del registry["models"][model_name]
            registry["last_updated"] = datetime.now().isoformat()
            registry["total_models"] = len(registry["models"])
            
            # Guardar registro actualizado
            registry_path = self.models_dir / self.models_registry_filename
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            
            # Si era el modelo activo, limpiar
            if self.get_active_model_name() == model_name:
                active_path = self.models_dir / self.active_model_filename
                if active_path.exists():
                    active_path.unlink()
            
            logger.info(f"✅ Modelo {model_name} eliminado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error eliminando modelo {model_name}: {e}")
            return False
    
    def _load_models_registry(self) -> Dict[str, Any]:
        """Cargar el registro de modelos."""
        registry_path = self.models_dir / self.models_registry_filename
        
        if not registry_path.exists():
            raise FileNotFoundError(f"Registro de modelos no encontrado: {registry_path}")
        
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_comparison_results(self) -> Dict[str, Any]:
        """Cargar los resultados de comparación."""
        comparison_path = self.models_dir / self.comparison_filename
        
        if not comparison_path.exists():
            raise FileNotFoundError(f"Comparación de modelos no encontrada: {comparison_path}")
        
        with open(comparison_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def check_models_availability(self) -> Dict[str, Any]:
        """
        Verificar la disponibilidad de todos los artefactos multi-modelo.
        
        Returns:
            Dict[str, Any]: Estado de disponibilidad completo
        """
        try:
            registry_available = (self.models_dir / self.models_registry_filename).exists()
            scaler_available = (self.models_dir / self.scaler_filename).exists()
            comparison_available = (self.models_dir / self.comparison_filename).exists()
            
            if not all([registry_available, scaler_available, comparison_available]):
                return {
                    "available": False,
                    "registry_available": registry_available,
                    "scaler_available": scaler_available,
                    "comparison_available": comparison_available,
                    "reason": "Missing core files"
                }
            
            # Verificar modelos individuales
            registry = self._load_models_registry()
            models_status = {}
            
            for model_name, model_info in registry["models"].items():
                model_path = self.models_dir / model_info["filename"]
                models_status[model_name] = model_path.exists()
            
            all_models_available = all(models_status.values())
            active_model = self.get_active_model_name()
            
            return {
                "available": all_models_available,
                "total_models": len(registry["models"]),
                "models_status": models_status,
                "active_model": active_model,
                "scaler_available": scaler_available,
                "registry_available": registry_available,
                "comparison_available": comparison_available,
                "models_directory": str(self.models_dir),
                "check_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error verificando disponibilidad: {e}")
            return {
                "available": False,
                "error": str(e),
                "check_timestamp": datetime.now().isoformat()
            }


# Funciones de conveniencia para compatibilidad con el sistema anterior
def save_trained_model(model: Any, scaler: StandardScaler, metadata: Dict[str, Any], 
                      models_dir: str = "models") -> Dict[str, str]:
    """
    Función de compatibilidad para guardar un único modelo.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler entrenado  
        metadata: Metadata del modelo
        models_dir: Directorio donde guardar
        
    Returns:
        Dict[str, str]: Rutas de archivos guardados
    """
    # Para compatibilidad, guardar como modelo único con el nombre "logistic_regression"
    model_name = "logistic_regression"
    
    # Crear estructura compatible con multi-modelo
    trained_models = {model_name: model}
    
    # Crear estructura de comparación simple
    comparison_results = {
        "experiment_info": {
            "total_models_trained": 1,
            "best_model": model_name,
            "timestamp": datetime.now().isoformat()
        },
        "models_summary": {
            model_name: {
                "model_name": model_name,
                "algorithm": type(model).__name__,
                "test_accuracy": metadata.get("performance_metrics", {}).get("accuracy", 0.0),
                "is_best": True
            }
        },
        "detailed_metrics": {
            model_name: metadata
        }
    }
    
    manager = MultiModelManager(models_dir)
    result = manager.save_multiple_models(trained_models, scaler, comparison_results)
    
    # Retornar formato compatible
    return {
        "model": result["files_saved"].get(f"model_{model_name}", ""),
        "scaler": result["files_saved"].get("scaler", ""),
        "metadata": result["files_saved"].get("comparison", "")
    }


def load_trained_model(models_dir: str = "models") -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    """
    Función de compatibilidad para cargar modelo activo.
    
    Args:
        models_dir: Directorio donde buscar modelos
        
    Returns:
        Tuple: Modelo activo, scaler, metadata
    """
    manager = MultiModelManager(models_dir)
    return manager.load_active_model()
