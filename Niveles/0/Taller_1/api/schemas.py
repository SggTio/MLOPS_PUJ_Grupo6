"""
Esquemas de datos para el API de clasificación de pingüinos Palmer - VERSIÓN CORREGIDA

Este módulo define todos los modelos de datos usando Pydantic V2, ahora con
las features exactas que espera el modelo entrenado, incluyendo la feature "year"
que estaba faltando y los nombres correctos para las features de sexo.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Optional, Any
from enum import Enum


class IslandEnum(str, Enum):
    """Enumeración de islas válidas en el dataset Palmer Penguins."""
    BISCOE = "Biscoe"
    DREAM = "Dream"
    TORGERSEN = "Torgersen"


class SexEnum(str, Enum):
    """Enumeración de sexos válidos en el dataset Palmer Penguins."""
    MALE = "Male"
    FEMALE = "Female"


class SpeciesEnum(str, Enum):
    """Enumeración de especies de pingüinos que el modelo puede predecir."""
    ADELIE = "Adelie"
    CHINSTRAP = "Chinstrap"
    GENTOO = "Gentoo"


class PenguinFeaturesSimple(BaseModel):
    """
    Esquema para entrada simplificada de características de pingüino.
    
    ACTUALIZADO para incluir el campo "year" que faltaba y corregir
    la transformación para que coincida exactamente con el modelo entrenado.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "year": 2007,
                "island": "Torgersen",
                "sex": "Male"
            }
        }
    )
    
    bill_length_mm: float = Field(
        ..., gt=0, le=100,
        description="Longitud del pico en milímetros",
        examples=[39.1]
    )
    
    bill_depth_mm: float = Field(
        ..., gt=0, le=50,
        description="Profundidad del pico en milímetros", 
        examples=[18.7]
    )
    
    flipper_length_mm: float = Field(
        ..., gt=0, le=300,
        description="Longitud de la aleta en milímetros",
        examples=[181.0]
    )
    
    body_mass_g: float = Field(
        ..., gt=0, le=10000,
        description="Masa corporal en gramos",
        examples=[3750.0]
    )
    
    year: int = Field(
        ..., ge=2007, le=2025,
        description="Año de observación del pingüino",
        examples=[2007]
    )
    
    island: IslandEnum = Field(
        ...,
        description="Isla donde fue observado el pingüino"
    )
    
    sex: SexEnum = Field(
        ...,
        description="Sexo del pingüino"
    )
    
    @field_validator('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g')
    @classmethod
    def validate_positive_measurements(cls, value: float) -> float:
        """Validador para asegurar medidas biológicamente plausibles."""
        if value <= 0:
            raise ValueError('Las medidas biológicas deben ser valores positivos')
        return value


class PenguinFeaturesComplete(BaseModel):
    """
    Esquema para entrada completa con one-hot encoding explícito.
    
    ACTUALIZADO para incluir "year" y usar los nombres exactos de features
    que espera el modelo: "sex_female" y "sex_male" (minúsculas).
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "year": 2007,
                "island_Biscoe": 0,
                "island_Dream": 0,
                "island_Torgersen": 1,
                "sex_female": 0,
                "sex_male": 1
            }
        }
    )
    
    # Features numéricas exactamente como las espera el modelo
    bill_length_mm: float = Field(..., description="Longitud del pico en milímetros")
    bill_depth_mm: float = Field(..., description="Profundidad del pico en milímetros") 
    flipper_length_mm: float = Field(..., description="Longitud de la aleta en milímetros")
    body_mass_g: float = Field(..., description="Masa corporal en gramos")
    year: int = Field(..., description="Año de observación")
    
    # Variables categóricas con one-hot encoding (nombres exactos del modelo)
    island_Biscoe: int = Field(0, ge=0, le=1, description="1 si es isla Biscoe, 0 en caso contrario")
    island_Dream: int = Field(0, ge=0, le=1, description="1 si es isla Dream, 0 en caso contrario")
    island_Torgersen: int = Field(0, ge=0, le=1, description="1 si es isla Torgersen, 0 en caso contrario")
    sex_female: int = Field(0, ge=0, le=1, description="1 si es hembra, 0 en caso contrario")
    sex_male: int = Field(0, ge=0, le=1, description="1 si es macho, 0 en caso contrario")
    
    @field_validator('island_Biscoe', 'island_Dream', 'island_Torgersen')
    @classmethod
    def validate_island_encoding(cls, value: int, info) -> int:
        """Validar que exactamente una isla esté marcada como 1."""
        data = info.data if info.data else {}
        island_fields = ['island_Biscoe', 'island_Dream', 'island_Torgersen']
        current_field = info.field_name
        
        island_values = [data.get(f, 0) for f in island_fields if f != current_field]
        island_values.append(value)
        
        total_islands = sum(island_values)
        if total_islands > 1:
            raise ValueError('Solo una isla puede estar marcada como 1')
        
        return value
    
    @field_validator('sex_female', 'sex_male')
    @classmethod
    def validate_sex_encoding(cls, value: int, info) -> int:
        """Validar que exactamente un sexo esté marcado como 1."""
        data = info.data if info.data else {}
        sex_fields = ['sex_female', 'sex_male']
        current_field = info.field_name
        
        sex_values = [data.get(f, 0) for f in sex_fields if f != current_field]
        sex_values.append(value)
        
        total_sex = sum(sex_values)
        if total_sex > 1:
            raise ValueError('Solo un sexo puede estar marcado como 1')
        
        return value


class PredictionResponse(BaseModel):
    """Esquema para la respuesta de predicción del modelo."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "species": "Adelie",
                "species_code": 1,
                "confidence": 0.85,
                "probabilities": {
                    "Adelie": 0.85,
                    "Chinstrap": 0.10,
                    "Gentoo": 0.05
                },
                "prediction_metadata": {
                    "processing_time_ms": 12.5,
                    "model_version": "1.0.0"
                }
            }
        }
    )
    
    species: SpeciesEnum = Field(..., description="Especie predicha por el modelo")
    species_code: int = Field(..., ge=1, le=3, description="Código numérico de la especie")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la predicción")
    probabilities: Dict[str, float] = Field(..., description="Probabilidades para cada especie")
    prediction_metadata: Optional[Dict[str, Any]] = Field(None, description="Información adicional")


class HealthResponse(BaseModel):
    """Esquema para la respuesta del endpoint de salud del servicio."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "scaler_loaded": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0"
            }
        }
    )
    
    status: str = Field(..., description="Estado general del servicio")
    model_loaded: bool = Field(..., description="Indica si el modelo está cargado correctamente")
    scaler_loaded: bool = Field(..., description="Indica si el scaler está cargado correctamente")
    timestamp: str = Field(..., description="Timestamp de la verificación de salud")
    version: Optional[str] = Field(None, description="Versión del API")


class ModelInfoResponse(BaseModel):
    """Esquema para información detallada sobre el modelo cargado."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "LogisticRegression", 
                "version": "1.0.0",
                "training_date": "2024-01-15T08:00:00Z",
                "accuracy": 0.95,
                "feature_count": 10,
                "target_classes": {
                    "1": "Adelie",
                    "2": "Chinstrap", 
                    "3": "Gentoo"
                },
                "features": [
                    "bill_length_mm", "bill_depth_mm", "flipper_length_mm", 
                    "body_mass_g", "year", "island_Biscoe", "island_Dream", 
                    "island_Torgersen", "sex_female", "sex_male"
                ]
            }
        }
    )
    
    model_type: str = Field(..., description="Tipo de algoritmo del modelo")
    version: str = Field(..., description="Versión del modelo")
    training_date: str = Field(..., description="Fecha de entrenamiento")
    accuracy: float = Field(..., description="Accuracy del modelo en datos de prueba")
    feature_count: int = Field(..., description="Número de características que espera el modelo")
    target_classes: Dict[str, str] = Field(..., description="Mapeo de códigos a nombres de especies")
    features: List[str] = Field(..., description="Lista de características que espera el modelo")


class ModelListItem(BaseModel):
    """Esquema para un elemento en la lista de modelos."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "random_forest",
                "algorithm": "RandomForestClassifier",
                "is_active": True,
                "is_best": False,
                "test_accuracy": 0.9750,
                "cv_accuracy": 0.9680,
                "training_time": 2.45,
                "file_size_kb": 125.7,
                "saved_timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    name: str = Field(..., description="Nombre del modelo")
    algorithm: str = Field(..., description="Algoritmo utilizado")
    is_active: bool = Field(..., description="Si es el modelo actualmente activo")
    is_best: bool = Field(..., description="Si es el modelo con mejor performance")
    test_accuracy: float = Field(..., description="Accuracy en datos de test")
    cv_accuracy: float = Field(..., description="Accuracy de validación cruzada")
    training_time: float = Field(..., description="Tiempo de entrenamiento en segundos")
    file_size_kb: float = Field(..., description="Tamaño del archivo en KB")
    saved_timestamp: str = Field(..., description="Timestamp cuando se guardó")


class ModelsListResponse(BaseModel):
    """Esquema para la respuesta de lista de modelos."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_models": 5,
                "active_model": "random_forest",
                "best_model": "gradient_boosting",
                "models": [
                    {
                        "name": "gradient_boosting",
                        "algorithm": "GradientBoostingClassifier",
                        "is_active": False,
                        "is_best": True,
                        "test_accuracy": 0.9851,
                        "cv_accuracy": 0.9720,
                        "training_time": 8.32,
                        "file_size_kb": 234.5,
                        "saved_timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "last_updated": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    total_models: int = Field(..., description="Número total de modelos disponibles")
    active_model: Optional[str] = Field(None, description="Nombre del modelo activo")
    best_model: Optional[str] = Field(None, description="Nombre del modelo con mejor performance")
    models: List[ModelListItem] = Field(..., description="Lista de modelos disponibles")
    last_updated: str = Field(..., description="Timestamp de última actualización")


class ModelActivationResponse(BaseModel):
    """Esquema para la respuesta de activación de modelo."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Modelo 'random_forest' activado exitosamente",
                "previous_active": "logistic_regression",
                "new_active": "random_forest",
                "model_info": {
                    "algorithm": "RandomForestClassifier",
                    "test_accuracy": 0.9750,
                    "training_date": "2024-01-15T10:30:00Z"
                }
            }
        }
    )
    
    success: bool = Field(..., description="Si la activación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo del resultado")
    previous_active: Optional[str] = Field(None, description="Modelo activo anterior")
    new_active: str = Field(..., description="Nuevo modelo activo")
    model_info: Dict[str, Any] = Field(..., description="Información del modelo activado")


class ModelComparisonResponse(BaseModel):
    """Esquema para la respuesta de comparación de modelos."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiment_info": {
                    "total_models_trained": 5,
                    "best_model": "gradient_boosting",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "cv_folds": 5,
                    "test_size": 0.2
                },
                "ranking": [
                    {"rank": 1, "model": "gradient_boosting", "accuracy": 0.9851},
                    {"rank": 2, "model": "random_forest", "accuracy": 0.9750}
                ],
                "models_summary": {}
            }
        }
    )
    
    experiment_info: Dict[str, Any] = Field(..., description="Información del experimento")
    ranking: List[Dict[str, Any]] = Field(..., description="Ranking de modelos por performance")
    models_summary: Dict[str, Any] = Field(..., description="Resumen de cada modelo")
    detailed_metrics: Optional[Dict[str, Any]] = Field(None, description="Métricas detalladas de cada modelo")


class ModelDeletionResponse(BaseModel):
    """Esquema para la respuesta de eliminación de modelo."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Modelo 'old_model' eliminado exitosamente",
                "deleted_model": "old_model",
                "remaining_models": 4,
                "new_active": "random_forest"
            }
        }
    )
    
    success: bool = Field(..., description="Si la eliminación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo del resultado")
    deleted_model: str = Field(..., description="Nombre del modelo eliminado")
    remaining_models: int = Field(..., description="Número de modelos restantes")
    new_active: Optional[str] = Field(None, description="Nuevo modelo activo si cambió")


class MultiModelTrainingRequest(BaseModel):
    """Esquema para solicitud de entrenamiento multi-modelo."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "algorithms": ["logistic_regression", "random_forest", "svm_rbf"],
                "cv_folds": 5,
                "test_size": 0.2,
                "auto_activate_best": True
            }
        }
    )
    
    algorithms: Optional[List[str]] = Field(
        None, 
        description="Lista de algoritmos a entrenar. Si es None, entrena todos los disponibles"
    )
    cv_folds: int = Field(5, ge=2, le=10, description="Número de folds para validación cruzada")
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Proporción de datos para testing")
    auto_activate_best: bool = Field(True, description="Activar automáticamente el mejor modelo")


class ErrorResponse(BaseModel):
    """Esquema estándar para respuestas de error."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "bill_length_mm debe ser un valor positivo",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456789"
            }
        }
    )
    
    error: str = Field(..., description="Tipo de error")
    message: str = Field(..., description="Descripción detallada del error")
    timestamp: str = Field(..., description="Timestamp cuando ocurrió el error")
    request_id: Optional[str] = Field(None, description="ID único para tracking del error")
