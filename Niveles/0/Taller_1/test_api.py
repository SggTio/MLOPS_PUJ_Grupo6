"""
Script de pruebas comprehensivas para el API de clasificación de pingüinos.

Este script valida todos los endpoints y casos de uso principales,
proporcionando confianza en el funcionamiento del sistema completo.
"""

import requests
import json
import time
import sys

# Configuración base
BASE_URL = "http://localhost:8989"
TIMEOUT = 30

def wait_for_api_ready(max_attempts=30):
    """Esperar a que el API esté listo para recibir requests."""
    print("🔄 Esperando a que el API esté listo...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("model_loaded", False):
                    print("✅ API está listo con modelo cargado")
                    return True
                else:
                    print("⚠️  API responde pero modelo no está cargado")
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        print(f"   Intento {attempt + 1}/{max_attempts}")
    
    print("❌ API no responde después de esperar")
    return False

def test_root_endpoint():
    """Probar endpoint raíz"""
    print("\n🔍 Probando endpoint raíz...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Endpoint raíz OK - Servicio: {data.get('service', 'unknown')}")
            print(f"   Versión: {data.get('version', 'unknown')}")
            print(f"   Modelo cargado: {data.get('model_loaded', False)}")
            return True
        else:
            print(f"❌ Error en endpoint raíz: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en endpoint raíz: {e}")
        return False

def test_health_endpoint():
    """Probar endpoint de salud"""
    print("\n🔍 Probando endpoint de salud...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check OK - Estado: {health_data.get('status', 'unknown')}")
            print(f"   Modelo cargado: {health_data.get('model_loaded', False)}")
            print(f"   Scaler cargado: {health_data.get('scaler_loaded', False)}")
            return health_data.get('model_loaded', False) and health_data.get('scaler_loaded', False)
        else:
            print(f"❌ Error en health check: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en health check: {e}")
        return False

def test_model_info():
    """Probar endpoint de información del modelo"""
    print("\n🔍 Probando información del modelo...")
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=TIMEOUT)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Info del modelo OK")
            print(f"   Tipo: {model_info.get('model_type', 'unknown')}")
            print(f"   Versión: {model_info.get('version', 'unknown')}")
            print(f"   Accuracy: {model_info.get('accuracy', 'unknown'):.4f}")
            print(f"   Features: {model_info.get('feature_count', 'unknown')}")
            return True
        else:
            print(f"❌ Error obteniendo info del modelo: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error obteniendo info del modelo: {e}")
        return False

def test_simple_prediction():
    """Probar predicción usando endpoint simplificado"""
    print("\n🔍 Probando predicción simplificada...")
    
    test_data = {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "island": "Torgersen",
        "sex": "Male"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/simple",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción simple exitosa")
            print(f"   Especie predicha: {result['species']}")
            print(f"   Confianza: {result['confidence']:.2%}")
            print(f"   Tiempo procesamiento: {result.get('prediction_metadata', {}).get('processing_time_ms', 'N/A')} ms")
            return True
        else:
            print(f"❌ Error en predicción simple: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Detalle: {error_detail}")
            except:
                print(f"   Respuesta: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error en predicción simple: {e}")
        return False

def test_complete_prediction():
    """Probar predicción usando endpoint completo"""
    print("\n🔍 Probando predicción completa...")
    
    test_data = {
        "bill_length_mm": 46.1,
        "bill_depth_mm": 13.2,
        "flipper_length_mm": 211.0,
        "body_mass_g": 4500.0,
        "island_Biscoe": 1,
        "island_Dream": 0,
        "island_Torgersen": 0,
        "sex_Female": 1,
        "sex_Male": 0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/complete",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción completa exitosa")
            print(f"   Especie predicha: {result['species']}")
            print(f"   Confianza: {result['confidence']:.2%}")
            print(f"   Tiempo procesamiento: {result.get('prediction_metadata', {}).get('processing_time_ms', 'N/A')} ms")
            return True
        else:
            print(f"❌ Error en predicción completa: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Detalle: {error_detail}")
            except:
                print(f"   Respuesta: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error en predicción completa: {e}")
        return False

def test_multiple_species():
    """Probar predicciones para diferentes especies esperadas"""
    print("\n🔍 Probando múltiples especies...")
    
    test_cases = [
        {
            "name": "Adelie típico",
            "data": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "island": "Torgersen",
                "sex": "Male"
            },
            "expected": "Adelie"
        },
        {
            "name": "Chinstrap típico",
            "data": {
                "bill_length_mm": 48.7,
                "bill_depth_mm": 18.4,
                "flipper_length_mm": 195.0,
                "body_mass_g": 3600.0,
                "island": "Dream",
                "sex": "Female"
            },
            "expected": "Chinstrap"
        },
        {
            "name": "Gentoo típico",
            "data": {
                "bill_length_mm": 46.1,
                "bill_depth_mm": 13.2,
                "flipper_length_mm": 211.0,
                "body_mass_g": 4500.0,
                "island": "Biscoe",
                "sex": "Male"
            },
            "expected": "Gentoo"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/predict/simple",
                json=case['data'],
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result['species']
                confidence = result['confidence']
                
                if predicted == case['expected']:
                    print(f"✅ {case['name']}: {predicted} (confianza: {confidence:.2%})")
                    passed += 1
                else:
                    print(f"⚠️  {case['name']}: predicho {predicted}, esperado {case['expected']} (confianza: {confidence:.2%})")
            else:
                print(f"❌ Error en {case['name']}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en {case['name']}: {e}")
    
    print(f"   Resultados especies: {passed}/{total} correctas")
    return passed == total

def main():
    """Ejecutar todas las pruebas del API"""
    print("🚀 Iniciando pruebas del API de Pingüinos Palmer")
    print("=" * 60)
    
    # Esperar a que el API esté listo
    if not wait_for_api_ready():
        print("❌ No se pudo conectar al API")
        sys.exit(1)
    
    # Ejecutar todas las pruebas
    tests = [
        ("Root endpoint", test_root_endpoint),
        ("Health check", test_health_endpoint),
        ("Model info", test_model_info),
        ("Simple prediction", test_simple_prediction),
        ("Complete prediction", test_complete_prediction),
        ("Multiple species", test_multiple_species)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS FINALES: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
        sys.exit(0)
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar logs anteriores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
