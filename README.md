# Análisis de Efectividad de Campañas de Marketing y ROI con Integración de Machine Learning y Power BI

## Visión General
Este proyecto desarrolla una solución integral para analizar el rendimiento de campañas de marketing en múltiples canales. Al aprovechar SQL para gestión de datos, Python para modelos predictivos y Power BI para visualización, el sistema identifica canales de alto rendimiento, calcula métricas precisas de ROI y proporciona insights accionables para optimización de presupuesto. El modelo Random Forest alcanza un 96.9% de precisión (R²) en la predicción de ROI, permitiendo decisiones basadas en datos.

**Explora los componentes de la aplicación:**  
- **Prueba el modelo [aquí](https://enncd44sfjc7sspt8yyeqr.streamlit.app/):**  
- **Ver el análisis de Power BI [aquí](https://github.com/Bokols/Analisis-de-Efectividad-y-ROI-de-Campanas-de-Marketing-/blob/master/Powerbi.pdf):**  

## Introducción
En el panorama actual del marketing digital, las organizaciones invierten fuertemente en múltiples canales - incluyendo redes sociales, email, motores de búsqueda y anuncios display - para atraer clientes e incrementar ingresos. Este proyecto aborda el desafío crítico de entender el verdadero impacto de estos esfuerzos mediante:

- Limpieza y preprocesamiento de 200,000 registros de campañas
- Resolución de problemas de calidad de datos (discrepancias en tasas de conversión, irrelevancia de scores de engagement)
- Desarrollo de modelos predictivos para estimación de ROI
- Creación de dashboards interactivos para monitoreo de desempeño
- Construcción de una aplicación Streamlit para simulación de campañas

La solución empodera a equipos de marketing para asignar presupuestos efectivamente identificando los canales y estrategias que generan mayor retorno de inversión.

## Componentes del Proyecto

### 1. Implementación de Base de Datos SQL
**Estructura de la Base de Datos:**  
- Creación de base de datos `marketing_roi_analysis` con tabla de 19 columnas almacenando:
  - Metadatos de campaña (ID, compañía, tipo, audiencia)
  - Métricas de desempeño (clicks, impresiones, tasa de conversión)
  - Datos financieros (costo de adquisición, ingresos en CLP)
  - Datos temporales (fechas de inicio/fin)

**Scripts SQL Clave:**  
- `CREATE DATABASE marketing_roi_analysis.sql` - Inicialización de base de datos  
- `CREATE TABLE.sql` - Definición de esquema para almacenamiento  
- `CHECK TABLE.sql` - Consultas de validación para integridad  

### 2. Análisis de Datos y Machine Learning
**Pipeline de Procesamiento:**  
- Limpieza de 200,000 registros (0 duplicados, 100% casos completos)  
- Resolución de problemas críticos:
  - 90% discrepancias en tasas de conversión (reportado vs calculado)
  - 10% campañas atípicas con CTRs irreales (>30%)
  - Scores de engagement sin correlación

**Ingeniería de Características:**  
- Creación de 37 características incluyendo:
  - Elementos temporales (mes/trimestre/año de campaña)
  - Métricas de eficiencia (CTR, costo por click, ingreso por click)
  - Indicadores de desempeño por canal
  - Banderas de alto rendimiento (top 20% campañas por ROI)

**Desarrollo de Modelos:**  

| Modelo       | RMSE   | R²     | Tiempo Entrenamiento |
|-------------|--------|--------|---------------|
| Random Forest | 0.3063 | 0.9688 | 87s           |
| XGBoost     | 0.0450 | 0.9313 | 12.6s         |
| LightGBM    | 0.0460 | 0.9310 | 8.4s          |

**Hallazgos Clave:**  
- Campañas en Facebook e influencers generan mayor ROI (5.02 y 5.01 respectivamente)  
- Campañas por email logran mayores tasas de conversión (límite 15%)  
- Scores de engagement no mostraron correlación con métricas  
- Características predictivas principales: clicks, ingresos, costo-por-click, duración  

### 3. Dashboard de Power BI
**Componentes Principales:**  
1. **Vista General de Rendimiento**  
   - Resumen de inversión total vs ingresos generados  
   - Tendencias históricas y evolución del ROI  
   - Comparación de periodos clave  

2. **Análisis por Canales Digitales**  
   - Distribución de ingresos entre plataformas principales  
   - Comparativa de tasas de clics y conversiones  
   - Eficiencia relativa por canal de marketing  

3. **Segmentación Demográfica**  
   - Distribución de audiencias por género  
   - Patrones de comportamiento por grupos de edad  
   - Efectividad comparada entre segmentos  

4. **Perspectiva Geográfica**  
   - Rendimiento comparado por regiones  
   - Distribución de alcance e impresiones territoriales  
   - Identificación de mercados clave  

5. **Evaluación de Estrategias**  
   - Comparación entre tipos de campañas  
   - Análisis de formatos publicitarios  
   - Identificación de mejores prácticas  

**Funcionalidades Interactivas:**  
- Selectores de rangos de fechas personalizables  
- Capacidad de filtrar por múltiples dimensiones  
- Herramientas de exportación para análisis avanzado  
- Visualización adaptable a diferentes niveles de detalle  

**Beneficios Clave:**  
- Panel unificado para monitoreo integral  
- Identificación visual de patrones y anomalías  
- Soporte para toma de decisiones estratégicas  
- Actualización dinámica con nuevos datos  

### 4. Aplicación Streamlit
**Módulos:**  
1. **Explorador de Datos**  
   - Filtrado interactivo de datos  
   - Sistema de visualización por pestañas (ROI, conversiones, tendencias)  
   - Exportación a CSV  

2. **Predictor de ROI**  
   - Modelo Random Forest para predicción  
   - Análisis de escenarios con parámetros ajustables  
   - Cálculo de métricas  

3. **Aplicación Principal**  
   - Metadatos del modelo (versión, desempeño)  
   - Visualización de importancia de características  
   - Información de equipo/contacto  

## Resultados Clave e Impacto
**Mejoras en Calidad de Datos:**  
- Resueltas 180,101 discrepancias en tasas (90% de registros)  
- Implementación de límites de CTR por canal (Email:15%, Display:3%)  
- Puntaje de calidad mejorado de 51.7 a 92.8/100  

**Insights de Desempeño:**  
- Duración promedio de campaña: 37.5 días (rango:15-60)  
- 20% de campañas con ROI ≥6.81  
- Redes sociales y email contribuyen más ingresos  

**Recomendaciones:**  
1. Reasignar 20% de presupuesto a canales de alto ROI  
2. Priorizar campañas con influencers y Facebook  
3. Implementar detección automática de anomalías (CTR >30%)  
4. Revisar metodología de scores de engagement  

## Conclusión
Este proyecto ofrece una solución integral para análisis de campañas que:  
1. Automatiza procesos manuales  
2. Mejora precisión con machine learning (96.9% R²)  
3. Identifica oportunidades de optimización  
4. Se integra con herramientas existentes  

Proporciona valor especial para:  
- Analistas y científicos de datos  
- Gerentes de campañas  
- Planificadores financieros  

**Mejoras Futuras:**  
- Integración en tiempo real  
- Modelado avanzado de escenarios "what-if"  
- Alertas automáticas de desempeño  

**Explora los componentes:**  
- **Prueba el modelo [aquí](https://enncd44sfjc7sspt8yyeqr.streamlit.app/):**  
- **Ver el análisis de Power BI [aquí](https://github.com/Bokols/Analisis-de-Efectividad-y-ROI-de-Campanas-de-Marketing-/blob/master/Powerbi.pdf):**  
