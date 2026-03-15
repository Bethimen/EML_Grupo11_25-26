# Aprendizaje por Refuerzo

## Información
- **Alumnos:** Vicente García, Víctor Emilio; Guirado Pérez, Jesús; Sánchez Torres, Antonio Luis
- **Curso:** 2025/2026
- **NombreGrupo:** EML_Grupo11_25-26

## Descripción
Este trabajo se enmarca en la evaluación de la asignatura de Aprendizaje por Refuerzo del Máster en Inteligencia Artificial. Tiene como objetivo evaluar las distintas estrategias de bandidos sobre distintas distribuciones. Analizar el desempeño de métodos tabulares en el entorno de Blackjack y analizar el desempeño de métodos aproximados sobre el entorno de Flappy Bird.

## Estructura
En la estructura raíz hay localizado un main.ipynb que se puede abrir desde colab, hará un clon del repositorio de prácticas de nuestro grupo y ejecutará todos los experimentos con los requeriments necesarios.

En eml_k_bandit-main está toda la parte del bandido de k-brazos, en donde en esa carpeta están los tres notebooks con los 3 experimentos, llamados por el tipo de bandido utilizado para cada experimento: Normal, Binomial y Bernoulli. La subcarpeta algorithms está todos los algoritmos utilizados para los experimentos, la subcarpeta arms para todos los tipos de brazos y subcarpeta plotting con el script para las gráficas plotting.py.

En el notebook RL_MetodosTabulares.ipynb está el la parte 2 de métodos tabulares

En la carpeta Entornos_continuos esta la parte 3, el notebook Entornos_continuos\parte3_evaluación.ipynb es el punto de entrada, y la carpeta Entornos_continuos\src contiene clases auxiliares para la ejecución del notebook como los agentes en \agents.
Por último resultados intermedios como configuración de los modelos o videos se guardan en la carpeta \Entornos_continuos\Entornos_continuos\artifacts

## Instalación y uso
Para instalar el entorno necesario para la ejecución de los notebooks se utiliza el main.py que en su primera celda se descarga el repositorio de github e instala las librerías necesarias del requirements.txt

## Tecnologías utilizadas
python, python notebook, markdown