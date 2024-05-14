import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.01, max_epocas=1000, criterio_parada='exactitud'):
        self.pesos = np.random.rand(num_entradas + 1)  # +1 para el sesgo
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_epocas = max_epocas
        self.criterio_parada = criterio_parada

    def activacion(self, x):
        return 1 if x >= 0 else -1

    def predecir(self, entradas):
        suma = np.dot(entradas, self.pesos[1:]) + self.pesos[0]  # Agregar sesgo
        return self.activacion(suma)

    def entrenar(self, entradas_entrenamiento, etiquetas):
        if self.criterio_parada == 'exactitud':
            for _ in range(self.max_epocas):
                for entradas, etiqueta in zip(entradas_entrenamiento, etiquetas):
                    prediccion = self.predecir(entradas)
                    self.pesos[1:] += self.tasa_aprendizaje * (etiqueta - prediccion) * entradas
                    self.pesos[0] += self.tasa_aprendizaje * (etiqueta - prediccion)  # Actualizar sesgo
                if all(self.predecir(entradas) == etiqueta for entradas, etiqueta in zip(entradas_entrenamiento, etiquetas)):
                    break
        elif self.criterio_parada == 'epocas':
            for epoca in range(self.max_epocas):
                errores = 0
                for entradas, etiqueta in zip(entradas_entrenamiento, etiquetas):
                    prediccion = self.predecir(entradas)
                    error = etiqueta - prediccion
                    self.pesos[1:] += self.tasa_aprendizaje * error * entradas
                    self.pesos[0] += self.tasa_aprendizaje * error  # Actualizar sesgo
                    errores += int(error != 0)
                if errores == 0:
                    break

    def probar(self, entradas_prueba, etiquetas_prueba):
        aciertos = 0
        for entradas, etiqueta in zip(entradas_prueba, etiquetas_prueba):
            prediccion = self.predecir(entradas)
            if prediccion == etiqueta:
                aciertos += 1
        return aciertos / len(entradas_prueba)

def leer_datos(nombre_archivo):
    datos = np.genfromtxt(nombre_archivo, delimiter=',')
    entradas = datos[:, :-1]
    etiquetas = datos[:, -1]
    return entradas, etiquetas

def graficar_datos(entradas, etiquetas, perceptron=None):
    plt.scatter(entradas[:, 0], entradas[:, 1], c=etiquetas)
    if perceptron:
        plt.plot([-1, 1], [(perceptron.pesos[0] + perceptron.pesos[1] * (-1)) / -perceptron.pesos[2],
                          (perceptron.pesos[0] + perceptron.pesos[1] * 1) / -perceptron.pesos[2]], 'k-')
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.title('Conjunto de Datos XOR')
    plt.show()

if __name__ == "__main__":
    # Selección de parámetros
    criterio_parada = input("Seleccione el criterio de parada (exactitud/epocas): ").lower()
    max_epocas = int(input("Seleccione el número máximo de épocas de entrenamiento: "))
    tasa_aprendizaje = float(input("Seleccione la tasa de aprendizaje: "))

    # Lectura de datos de entrenamiento
    entradas_entrenamiento, etiquetas_entrenamiento = leer_datos("XORtrn.csv")
    # Lectura de datos de prueba
    entradas_prueba, etiquetas_prueba = leer_datos("XORtst.csv")

    # Creación y entrenamiento del perceptrón
    perceptron = Perceptron(num_entradas=2, tasa_aprendizaje=tasa_aprendizaje, max_epocas=max_epocas, criterio_parada=criterio_parada)
    perceptron.entrenar(entradas_entrenamiento, etiquetas_entrenamiento)

    # Visualización de los datos de entrenamiento y la recta que los separa
    graficar_datos(entradas_entrenamiento, etiquetas_entrenamiento, perceptron)

    # Prueba del perceptrón en datos reales
    exactitud = perceptron.probar(entradas_prueba, etiquetas_prueba)
    print("Exactitud del perceptrón en datos de prueba:", exactitud)


