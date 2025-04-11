from PIL import Image

# Cargar la misma imagen dos veces
image = Image.open('src/prueba1.jpg')           # Imagen base
imagenConBordes = Image.open('src/prueba1.jpg')  # Imagen con bordes

width = image.size[0] - 1
height = image.size[1] - 1

for i in range(0, width):  # Lee todos los pixeles de la imagen
    for j in range(0, height):
        aqui = imagenConBordes.getpixel((i, j))
        abajo = image.getpixel((i, j + 1))
        derecha = image.getpixel((i + 1, j))

        aquiLuz = (aqui[0] + aqui[1] + aqui[2]) / 3
        abajoLuz = (abajo[0] + abajo[1] + abajo[2]) / 3
        derechaLuz = (derecha[0] + derecha[1] + derecha[2]) / 3
        restaX = abs(aquiLuz - derechaLuz)
        restaY = abs(aquiLuz - abajoLuz)

        color = int(aquiLuz - 1)
        if (restaX + restaY) < 10:
            image.putpixel((i, j), (0, 0, 0))
        else:
            image.putpixel((i, j), (color, color, color))

# Guarda la imagen procesada
image.save("src/sincolor.jpg")
