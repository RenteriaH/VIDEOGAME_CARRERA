# Simulador de carreras 2D (pygame)

Juego de carreras en Python: eliges pista y coche y corres un número fijo de vueltas evitando obstáculos.
La pista no se dibuja a mano: se **calcula a partir de una imagen**, y esa misma máscara decide dónde puede circular el coche.

## Cómo funciona

| Parte | Implementación |
|---|---|
| Pista | La imagen del mapa se convierte en una máscara binaria (NumPy); `scipy.ndimage` obtiene el contorno y separa regiones |
| Colisiones | El coche solo avanza sobre píxeles de pista; los obstáculos son círculos generados al azar dentro de zonas seguras de la máscara |
| Sensores | Cinco rayos de distancia desde el coche hasta el borde de la pista, mostrados en el HUD |
| Estados | Menú → selección de pista y coche → carrera → fin de la partida |
| Contenido | 3 pistas y 8 coches (algunos con tinte de color aplicado con Pillow) |

## Ejecutar

Requiere Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulador_carrera.py
```

## Controles

| Tecla | Acción |
|---|---|
| W / ↑ | Acelerar |
| S / ↓ | Frenar |
| A / ← · D / → | Girar |
| C | Mostrar u ocultar el contorno de la pista |
| R | Reiniciar |
| ESC | Volver al menú |

## Limitaciones

- El coche se controla con el teclado: los sensores miden distancias, pero todavía no hay un piloto automático que las use.
- Todo el código está en un solo archivo (`simulador_carrera.py`, ~840 líneas).
- Algunas imágenes de coches son de marcas comerciales y se usan solo con fines académicos.

## Tecnologías

Python · pygame · NumPy · SciPy · Pillow
