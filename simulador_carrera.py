"""
=============================================================
  SIMULADOR DE CARRERA CON IA  -  Fase 1: Motor Base
  Pygame + Sensores + Física del carro
  
  INSTRUCCIONES:
  1. Instala dependencias:
       pip install pygame pillow numpy
  2. Pon este archivo en la MISMA carpeta que:
       carrera.jpeg   (la pista)
       carrito_.jpeg  (el carro)
  3. Ejecuta:
       python simulador_carrera.py
  
  CONTROLES MANUALES (para probar antes de la IA):
       W / Flecha Arriba  → Acelerar
       S / Flecha Abajo   → Frenar / Reversa
       A / Flecha Izq     → Girar izquierda
       D / Flecha Der     → Girar derecha
       R                  → Reiniciar posición
       ESC                → Salir
=============================================================
"""

import pygame
import numpy as np
import math
import sys
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────
ANCHO_VENTANA   = 800
ALTO_VENTANA    = 800
FPS             = 60
TITULO          = "Simulador IA - Circuito de Carreras"

# Colores utilitarios
COLOR_SENSOR    = (0,   255,  0)    # verde  → sensor libre
COLOR_PELIGRO   = (255,  80,  0)    # naranja → sensor cerca de borde
COLOR_COLISION  = (255,   0,  0)    # rojo   → fuera de pista
COLOR_HUD_BG    = (10,   10, 10)
COLOR_HUD_TEXT  = (220, 220, 220)
COLOR_AMARILLO  = (255, 220,  50)

# ─────────────────────────────────────────────
#  DETECCIÓN DE PISTA  (por color de píxel)
# ─────────────────────────────────────────────
def construir_mascara_pista(ruta_imagen: str, ancho: int, alto: int) -> np.ndarray:
    """
    Retorna una matriz booleana: True = asfalto (pista), False = fuera.
    Se basa en que el asfalto pixel-art es gris (canales R≈G≈B, tono frío).
    """
    img = Image.open(ruta_imagen).convert("RGB").resize((ancho, alto))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    calidez    = r - b                          # tierra/pasto = cálido (>20)
    brillo     = (r + g + b) / 3.0             # muy oscuro o muy claro = fuera

    mascara = (calidez < 20) & (brillo > 40) & (brillo < 165)
    return mascara  # shape (alto, ancho)


# ─────────────────────────────────────────────
#  CLASE: CARRO
# ─────────────────────────────────────────────
class Carro:
    # ---- Física ----
    ACELERACION     =  0.18
    FRENADO         =  0.22
    FRICCION        =  0.96   # multiplicador de velocidad cada frame
    VELOCIDAD_MAX   =  5.5
    VELOCIDAD_GIRO  =  3.2    # grados por frame a vel máx
    LARGO_SENSOR    = 120     # píxeles de alcance de cada sensor

    # Ángulos de los 5 sensores (relativo a la dirección del carro)
    ANGULOS_SENSOR  = [-90, -45, 0, 45, 90]

    def __init__(self, x: float, y: float, angulo: float, imagen: pygame.Surface):
        # Posición y orientación
        self.x       = x
        self.y       = y
        self.angulo  = angulo   # grados, 0 = derecha
        self.vel     = 0.0      # velocidad actual

        # Sprite original (sin rotar)
        self.imagen_orig = imagen
        self.imagen      = imagen
        self.rect        = self.imagen.get_rect(center=(int(x), int(y)))

        # Estado
        self.vivo        = True
        self.distancia   = 0.0  # distancia recorrida en pista
        self.lecturas    = [self.LARGO_SENSOR] * len(self.ANGULOS_SENSOR)

    # ── Movimiento ──────────────────────────────
    def actualizar(self, teclas, mascara: np.ndarray):
        if not self.vivo:
            return

        # Entrada (manual O desde IA más adelante)
        acelerando = teclas[pygame.K_w] or teclas[pygame.K_UP]
        frenando   = teclas[pygame.K_s] or teclas[pygame.K_DOWN]
        izquierda  = teclas[pygame.K_a] or teclas[pygame.K_LEFT]
        derecha    = teclas[pygame.K_d] or teclas[pygame.K_RIGHT]

        if acelerando:
            self.vel += self.ACELERACION
        if frenando:
            self.vel -= self.FRENADO

        self.vel  = max(-2.0, min(self.VELOCIDAD_MAX, self.vel))
        self.vel *= self.FRICCION

        # Giro proporcional a la velocidad (más velocidad → giro más ágil)
        factor_giro = (abs(self.vel) / self.VELOCIDAD_MAX) * self.VELOCIDAD_GIRO
        if izquierda:
            self.angulo -= factor_giro
        if derecha:
            self.angulo += factor_giro

        # Mover
        rad = math.radians(self.angulo)
        nuevo_x = self.x + math.cos(rad) * self.vel
        nuevo_y = self.y + math.sin(rad) * self.vel

        # Verificar colisión ANTES de mover
        nx, ny = int(nuevo_x), int(nuevo_y)
        alto, ancho = mascara.shape
        if 0 <= ny < alto and 0 <= nx < ancho:
            if mascara[ny, nx]:          # es asfalto → moverse
                self.x = nuevo_x
                self.y = nuevo_y
                self.distancia += abs(self.vel)
            else:                        # salió de pista
                self.vel *= 0.5          # penalizar velocidad
                # Permitir salir un poco pero marcar como muerto si es grave
                self.vivo = False
        else:
            self.vivo = False

        # Actualizar sprite rotado
        self.imagen = pygame.transform.rotate(self.imagen_orig, -self.angulo-90)
       
        self.rect   = self.imagen.get_rect(center=(int(self.x), int(self.y)))

        # Actualizar sensores
        self._actualizar_sensores(mascara)

    # ── Sensores ─────────────────────────────────
    def _actualizar_sensores(self, mascara: np.ndarray):
        alto, ancho = mascara.shape
        for i, offset_ang in enumerate(self.ANGULOS_SENSOR):
            ang_rad  = math.radians(self.angulo + offset_ang)
            distancia = self.LARGO_SENSOR   # distancia libre (máx)

            for d in range(1, self.LARGO_SENSOR + 1):
                px = int(self.x + math.cos(ang_rad) * d)
                py = int(self.y + math.sin(ang_rad) * d)
                if not (0 <= py < alto and 0 <= px < ancho):
                    distancia = d
                    break
                if not mascara[py, px]:  # encontró borde
                    distancia = d
                    break

            self.lecturas[i] = distancia

    def get_inputs_ia(self) -> list:
        """Devuelve las lecturas normalizadas [0..1] para la red neuronal."""
        return [d / self.LARGO_SENSOR for d in self.lecturas]

    # ── Dibujar ──────────────────────────────────
    def dibujar(self, pantalla: pygame.Surface):
        if not self.vivo:
            return
        # Carro
        pantalla.blit(self.imagen, self.rect)

        # Sensores
        for i, (offset_ang, distancia) in enumerate(zip(self.ANGULOS_SENSOR, self.lecturas)):
            ang_rad = math.radians(self.angulo + offset_ang)
            fin_x   = int(self.x + math.cos(ang_rad) * distancia)
            fin_y   = int(self.y + math.sin(ang_rad) * distancia)

            ratio = distancia / self.LARGO_SENSOR
            if ratio > 0.5:
                color = COLOR_SENSOR
            else:
                color = COLOR_PELIGRO

            pygame.draw.line(pantalla, color, (int(self.x), int(self.y)), (fin_x, fin_y), 1)
            pygame.draw.circle(pantalla, color, (fin_x, fin_y), 3)

    def reiniciar(self, x, y, angulo):
        self.x       = x
        self.y       = y
        self.angulo  = angulo
        self.vel     = 0.0
        self.vivo    = True
        self.distancia = 0.0
        self.lecturas  = [self.LARGO_SENSOR] * len(self.ANGULOS_SENSOR)


# ─────────────────────────────────────────────
#  HUD  (información en pantalla)
# ─────────────────────────────────────────────
def dibujar_hud(pantalla, fuente, carro, fps_real):
    # Fondo translúcido
    hud = pygame.Surface((220, 130), pygame.SRCALPHA)
    hud.fill((10, 10, 10, 170))
    pantalla.blit(hud, (8, 8))

    lineas = [
        f"FPS       : {fps_real:.0f}",
        f"Velocidad : {abs(carro.vel):.2f}",
        f"Ángulo    : {carro.angulo % 360:.1f}°",
        f"Distancia : {carro.distancia:.0f} px",
        f"Sensores  : {[int(l) for l in carro.lecturas]}",
        f"Estado    : {'EN PISTA' if carro.vivo else '💥 FUERA'}",
    ]
    for i, linea in enumerate(lineas):
        color = COLOR_HUD_TEXT if carro.vivo else COLOR_COLISION
        if "FUERA" in linea:
            color = COLOR_COLISION
        surf = fuente.render(linea, True, color)
        pantalla.blit(surf, (14, 14 + i * 19))

def dibujar_titulo_controles(pantalla, fuente_peq):
    info = "[W/S] Acelerar/Frenar  [A/D] Girar  [R] Reiniciar  [ESC] Salir"
    surf = fuente_peq.render(info, True, (180, 180, 180))
    pantalla.blit(surf, (ANCHO_VENTANA//2 - surf.get_width()//2, ALTO_VENTANA - 22))

def dibujar_pantalla_colision(pantalla, fuente_grande, fuente):
    overlay = pygame.Surface((ANCHO_VENTANA, ALTO_VENTANA), pygame.SRCALPHA)
    overlay.fill((180, 0, 0, 80))
    pantalla.blit(overlay, (0, 0))
    txt  = fuente_grande.render("¡FUERA DE PISTA!", True, (255, 60, 60))
    txt2 = fuente.render("Presiona  R  para reiniciar", True, (255, 220, 220))
    pantalla.blit(txt,  (ANCHO_VENTANA//2 - txt.get_width()//2,  ALTO_VENTANA//2 - 30))
    pantalla.blit(txt2, (ANCHO_VENTANA//2 - txt2.get_width()//2, ALTO_VENTANA//2 + 20))


# ─────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────
def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
    pygame.display.set_caption(TITULO)
    reloj   = pygame.time.Clock()

    # ── Fuentes ──
    fuente       = pygame.font.SysFont("Consolas", 14)
    fuente_peq   = pygame.font.SysFont("Consolas", 12)
    fuente_grande= pygame.font.SysFont("Consolas", 28, bold=True)

    # ── Cargar pista ──
    try:
        fondo_orig = pygame.image.load("carrera.jpeg").convert()
    except FileNotFoundError:
        print("ERROR: No se encontró 'carrera.jpeg' en la carpeta actual.")
        print("       Asegúrate de que el archivo esté junto a este script.")
        sys.exit(1)

    fondo = pygame.transform.scale(fondo_orig, (ANCHO_VENTANA, ALTO_VENTANA))

    print("Construyendo máscara de pista... (puede tardar unos segundos)")
    mascara = construir_mascara_pista("carrera.jpeg", ANCHO_VENTANA, ALTO_VENTANA)
    print(f"  → Pista detectada: {100*mascara.sum()/(ANCHO_VENTANA*ALTO_VENTANA):.1f}% del área")

    # ── Cargar carro ──
    try:
        img_carro_orig = pygame.image.load("carrito_.jpeg").convert_alpha()
    except FileNotFoundError:
        # Si no carga con transparencia, intentar sin alpha
        try:
            img_carro_orig = pygame.image.load("carrito_.jpeg").convert()
        except FileNotFoundError:
            print("ERROR: No se encontró 'carrito_.jpeg' en la carpeta actual.")
            sys.exit(1)

    # Escalar carro a tamaño apropiado para la pista
    tam_carro = 35
    img_carro = pygame.transform.scale(img_carro_orig, (35, 38))

    # Hacer fondo gris del JPG transparente (el carrito tiene fondo gris claro)
    img_carro = img_carro.convert_alpha()
    arr_carro = pygame.surfarray.pixels3d(img_carro)
    alpha_arr = pygame.surfarray.pixels_alpha(img_carro)
    # Píxeles grises claros → transparentes
    es_fondo = (
        (arr_carro[:,:,0].astype(int) > 180) &
        (arr_carro[:,:,1].astype(int) > 180) &
        (arr_carro[:,:,2].astype(int) > 180) &
        (np.abs(arr_carro[:,:,0].astype(int) - arr_carro[:,:,1].astype(int)) < 20) &
        (np.abs(arr_carro[:,:,1].astype(int) - arr_carro[:,:,2].astype(int)) < 20)
    )
    alpha_arr[es_fondo] = 0
    del arr_carro, alpha_arr  # liberar locks

    # ── Posición inicial del carro (entrada de pista, lado derecho) ──
    # Ajusta estos valores si el carro aparece fuera de la pista
    START_X    = 610
    START_Y    = 430
    START_ANG  = 270   # apuntando hacia arriba

    carro = Carro(START_X, START_Y, START_ANG, img_carro)

    # ─────────────────────────────────────────
    #  LOOP PRINCIPAL
    # ─────────────────────────────────────────
    ejecutando = True
    while ejecutando:
        dt = reloj.tick(FPS)
        fps_real = reloj.get_fps()

        # ── Eventos ──
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    ejecutando = False
                if evento.key == pygame.K_r:
                    carro.reiniciar(START_X, START_Y, START_ANG)

        # ── Actualizar ──
        teclas = pygame.key.get_pressed()
        carro.actualizar(teclas, mascara)

        # ── Dibujar ──
        pantalla.blit(fondo, (0, 0))    # 1. Pista de fondo
        carro.dibujar(pantalla)         # 2. Carro + sensores

        # 3. HUD
        dibujar_hud(pantalla, fuente, carro, fps_real)
        dibujar_titulo_controles(pantalla, fuente_peq)

        # 4. Overlay de colisión
        if not carro.vivo:
            dibujar_pantalla_colision(pantalla, fuente_grande, fuente)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()