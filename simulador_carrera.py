"""
=============================================================
  SIMULADOR DE CARRERA CON IA  -  Fase 1: Motor Base
  CONTROLES:
       W / ↑   → Acelerar       S / ↓  → Frenar
       A / ←   → Girar izq      D / →  → Girar der
       C       → Ver/Ocultar contorno de pista
       R       → Reiniciar      ESC    → Salir
=============================================================
"""

import pygame
import numpy as np
import math
import sys
from PIL import Image
from scipy.ndimage import binary_erosion, label

ANCHO_VENTANA  = 1024
ALTO_VENTANA   = 1024
FPS            = 60
TITULO         = "Simulador IA - Circuito de Carreras"

COLOR_SENSOR_OK = (0,  255,   0)
COLOR_PELIGRO   = (255, 140,  0)
COLOR_COLISION  = (255,   0,  0)
COLOR_HUD_TEXT  = (220, 220, 220)

START_X   = 610
START_Y   = 430
START_ANG = 270


def construir_mascara(ruta, ancho, alto):
    img = Image.open(ruta).convert("RGB").resize((ancho, alto))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    calidez = r - b
    brillo  = (r + g + b) / 3.0
    cruda   = (calidez < 20) & (brillo > 25) & (brillo < 165)
    etiquetada, _ = label(cruda)
    tamanios = np.bincount(etiquetada.ravel())
    tamanios[0] = 0
    region_pista = tamanios.argmax()
    return etiquetada == region_pista


def construir_contorno(mascara):
    alto, ancho = mascara.shape
    erosionada  = binary_erosion(mascara, iterations=3)
    borde       = mascara & ~erosionada

    surf = pygame.Surface((ancho, alto), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))   # todo transparente

    # Acceder a los arrays de color y alpha por separado (funciona con SRCALPHA)
    px_rgb   = pygame.surfarray.pixels3d(surf)       # shape (ancho, alto, 3)
    px_alpha = pygame.surfarray.pixels_alpha(surf)   # shape (ancho, alto)

    # Transponer mascaras de (alto,ancho) a (ancho,alto) para que coincidan
    fuera_t = (~mascara).T
    borde_t = borde.T

    # Zona fuera de pista: rojo tenue
    px_rgb[fuera_t, 0] = 180
    px_rgb[fuera_t, 1] = 0
    px_rgb[fuera_t, 2] = 0
    px_alpha[fuera_t]  = 40

    # Borde exacto: rojo brillante
    px_rgb[borde_t, 0] = 255
    px_rgb[borde_t, 1] = 30
    px_rgb[borde_t, 2] = 30
    px_alpha[borde_t]  = 220

    # Liberar locks antes de retornar
    del px_rgb, px_alpha

    return surf


class Carro:
    ACELERACION    = 0.18
    FRENADO        = 0.22
    FRICCION       = 0.96
    VELOCIDAD_MAX  = 5.5
    VELOCIDAD_GIRO = 3.2
    LARGO_SENSOR   = 130
    ANGULOS_SENSOR = [-90, -45, 0, 45, 90]

    def __init__(self, x, y, angulo, imagen):
        self.x = float(x); self.y = float(y)
        self.angulo = float(angulo); self.vel = 0.0
        self.imagen_orig = imagen; self.imagen = imagen
        self.rect = imagen.get_rect(center=(int(x), int(y)))
        self.vivo = True; self.distancia = 0.0
        self.lecturas = [self.LARGO_SENSOR] * len(self.ANGULOS_SENSOR)

    def actualizar(self, teclas, mascara):
        if not self.vivo:
            return
        acel = teclas[pygame.K_w] or teclas[pygame.K_UP]
        fren = teclas[pygame.K_s] or teclas[pygame.K_DOWN]
        izq  = teclas[pygame.K_a] or teclas[pygame.K_LEFT]
        der  = teclas[pygame.K_d] or teclas[pygame.K_RIGHT]
        if acel: self.vel += self.ACELERACION
        if fren: self.vel -= self.FRENADO
        self.vel = max(-2.0, min(self.VELOCIDAD_MAX, self.vel))
        self.vel *= self.FRICCION
        giro = (abs(self.vel) / self.VELOCIDAD_MAX) * self.VELOCIDAD_GIRO
        if izq: self.angulo -= giro
        if der: self.angulo += giro
        rad = math.radians(self.angulo)
        nuevo_x = self.x + math.cos(rad) * self.vel
        nuevo_y = self.y + math.sin(rad) * self.vel
        nx, ny = int(nuevo_x), int(nuevo_y)
        alto, ancho = mascara.shape
        if 0 <= ny < alto and 0 <= nx < ancho:
            if mascara[ny, nx]:
                self.x = nuevo_x; self.y = nuevo_y
                self.distancia += abs(self.vel)
            else:
                self.vel *= 0.4; self.vivo = False
        else:
            self.vivo = False
        self.imagen = pygame.transform.rotate(self.imagen_orig, -self.angulo + 90)
        self.rect   = self.imagen.get_rect(center=(int(self.x), int(self.y)))
        self._sensores(mascara)

    def _sensores(self, mascara):
        alto, ancho = mascara.shape
        for i, offset in enumerate(self.ANGULOS_SENSOR):
            rad  = math.radians(self.angulo + offset)
            dist = self.LARGO_SENSOR
            for d in range(1, self.LARGO_SENSOR + 1):
                px = int(self.x + math.cos(rad) * d)
                py = int(self.y + math.sin(rad) * d)
                if not (0 <= py < alto and 0 <= px < ancho):
                    dist = d; break
                if not mascara[py, px]:
                    dist = d; break
            self.lecturas[i] = dist

    def get_inputs_ia(self):
        return [d / self.LARGO_SENSOR for d in self.lecturas]

    def dibujar(self, pantalla):
        if not self.vivo:
            return
        pantalla.blit(self.imagen, self.rect)
        for offset, dist in zip(self.ANGULOS_SENSOR, self.lecturas):
            rad   = math.radians(self.angulo + offset)
            fx    = int(self.x + math.cos(rad) * dist)
            fy    = int(self.y + math.sin(rad) * dist)
            color = COLOR_SENSOR_OK if dist / self.LARGO_SENSOR > 0.4 else COLOR_PELIGRO
            pygame.draw.line(pantalla, color, (int(self.x), int(self.y)), (fx, fy), 1)
            pygame.draw.circle(pantalla, color, (fx, fy), 3)

    def reiniciar(self, x, y, angulo):
        self.x = float(x); self.y = float(y)
        self.angulo = float(angulo); self.vel = 0.0
        self.vivo = True; self.distancia = 0.0
        self.lecturas = [self.LARGO_SENSOR] * len(self.ANGULOS_SENSOR)


def dibujar_hud(pantalla, fuente, fuente_peq, carro, fps, contorno_on):
    bg = pygame.Surface((250, 148), pygame.SRCALPHA)
    bg.fill((10, 10, 10, 175))
    pantalla.blit(bg, (8, 8))
    lineas = [
        f"FPS        : {fps:.0f}",
        f"Velocidad  : {abs(carro.vel):.2f}",
        f"Angulo     : {carro.angulo % 360:.1f} grados",
        f"Distancia  : {carro.distancia:.0f} px",
        f"Sensores   : {[int(l) for l in carro.lecturas]}",
        f"Estado     : {'EN PISTA' if carro.vivo else 'FUERA'}",
        f"Contorno   : {'ON  [C]' if contorno_on else 'OFF [C]'}",
    ]
    for i, lin in enumerate(lineas):
        color = (255, 80, 80) if "FUERA" in lin else COLOR_HUD_TEXT
        pantalla.blit(fuente.render(lin, True, color), (14, 14 + i * 19))
    ayuda = "[W/S] Accel/Freno   [A/D] Girar   [C] Contorno   [R] Reset   [ESC] Salir"
    pantalla.blit(fuente_peq.render(ayuda, True, (150, 150, 150)),
                  (ANCHO_VENTANA // 2 - 290, ALTO_VENTANA - 20))


def dibujar_game_over(pantalla, fg, f):
    ov = pygame.Surface((ANCHO_VENTANA, ALTO_VENTANA), pygame.SRCALPHA)
    ov.fill((160, 0, 0, 70))
    pantalla.blit(ov, (0, 0))
    t1 = fg.render("FUERA DE PISTA!", True, (255, 60, 60))
    t2 = f.render("Presiona  R  para reiniciar", True, (255, 210, 210))
    pantalla.blit(t1, (ANCHO_VENTANA//2 - t1.get_width()//2, ALTO_VENTANA//2 - 30))
    pantalla.blit(t2, (ANCHO_VENTANA//2 - t2.get_width()//2, ALTO_VENTANA//2 + 22))


def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
    pygame.display.set_caption(TITULO)
    reloj = pygame.time.Clock()

    fuente     = pygame.font.SysFont("Consolas", 13)
    fuente_peq = pygame.font.SysFont("Consolas", 11)
    fuente_gde = pygame.font.SysFont("Consolas", 30, bold=True)

    try:
        fondo = pygame.image.load("carrera.jpeg").convert()
        fondo = pygame.transform.scale(fondo, (ANCHO_VENTANA, ALTO_VENTANA))
    except FileNotFoundError:
        print("ERROR: No se encontro 'carrera.jpeg'"); sys.exit(1)

    print("Analizando pista... espera un momento")
    mascara = construir_mascara("carrera.jpeg", ANCHO_VENTANA, ALTO_VENTANA)
    print(f"  Pista detectada: {100*mascara.sum()/(ANCHO_VENTANA*ALTO_VENTANA):.1f}%")

    print("Generando contorno visual...")
    contorno_surf = construir_contorno(mascara)
    print("  Listo! Presiona C para mostrar/ocultar el contorno")

    try:
        img_raw = pygame.image.load("carrito_.jpeg").convert_alpha()
    except Exception:
        try:
            img_raw = pygame.image.load("carrito_.jpeg").convert()
        except FileNotFoundError:
            print("ERROR: No se encontro 'carrito_.jpeg'"); sys.exit(1)

    img_carro = pygame.transform.scale(img_raw, (35, 38))
    img_carro = img_carro.convert_alpha()
    ac = pygame.surfarray.pixels3d(img_carro)
    aa = pygame.surfarray.pixels_alpha(img_carro)
    fondo_gris = (
        (ac[:,:,0].astype(int) > 175) &
        (ac[:,:,1].astype(int) > 175) &
        (ac[:,:,2].astype(int) > 175) &
        (np.abs(ac[:,:,0].astype(int) - ac[:,:,1].astype(int)) < 25) &
        (np.abs(ac[:,:,1].astype(int) - ac[:,:,2].astype(int)) < 25)
    )
    aa[fondo_gris] = 0
    del ac, aa

    carro = Carro(START_X, START_Y, START_ANG, img_carro)
    mostrar_contorno = True

    ejecutando = True
    while ejecutando:
        reloj.tick(FPS)
        fps_real = reloj.get_fps()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: ejecutando = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: ejecutando = False
                if ev.key == pygame.K_r:      carro.reiniciar(START_X, START_Y, START_ANG)
                if ev.key == pygame.K_c:      mostrar_contorno = not mostrar_contorno

        carro.actualizar(pygame.key.get_pressed(), mascara)
        pantalla.blit(fondo, (0, 0))
        if mostrar_contorno:
            pantalla.blit(contorno_surf, (0, 0))
        carro.dibujar(pantalla)
        dibujar_hud(pantalla, fuente, fuente_peq, carro, fps_real, mostrar_contorno)
        if not carro.vivo:
            dibujar_game_over(pantalla, fuente_gde, fuente)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()