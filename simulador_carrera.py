"""
=============================================================
  SIMULADOR DE CARRERA CON IA  -  Con Pantalla de Inicio
  ESTADOS:
    MENU     -> Pantalla principal con título y opciones
    SELECCION -> Elegir mapa y carro
    JUEGO    -> Simulación normal

  CONTROLES EN JUEGO:
       W / ↑   → Acelerar       S / ↓  → Frenar
       A / ←   → Girar izq      D / →  → Girar der
       C       → Ver/Ocultar contorno de pista
       R       → Reiniciar      ESC    → Volver al menú
=============================================================
"""

import sys
import pygame
import numpy as np
import math
import random
from PIL import Image
from scipy.ndimage import binary_erosion, label


# ===================== CONFIG =====================
ANCHO_VENTANA  = 720
ALTO_VENTANA   = 720
FPS            = 60
TITULO         = "Simulador IA - Circuito de Carreras"

COLOR_SENSOR_OK = (0,  255,   0)
COLOR_PELIGRO   = (255, 140,  0)
COLOR_COLISION  = (255,   0,  0)
COLOR_HUD_TEXT  = (220, 220, 220)

TOTAL_VUELTAS = 3

# Estados del juego
ESTADO_MENU      = "menu"
ESTADO_SELECCION = "seleccion"
ESTADO_JUEGO     = "juego"

# Lista de mapas
MAPAS = [
    {
        "img": "carrera.jpeg",
        "nombre": "Circuito Clásico",
        "start": (610, 430, 270),
        "meta": pygame.Rect(496, 359, 160, 20),
        "checkpoint": pygame.Rect(100, 600, 20, 200)
    },
    {
        "img": "carrera2.jpeg",
        "nombre": "Circuito Urbano",
        "start": (610, 430, 270),
        "meta": pygame.Rect(496, 359, 160, 20),
        "checkpoint": pygame.Rect(100, 600, 20, 200)
    },
    {
        "img": "carrera3.png",
        "nombre": "Nafcard Speedway",
        "start": (610, 430, 270),
        "meta": pygame.Rect(496, 359, 160, 20),
        "checkpoint": pygame.Rect(100, 600, 20, 200)
    },
]

# Lista de carros (archivos de imagen)
CARROS_IMAGENES = [
    {"img": "carrito_.jpeg", "nombre": "Rojo Clásico"},
    {"img": "carrito_.jpeg", "nombre": "Azul Turbo"},   # mismo archivo, distinto color tinte
    {"img": "carrito_.jpeg", "nombre": "Verde Nitro"},
    {"img": "Cadillac.png", "nombre": "Cadillac"},
    {"img": "Octano.png", "nombre": "Ocatano"},
    {"img": "Vocho.png", "nombre": "Vocho Cadillac"},
    {"img": "Red_Bull.png", "nombre": "Red Bull"},
    {"img": "Ferrari.png", "nombre": "Ferrari"},




]

TINTES_CARROS = [
    None,          # sin tinte (original)
    (50, 100, 220),  # azul
    (50, 180, 60),   # verde
]

# ===================== OBSTÁCULOS =====================
OBSTACULOS = []


def carro_choca_obstaculo(carro, obstaculos):
    cx, cy = carro.x, carro.y
    car_r = 10
    for o in obstaculos:
        dx = cx - o["x"]
        dy = cy - o["y"]
        if math.hypot(dx, dy) < (car_r + o["r"]):
            return True
    return False


def dibujar_obstaculos(pantalla, obstaculos):
    for o in obstaculos:
        x, y, r = int(o["x"]), int(o["y"]), int(o["r"])
        pygame.draw.circle(pantalla, (0, 0, 0), (x, y + 4), int(r * 1.10))
        pygame.draw.circle(pantalla, (255, 140, 0), (x, y), r)
        pygame.draw.circle(pantalla, (20, 20, 20), (x, y), r, 2)
        pygame.draw.circle(pantalla, (245, 245, 245), (x, y), max(2, r // 4))


def generar_obstaculos_en_pista(mascara, cantidad=10, radio=15, margen=8, min_sep=8, max_intentos=6000):
    alto, ancho = mascara.shape
    nuevos = []
    intentos = 0

    def zona_es_segura(x, y):
        if not mascara[y, x]:
            return False
        check_d = r + margen
        for ang in (0, 45, 90, 135, 180, 225, 270, 315):
            rad = math.radians(ang)
            cx = int(x + math.cos(rad) * check_d)
            cy = int(y + math.sin(rad) * check_d)
            if not (0 <= cy < alto and 0 <= cx < ancho):
                return False
            if not mascara[cy, cx]:
                return False
        for o in nuevos:
            if math.hypot(x - o["x"], y - o["y"]) < (radio + o["r"] + min_sep):
                return False
        return True

    while len(nuevos) < cantidad and intentos < max_intentos:
        intentos += 1
        r = radio
        x = random.randint(r + 2, ancho - r - 3)
        y = random.randint(r + 2, alto - r - 3)
        if zona_es_segura(x, y):
            nuevos.append({"x": x, "y": y, "r": r})

    return nuevos


# ===================== PISTA =====================
def construir_mascara(ruta, ancho, alto):
    img = Image.open(ruta).convert("RGB").resize((ancho, alto))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    calidez = r - b
    brillo  = (r + g + b) / 3.0
    cruda = (calidez < 20) & (brillo > 15) & (brillo < 240)
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
    surf.fill((0, 0, 0, 0))
    px_rgb   = pygame.surfarray.pixels3d(surf)
    px_alpha = pygame.surfarray.pixels_alpha(surf)
    fuera_t = (~mascara).T
    borde_t = borde.T
    px_rgb[fuera_t, 0] = 180
    px_rgb[fuera_t, 1] = 0
    px_rgb[fuera_t, 2] = 0
    px_alpha[fuera_t]  = 40
    px_rgb[borde_t, 0] = 255
    px_rgb[borde_t, 1] = 30
    px_rgb[borde_t, 2] = 30
    px_alpha[borde_t]  = 220
    del px_rgb, px_alpha
    return surf

#holaaa
def cargar_mapa(mapa, ancho, alto):
    ruta = mapa["img"]
    try:
        fondo = pygame.image.load(ruta).convert()
    except FileNotFoundError:
        print(f"ERROR: No se encontró el mapa '{ruta}'.")
        raise
    fondo = pygame.transform.scale(fondo, (ancho, alto))
    print(f"Analizando pista: {ruta} ...")
    mascara = construir_mascara(ruta, ancho, alto)
    print(f"  Pista detectada: {100*mascara.sum()/(ancho*alto):.1f}%")
    print("Generando contorno visual...")
    contorno_surf = construir_contorno(mascara)
    print("  Listo!")
    sx, sy, sang = mapa["start"]
    return fondo, mascara, contorno_surf, sx, sy, sang


def cargar_imagen_carro(ruta, tinte=None):
    try:
        img_raw = pygame.image.load(ruta).convert_alpha()
    except Exception:
        try:
            img_raw = pygame.image.load(ruta).convert()
        except FileNotFoundError:
            print(f"ERROR: No se encontró '{ruta}'")
            sys.exit(1)

    img = pygame.transform.smoothscale(img_raw, (60, 60)).convert_alpha()

    # Quitar fondo gris/blanco
    ac = pygame.surfarray.pixels3d(img)
    aa = pygame.surfarray.pixels_alpha(img)
    fondo_gris = (
        (ac[:, :, 0].astype(int) > 175) &
        (ac[:, :, 1].astype(int) > 175) &
        (ac[:, :, 2].astype(int) > 175) &
        (np.abs(ac[:, :, 0].astype(int) - ac[:, :, 1].astype(int)) < 25) &
        (np.abs(ac[:, :, 1].astype(int) - ac[:, :, 2].astype(int)) < 25)
    )
    aa[fondo_gris] = 0

    # Aplicar tinte si hay
    if tinte is not None:
        visible = aa > 10
        ac[visible, 0] = np.clip(ac[visible, 0].astype(int) * tinte[0] // 200, 0, 255).astype(np.uint8)
        ac[visible, 1] = np.clip(ac[visible, 1].astype(int) * tinte[1] // 200, 0, 255).astype(np.uint8)
        ac[visible, 2] = np.clip(ac[visible, 2].astype(int) * tinte[2] // 200, 0, 255).astype(np.uint8)

    del ac, aa
    return img


# ===================== CARRO =====================
class Carro:
    ACELERACION    = 0.18
    FRENADO        = 0.22
    FRICCION       = 0.96
    VELOCIDAD_MAX  = 5.5
    VELOCIDAD_GIRO = 3.2
    LARGO_SENSOR   = 130
    ANGULOS_SENSOR = [-90, -45, 0, 45, 90]

    def __init__(self, x, y, angulo, imagen):
        self.x = float(x)
        self.y = float(y)
        self.angulo = float(angulo)
        self.vel = 0.0
        self.imagen_orig = imagen
        self.imagen = imagen
        self.rect = imagen.get_rect(center=(int(x), int(y)))
        self.vivo = True
        self.distancia = 0.0
        self.lecturas = [self.LARGO_SENSOR] * len(self.ANGULOS_SENSOR)
        self.vueltas = 0
        self.paso_checkpoint = False
        self.tiempo_vuelta_visible = FPS * 3  # muestra "VUELTA 1" al arrancar
        self.meta_cooldown = int(FPS * 1.5)  # evita contar meta al arrancar

    def actualizar(self, teclas, mascara):
        if not self.vivo:
            return

        acel = teclas[pygame.K_w] or teclas[pygame.K_UP]
        fren = teclas[pygame.K_s] or teclas[pygame.K_DOWN]
        izq  = teclas[pygame.K_a] or teclas[pygame.K_LEFT]
        der  = teclas[pygame.K_d] or teclas[pygame.K_RIGHT]

        if acel:
            self.vel += self.ACELERACION
        if fren:
            self.vel -= self.FRENADO

        self.vel = max(-2.0, min(self.VELOCIDAD_MAX, self.vel))
        self.vel *= self.FRICCION

        giro = (abs(self.vel) / self.VELOCIDAD_MAX) * self.VELOCIDAD_GIRO
        if izq:
            self.angulo -= giro
        if der:
            self.angulo += giro

        rad = math.radians(self.angulo)
        nuevo_x = self.x + math.cos(rad) * self.vel
        nuevo_y = self.y + math.sin(rad) * self.vel

        nx, ny = int(nuevo_x), int(nuevo_y)
        alto, ancho = mascara.shape

        if 0 <= ny < alto and 0 <= nx < ancho:
            if mascara[ny, nx]:
                self.x = nuevo_x
                self.y = nuevo_y
                self.distancia += abs(self.vel)
            else:
                self.vel *= 0.4
                self.vivo = False
        else:
            self.vivo = False

        self.imagen = pygame.transform.rotate(self.imagen_orig, -self.angulo - 90)
        self.rect   = self.imagen.get_rect(center=(int(self.x), int(self.y)))
        self._sensores(mascara)

        if self.tiempo_vuelta_visible > 0:
            self.tiempo_vuelta_visible -= 1
        if self.meta_cooldown > 0:
            self.meta_cooldown -= 1

    def _sensores(self, mascara):
        alto, ancho = mascara.shape
        for i, offset in enumerate(self.ANGULOS_SENSOR):
            rad  = math.radians(self.angulo + offset)
            dist = self.LARGO_SENSOR
            for d in range(1, self.LARGO_SENSOR + 1):
                px = int(self.x + math.cos(rad) * d)
                py = int(self.y + math.sin(rad) * d)
                if not (0 <= py < alto and 0 <= px < ancho):
                    dist = d
                    break
                if not mascara[py, px]:
                    dist = d
                    break
            self.lecturas[i] = dist

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
        self.x = float(x)
        self.y = float(y)
        self.angulo = float(angulo)
        self.vel = 0.0
        self.vivo = True
        self.distancia = 0.0
        self.lecturas = [self.LARGO_SENSOR] * len(self.ANGULOS_SENSOR)
        self.vueltas = 0
        self.paso_checkpoint = False
        self.tiempo_vuelta_visible = FPS * 3  # muestra "VUELTA 1" al reiniciar
        self.meta_cooldown = int(FPS * 1.5)


# ===================== UI JUEGO =====================
def dibujar_hud(pantalla, fuente, fuente_peq, carro, fps, contorno_on):
    bg = pygame.Surface((320, 175), pygame.SRCALPHA)
    bg.fill((10, 10, 10, 180))
    pantalla.blit(bg, (8, 8))

    lineas = [
        f"FPS        : {fps:.0f}",
        f"Velocidad  : {abs(carro.vel):.2f}",
        f"Angulo     : {carro.angulo % 360:.1f} grados",
        f"Distancia  : {carro.distancia:.0f} px",
        f"Sensores   : {[int(l) for l in carro.lecturas]}",
        f"Estado     : {'EN PISTA' if carro.vivo else 'FUERA'}",
        f"Vuelta     : {carro.vueltas} / {TOTAL_VUELTAS}",
        f"Checkpoint : {'✓ LISTO' if carro.paso_checkpoint else '— pendiente'}",
        f"Contorno   : {'ON  [C]' if contorno_on else 'OFF [C]'}",
    ]

    for i, lin in enumerate(lineas):
        if "FUERA" in lin:
            color = (255, 80, 80)
        elif "✓" in lin:
            color = (80, 255, 80)
        else:
            color = COLOR_HUD_TEXT
        pantalla.blit(fuente.render(lin, True, color), (14, 14 + i * 19))

    ayuda = "[W/S] Accel/Freno  [A/D] Girar  [C] Contorno  [T] HUD  [X] Coords  [R] Reset  [ESC] Menú"
    pantalla.blit(fuente_peq.render(ayuda, True, (150, 150, 150)),
                  (ANCHO_VENTANA // 2 - 290, ALTO_VENTANA - 20))


def dibujar_hud_coords(pantalla, fuente, fuente_peq, carro, mapa_actual):
    """Panel inferior derecho con coordenadas del carro, meta y checkpoint."""
    lineas = [
        "── COORDENADAS DEBUG ──",
        f"Carro X,Y  : ({carro.x:.0f}, {carro.y:.0f})",
        f"Meta  rect : x={mapa_actual['meta'].x}  y={mapa_actual['meta'].y}",
        f"            w={mapa_actual['meta'].width}  h={mapa_actual['meta'].height}",
        f"Chkpt rect : x={mapa_actual['checkpoint'].x}  y={mapa_actual['checkpoint'].y}",
        f"            w={mapa_actual['checkpoint'].width}  h={mapa_actual['checkpoint'].height}",
        f"Chkpt OK   : {'SÍ ✓' if carro.paso_checkpoint else 'NO'}",
        f"MetaCooldown: {carro.meta_cooldown}",
    ]

    alto_panel = len(lineas) * 17 + 10
    ancho_panel = 265
    bx = ANCHO_VENTANA - ancho_panel - 8
    by = ALTO_VENTANA  - alto_panel - 28

    bg = pygame.Surface((ancho_panel, alto_panel), pygame.SRCALPHA)
    bg.fill((10, 10, 10, 190))
    pantalla.blit(bg, (bx, by))
    pygame.draw.rect(pantalla, (80, 80, 80), (bx, by, ancho_panel, alto_panel), 1, border_radius=6)

    for i, lin in enumerate(lineas):
        if i == 0:
            color = (255, 180, 50)
        elif "Carro" in lin:
            color = (100, 220, 255)
        elif "Meta" in lin or "Chkpt" in lin or "w=" in lin or "h=" in lin:
            color = (180, 255, 120)
        elif "SÍ" in lin:
            color = (80, 255, 80)
        else:
            color = (180, 180, 180)
        pantalla.blit(fuente_peq.render(lin, True, color), (bx + 6, by + 5 + i * 17))

    # Dibuja los rectángulos de meta y checkpoint sobre la pista
    # Meta → línea amarilla
    pygame.draw.rect(pantalla, (255, 220, 0), mapa_actual["meta"], 2)
    # Checkpoint → línea cian
    pygame.draw.rect(pantalla, (0, 220, 255), mapa_actual["checkpoint"], 2)
    # Etiquetas
    lm = fuente_peq.render("META", True, (255, 220, 0))
    lc = fuente_peq.render("CHKPT", True, (0, 220, 255))
    pantalla.blit(lm, (mapa_actual["meta"].x + 3,       mapa_actual["meta"].y - 14))
    pantalla.blit(lc, (mapa_actual["checkpoint"].x + 3, mapa_actual["checkpoint"].y - 14))


def dibujar_game_over(pantalla, fg, f):
    ov = pygame.Surface((ANCHO_VENTANA, ALTO_VENTANA), pygame.SRCALPHA)
    ov.fill((160, 0, 0, 70))
    pantalla.blit(ov, (0, 0))
    t1 = fg.render("FUERA DE PISTA!", True, (255, 60, 60))
    t2 = f.render("Presiona  R  para reiniciar  /  ESC para menú", True, (255, 210, 210))
    pantalla.blit(t1, (ANCHO_VENTANA//2 - t1.get_width()//2, ALTO_VENTANA//2 - 30))
    pantalla.blit(t2, (ANCHO_VENTANA//2 - t2.get_width()//2, ALTO_VENTANA//2 + 22))


def dibujar_aviso_vuelta(pantalla, fuente_gde, fuente, vuelta_actual):
    # vuelta_actual = cuántas vueltas se han COMPLETADO
    # Al inicio (0 completadas) → "VUELTA 1"
    # Al completar 1 → "VUELTA 2"
    # Al completar todas → "¡CARRERA TERMINADA!"
    if vuelta_actual >= TOTAL_VUELTAS:
        texto_grande = "¡CARRERA TERMINADA!"
        texto_peq    = f"Completaste las {TOTAL_VUELTAS} vueltas · Presiona R"
        color_grande = (255, 220, 0)
    else:
        texto_grande = f"VUELTA  {vuelta_actual}"
        texto_peq    = f"de {TOTAL_VUELTAS}  —  ¡Vamos!"
        color_grande = (255, 255, 255)

    # Fondo semitransparente centrado
    ancho_panel = 380
    alto_panel  = 90
    panel = pygame.Surface((ancho_panel, alto_panel), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    px = ANCHO_VENTANA // 2 - ancho_panel // 2
    py = ALTO_VENTANA  // 2 - 120
    pantalla.blit(panel, (px, py))
    pygame.draw.rect(pantalla, (255, 140, 0), (px, py, ancho_panel, alto_panel), 2, border_radius=10)

    t1 = fuente_gde.render(texto_grande, True, color_grande)
    t2 = fuente.render(texto_peq, True, (200, 200, 200))
    pantalla.blit(t1, t1.get_rect(center=(ANCHO_VENTANA//2, py + 32)))
    pantalla.blit(t2, t2.get_rect(center=(ANCHO_VENTANA//2, py + 65)))


# ===================== UI MENÚ / SELECCIÓN =====================

def dibujar_degradado_fondo(pantalla):
    """Fondo con degradado gris oscuro de arriba a abajo."""
    for y in range(ALTO_VENTANA):
        t = y / ALTO_VENTANA
        c = int(15 + t * 35)   # de 15 (muy oscuro) a 50 (gris medio)
        pygame.draw.line(pantalla, (c, c, c), (0, y), (ANCHO_VENTANA, y))


def dibujar_lineas_decorativas(pantalla):
    """Unas líneas tipo pista de fondo decorativas."""
    color = (40, 40, 40)
    for i in range(0, ANCHO_VENTANA, 60):
        pygame.draw.line(pantalla, color, (i, 0), (i, ALTO_VENTANA), 1)
    for j in range(0, ALTO_VENTANA, 60):
        pygame.draw.line(pantalla, color, (0, j), (ANCHO_VENTANA, j), 1)


def dibujar_boton_menu(pantalla, fuente, rect, texto, activo=False, mouse_pos=(0,0)):
    hover = rect.collidepoint(mouse_pos)
    if activo:
        bg    = (200, 100, 0)
        borde = (255, 180, 50)
    elif hover:
        bg    = (60, 60, 60)
        borde = (255, 140, 0)
    else:
        bg    = (30, 30, 30)
        borde = (120, 120, 120)

    pygame.draw.rect(pantalla, bg, rect, border_radius=12)
    pygame.draw.rect(pantalla, borde, rect, 2, border_radius=12)
    txt = fuente.render(texto, True, (240, 240, 240))
    pantalla.blit(txt, txt.get_rect(center=rect.center))


def pantalla_menu(pantalla, fuente, fuente_gde, fuente_titulo, mouse_pos):
    """
    Retorna:
      "inicio"  -> si se hace click en INICIAR CARRERA
      "salir"   -> si se hace click en SALIR
      None      -> sin acción
    """
    dibujar_degradado_fondo(pantalla)
    dibujar_lineas_decorativas(pantalla)

    # Rectángulo central semi-transparente
    panel = pygame.Surface((500, 420), pygame.SRCALPHA)
    panel.fill((10, 10, 10, 200))
    cx = ANCHO_VENTANA // 2 - 250
    cy = ALTO_VENTANA  // 2 - 200
    pantalla.blit(panel, (cx, cy))
    pygame.draw.rect(pantalla, (255, 140, 0), (cx, cy, 500, 420), 2, border_radius=18)

    # Título
    t_titulo = fuente_titulo.render("RACE SIMULATOR", True, (255, 200, 0))
    t_sub    = fuente.render("Simulador de Carreras con IA", True, (180, 180, 180))
    t_ver    = fuente.render("v1.0  -  Motor Base + Obstáculos", True, (100, 100, 100))

    pantalla.blit(t_titulo, t_titulo.get_rect(center=(ANCHO_VENTANA//2, cy + 70)))
    pantalla.blit(t_sub,    t_sub.get_rect(center=(ANCHO_VENTANA//2, cy + 120)))
    pantalla.blit(t_ver,    t_ver.get_rect(center=(ANCHO_VENTANA//2, cy + 148)))

    # Línea divisoria
    pygame.draw.line(pantalla, (255, 140, 0),
                     (cx + 30, cy + 170), (cx + 470, cy + 170), 1)

    # Info
    info_lineas = [
        "Elige tu mapa y carro en la pantalla siguiente.",
        "Usa W/A/S/D o flechas para conducir.",
        f"Completa {TOTAL_VUELTAS} vueltas sin salirte de la pista.",
    ]
    for i, lin in enumerate(info_lineas):
        surf = fuente.render(lin, True, (160, 160, 160))
        pantalla.blit(surf, surf.get_rect(center=(ANCHO_VENTANA//2, cy + 195 + i * 22)))

    # Botones
    btn_start = pygame.Rect(ANCHO_VENTANA//2 - 140, cy + 295, 280, 50)
    btn_salir = pygame.Rect(ANCHO_VENTANA//2 - 100, cy + 358, 200, 38)

    dibujar_boton_menu(pantalla, fuente_gde, btn_start, "►  INICIAR CARRERA", mouse_pos=mouse_pos)
    dibujar_boton_menu(pantalla, fuente,     btn_salir, "SALIR", mouse_pos=mouse_pos)

    return btn_start, btn_salir


def pantalla_seleccion(pantalla, fuente, fuente_gde, fuente_titulo,
                       mouse_pos, mapa_sel, carro_sel,
                       previews_mapa, previews_carro):
    """
    Pantalla de selección de mapa y carro.
    Retorna botones para click detection.
    """
    dibujar_degradado_fondo(pantalla)
    dibujar_lineas_decorativas(pantalla)

    # Título
    t = fuente_titulo.render("SELECCIONA TU CONFIGURACIÓN", True, (255, 200, 0))
    pantalla.blit(t, t.get_rect(center=(ANCHO_VENTANA//2, 42)))

    # -------- SECCIÓN MAPA --------
    panel_mapa = pygame.Surface((340, 340), pygame.SRCALPHA)
    panel_mapa.fill((10, 10, 10, 200))
    pantalla.blit(panel_mapa, (20, 80))
    pygame.draw.rect(pantalla, (80, 80, 80), (20, 80, 340, 340), 1, border_radius=12)

    lbl = fuente_gde.render("MAPA", True, (200, 200, 200))
    pantalla.blit(lbl, (20 + 340//2 - lbl.get_width()//2, 88))

    # Preview del mapa seleccionado
    if previews_mapa and mapa_sel < len(previews_mapa):
        prev = previews_mapa[mapa_sel]
        pantalla.blit(prev, (30, 115))
        # Nombre
        nm = fuente.render(MAPAS[mapa_sel]["nombre"], True, (220, 220, 220))
        pantalla.blit(nm, nm.get_rect(center=(20 + 170, 115 + 200 + 12)))

    # Botones << >>
    btn_mapa_prev = pygame.Rect(30,  348, 60, 32)
    btn_mapa_next = pygame.Rect(270, 348, 60, 32)
    dibujar_boton_menu(pantalla, fuente, btn_mapa_prev, "◄◄", mouse_pos=mouse_pos)
    dibujar_boton_menu(pantalla, fuente, btn_mapa_next, "►►", mouse_pos=mouse_pos)

    num_mapa = fuente.render(f"{mapa_sel+1} / {len(MAPAS)}", True, (160,160,160))
    pantalla.blit(num_mapa, num_mapa.get_rect(center=(20 + 170, 363)))

    # -------- SECCIÓN CARRO --------
    panel_carro = pygame.Surface((320, 340), pygame.SRCALPHA)
    panel_carro.fill((10, 10, 10, 200))
    pantalla.blit(panel_carro, (380, 80))
    pygame.draw.rect(pantalla, (80, 80, 80), (380, 80, 320, 340), 1, border_radius=12)

    lbl2 = fuente_gde.render("CARRO", True, (200, 200, 200))
    pantalla.blit(lbl2, (380 + 160 - lbl2.get_width()//2, 88))

    # Preview del carro seleccionado
    if previews_carro and carro_sel < len(previews_carro):
        prev_c = previews_carro[carro_sel]
        # centrar en la zona
        px = 380 + 160 - prev_c.get_width()//2
        py = 130
        pantalla.blit(prev_c, (px, py))
        # Nombre
        nc = fuente.render(CARROS_IMAGENES[carro_sel]["nombre"], True, (220, 220, 220))
        pantalla.blit(nc, nc.get_rect(center=(380 + 160, 325)))

    btn_carro_prev = pygame.Rect(390, 348, 60, 32)
    btn_carro_next = pygame.Rect(630, 348, 60, 32)
    dibujar_boton_menu(pantalla, fuente, btn_carro_prev, "◄◄", mouse_pos=mouse_pos)
    dibujar_boton_menu(pantalla, fuente, btn_carro_next, "►►", mouse_pos=mouse_pos)

    num_carro = fuente.render(f"{carro_sel+1} / {len(CARROS_IMAGENES)}", True, (160,160,160))
    pantalla.blit(num_carro, num_carro.get_rect(center=(380 + 160, 363)))

    # -------- BOTONES INFERIORES --------
    btn_start  = pygame.Rect(ANCHO_VENTANA//2 - 160, 455, 320, 55)
    btn_volver = pygame.Rect(ANCHO_VENTANA//2 - 100, 520, 200, 38)

    dibujar_boton_menu(pantalla, fuente_gde, btn_start,  "▶  COMENZAR",  mouse_pos=mouse_pos)
    dibujar_boton_menu(pantalla, fuente,     btn_volver, "◄ Volver",      mouse_pos=mouse_pos)

    # Descripción selección actual
    desc = fuente.render(
        f"Mapa: {MAPAS[mapa_sel]['nombre']}   |   Carro: {CARROS_IMAGENES[carro_sel]['nombre']}",
        True, (255, 180, 50)
    )
    pantalla.blit(desc, desc.get_rect(center=(ANCHO_VENTANA//2, 440)))

    return btn_mapa_prev, btn_mapa_next, btn_carro_prev, btn_carro_next, btn_start, btn_volver


# ===================== MAIN =====================
def main():
    global OBSTACULOS

    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
    pygame.display.set_caption(TITULO)
    reloj = pygame.time.Clock()

    fuente       = pygame.font.SysFont("Consolas", 13)
    fuente_peq   = pygame.font.SysFont("Consolas", 11)
    fuente_gde   = pygame.font.SysFont("Consolas", 22, bold=True)
    fuente_titulo = pygame.font.SysFont("Consolas", 26, bold=True)

    # ---- Generar previews de mapas (thumbnail) ----
    previews_mapa = []
    for mp in MAPAS:
        try:
            img_m = pygame.image.load(mp["img"]).convert()
            img_m = pygame.transform.scale(img_m, (300, 195))
            previews_mapa.append(img_m)
        except FileNotFoundError:
            # Placeholder gris
            ph = pygame.Surface((300, 195))
            ph.fill((60, 60, 60))
            f_ph = pygame.font.SysFont("Consolas", 14)
            t_ph = f_ph.render("(imagen no encontrada)", True, (150,150,150))
            ph.blit(t_ph, t_ph.get_rect(center=(150, 97)))
            previews_mapa.append(ph)

    # ---- Generar previews de carros (grande) ----
    previews_carro = []
    for i, c_info in enumerate(CARROS_IMAGENES):
        tinte = TINTES_CARROS[i] if i < len(TINTES_CARROS) else None
        img_c = cargar_imagen_carro(c_info["img"], tinte)
        img_c_big = pygame.transform.smoothscale(img_c, (160, 160))
        previews_carro.append(img_c_big)

    # ---- Estado inicial ----
    estado       = ESTADO_MENU
    mapa_sel     = 0
    carro_sel    = 0

    # Variables de juego (se asignan al entrar al estado JUEGO)
    fondo        = None
    mascara      = None
    contorno_surf = None
    sx = sy = sang = 0
    carro        = None
    mostrar_contorno = True
    mostrar_hud = True
    mostrar_coords = False
    mapa_actual  = None

    ejecutando = True
    while ejecutando:
        reloj.tick(FPS)
        fps_real = reloj.get_fps()
        mouse_pos = pygame.mouse.get_pos()

        # ==================== MENU ====================
        if estado == ESTADO_MENU:
            btn_start, btn_salir = pantalla_menu(
                pantalla, fuente, fuente_gde, fuente_titulo, mouse_pos
            )

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    ejecutando = False
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    ejecutando = False
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if btn_start.collidepoint(ev.pos):
                        estado = ESTADO_SELECCION
                    if btn_salir.collidepoint(ev.pos):
                        ejecutando = False

        # ==================== SELECCION ====================
        elif estado == ESTADO_SELECCION:
            (btn_mapa_prev, btn_mapa_next,
             btn_carro_prev, btn_carro_next,
             btn_comenzar, btn_volver) = pantalla_seleccion(
                pantalla, fuente, fuente_gde, fuente_titulo,
                mouse_pos, mapa_sel, carro_sel,
                previews_mapa, previews_carro
            )

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    ejecutando = False
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        estado = ESTADO_MENU
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if btn_mapa_prev.collidepoint(ev.pos):
                        mapa_sel = (mapa_sel - 1) % len(MAPAS)
                    if btn_mapa_next.collidepoint(ev.pos):
                        mapa_sel = (mapa_sel + 1) % len(MAPAS)
                    if btn_carro_prev.collidepoint(ev.pos):
                        carro_sel = (carro_sel - 1) % len(CARROS_IMAGENES)
                    if btn_carro_next.collidepoint(ev.pos):
                        carro_sel = (carro_sel + 1) % len(CARROS_IMAGENES)
                    if btn_volver.collidepoint(ev.pos):
                        estado = ESTADO_MENU
                    if btn_comenzar.collidepoint(ev.pos):
                        # Cargar mapa y carro seleccionados
                        mapa_actual = MAPAS[mapa_sel]
                        try:
                            fondo, mascara, contorno_surf, sx, sy, sang = cargar_mapa(
                                mapa_actual, ANCHO_VENTANA, ALTO_VENTANA
                            )
                        except FileNotFoundError:
                            # Si falla, quedarse en selección
                            continue

                        tinte    = TINTES_CARROS[carro_sel] if carro_sel < len(TINTES_CARROS) else None
                        img_carro = cargar_imagen_carro(CARROS_IMAGENES[carro_sel]["img"], tinte)
                        carro    = Carro(sx, sy, sang, img_carro)
                        OBSTACULOS.clear()
                        mostrar_contorno = True
                        mostrar_hud = True
                        mostrar_coords = False
                        estado = ESTADO_JUEGO

        # ==================== JUEGO ====================
        elif estado == ESTADO_JUEGO:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    ejecutando = False
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        estado = ESTADO_MENU
                    if ev.key == pygame.K_r:
                        carro.reiniciar(sx, sy, sang)
                    if ev.key == pygame.K_c:
                        mostrar_contorno = not mostrar_contorno
                    if ev.key == pygame.K_t:
                        mostrar_hud = not mostrar_hud
                    if ev.key == pygame.K_x:
                        mostrar_coords = not mostrar_coords

            # Update
            carro.actualizar(pygame.key.get_pressed(), mascara)

            if carro.vivo and carro_choca_obstaculo(carro, OBSTACULOS):
                carro.vel *= 0.2
                carro.vivo = False

            # Lógica de vueltas
            if carro.vivo and carro.vueltas < TOTAL_VUELTAS:
                carro_rect_colision = carro.rect.inflate(20, 20)
                # Checkpoint: a mitad del recorrido, valida que dio media vuelta
                if not carro.paso_checkpoint and carro_rect_colision.colliderect(mapa_actual["checkpoint"]):
                    carro.paso_checkpoint = True
                # Meta: solo cuenta si ya pasó checkpoint Y ya pasó el cooldown inicial
                if (carro.paso_checkpoint
                        and carro.meta_cooldown == 0
                        and carro_rect_colision.colliderect(mapa_actual["meta"])):
                    carro.vueltas += 1
                    carro.paso_checkpoint = False
                    carro.tiempo_vuelta_visible = FPS * 3

            # Dibujo
            pantalla.blit(fondo, (0, 0))
            if mostrar_contorno:
                pantalla.blit(contorno_surf, (0, 0))

            dibujar_obstaculos(pantalla, OBSTACULOS)
            carro.dibujar(pantalla)
            if mostrar_hud:
                dibujar_hud(pantalla, fuente, fuente_peq, carro, fps_real, mostrar_contorno)
            if mostrar_coords:
                dibujar_hud_coords(pantalla, fuente, fuente_peq, carro, mapa_actual)

            if not carro.vivo:
                dibujar_game_over(pantalla, fuente_gde, fuente)

            if carro.tiempo_vuelta_visible > 0:
                dibujar_aviso_vuelta(pantalla, fuente_gde, fuente, carro.vueltas)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()