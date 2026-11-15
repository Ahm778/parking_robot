"""Décor du mode démo (vidéos) : marquages au sol, lampadaires, arbres, cônes, vue caméra de départ.

Appelé par scripts/generate_world.py seulement si PARKING_DEMO=1. Rien de tout cela n'existe en mode normal.
Les cônes et lampadaires ont une collision : le lidar les voit, le robot les contourne (obstacles fixes).
Les marquages au sol n'ont pas de collision (simple peinture).
"""
import math
import os
from pathlib import Path

from parking_robot.geometry import AISLES_Y, AISLE_HALF, PED, ROWS, SLOT_L, SLOT_W, WALL_X, WALL_Y, XS

ASPHALT = '0.17 0.17 0.18 1'
WHITE, YELLOW = '0.95 0.95 0.95 1', '0.95 0.78 0.1 1'


def _model(name, x, y, links, yaw=0.0, static=True):
    return (f"<model name='{name}'><static>{str(static).lower()}</static><pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose>"
            + ''.join(links) + '</model>')


def _link(lname, geom, rgba, z, collide=True, emissive=None, pose_xy=(0.0, 0.0)):
    mat = f'<ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>' + (f'<emissive>{emissive}</emissive>' if emissive else '')
    col = f"<collision name='c'><geometry>{geom}</geometry></collision>" if collide else ''
    return (f"<link name='{lname}'><pose>{pose_xy[0]} {pose_xy[1]} {z} 0 0 0</pose>{col}"
            f"<visual name='v'><geometry>{geom}</geometry><material>{mat}</material></visual></link>")


def _paint(name, x, y, sx, sy, rgba, yaw=0.0):
    return _model(name, x, y, [_link('l', f'<box><size>{sx} {sy} 0.006</size></box>', rgba, 0.004, collide=False)], yaw)


def _sign_model():
    """Panneau « P » bleu (texture PNG sur un maillage OBJ, comme les plaques), écrit dans ~/gazebo_models."""
    from PIL import Image, ImageDraw, ImageFont
    root = Path(os.path.expanduser('~/gazebo_models')) / 'demo_panneau_p'
    (root / 'meshes').mkdir(parents=True, exist_ok=True)
    (root / 'materials' / 'textures').mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (512, 512), (20, 70, 170))
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, 499, 499], outline=(255, 255, 255), width=18)
    font = None
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'):
        if os.path.exists(f):
            font = ImageFont.truetype(f, 380)
            break
    font = font or ImageFont.load_default()
    b = d.textbbox((0, 0), 'P', font=font)
    d.text(((512 - (b[2] - b[0])) / 2 - b[0], (512 - (b[3] - b[1])) / 2 - b[1]), 'P', font=font, fill=(255, 255, 255))
    for where in (root / 'meshes', root / 'materials' / 'textures'):
        img.save(where / 'panneau.png')
    (root / 'meshes' / 'panneau.mtl').write_text('newmtl p\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd panneau.png\n')
    h = 0.45                                           # demi-côté du panneau (90 cm)
    (root / 'meshes' / 'panneau.obj').write_text(       # deux faces (+x et -x), lisibles des deux côtés
        'mtllib panneau.mtl\n'
        f'v 0.01 {-h} {-h}\nv 0.01 {h} {-h}\nv 0.01 {h} {h}\nv 0.01 {-h} {h}\n'
        f'v -0.01 {h} {-h}\nv -0.01 {-h} {-h}\nv -0.01 {-h} {h}\nv -0.01 {h} {h}\n'
        'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 1 0 0\nvn -1 0 0\nusemtl p\n'
        'f 1/1/1 2/2/1 3/3/1 4/4/1\nf 5/1/2 6/2/2 7/3/2 8/4/2\n')
    (root / 'model.config').write_text('<?xml version="1.0"?><model><name>Panneau P</name><version>1.0</version>'
                                       '<sdf version="1.6">model.sdf</sdf></model>')
    (root / 'model.sdf').write_text(
        "<sdf version='1.6'><model name='demo_panneau_p'><static>true</static>"
        "<link name='mat'><pose>0 0 1.1 0 0 0</pose><collision name='c'><geometry><cylinder><radius>0.05</radius>"
        "<length>2.2</length></cylinder></geometry></collision><visual name='v'><geometry><cylinder><radius>0.05</radius>"
        "<length>2.2</length></cylinder></geometry><material><ambient>0.6 0.6 0.62 1</ambient><diffuse>0.6 0.6 0.62 1"
        "</diffuse></material></visual></link>"
        "<link name='panneau'><pose>0 0 2.6 0 0 0</pose><visual name='v'><geometry><mesh>"
        "<uri>model://demo_panneau_p/meshes/panneau.obj</uri></mesh></geometry></visual></link></model></sdf>")


def decor():
    parts = []
    # --- vue de départ de la fenêtre Gazebo (vue plongeante de trois quarts) ---
    parts.append("<gui fullscreen='0'><camera name='user_camera'><pose>-24 -36 27 0 0.55 0.98</pose>"
                 "<view_controller>orbit</view_controller></camera></gui>")
    # --- ciel, lumière d'ambiance, sans la grille de Gazebo ---
    parts.append('<scene><ambient>0.55 0.55 0.58 1</ambient><background>0.62 0.76 0.92 1</background>'
                 '<sky><clouds><speed>3</speed></clouds></sky><grid>false</grid><origin_visual>false</origin_visual>'
                 '<shadows>false</shadows></scene>')
    # --- pelouse autour du parking et trottoir clair le long des murs ---
    g, o = 12.0, 0.3                                                   # largeur de pelouse, débord sous le mur
    for i, (x, y, sx, sy) in enumerate([(0, WALL_Y + g / 2 - o, 2 * (WALL_X + g), g), (0, -WALL_Y - g / 2 + o, 2 * (WALL_X + g), g),
                                         (WALL_X + g / 2 - o, 0, g, 2 * WALL_Y), (-WALL_X - g / 2 + o, 0, g, 2 * WALL_Y)]):
        parts.append(_model(f'pelouse_{i}', x, y, [_link('l', f'<box><size>{sx} {sy} 0.004</size></box>', '0.27 0.48 0.22 1',
                                                         0.002, collide=False)]))
    t = 1.0                                                            # largeur du trottoir peint
    for i, (x, y, sx, sy) in enumerate([(0, WALL_Y - t / 2, 2 * WALL_X, t), (0, -WALL_Y + t / 2, 2 * WALL_X, t),
                                         (WALL_X - t / 2, 0, t, 2 * WALL_Y), (-WALL_X + t / 2, 0, t, 2 * WALL_Y)]):
        parts.append(_paint(f'trottoir_{i}', x, y, sx, sy, '0.55 0.55 0.55 1'))
    # --- panneau « P » à l'entrée (voie de gauche) ---
    _sign_model()
    parts.append(f"<include><name>panneau_p</name><uri>model://demo_panneau_p</uri><static>true</static>"
                 f"<pose>{-WALL_X + 0.45} -4.0 0 0 0 0</pose></include>")
    # --- asphalte (sous les marquages) ---
    parts.append(_model('asphalte', 0, 0, [_link('l', f'<box><size>{2 * WALL_X} {2 * WALL_Y} 0.002</size></box>',
                                                   ASPHALT, 0.001, collide=False)]))
    # --- allées : ligne centrale jaune en tirets et flèches de sens de circulation ---
    x0, x1 = XS[0] - SLOT_W / 2, XS[-1] + SLOT_W / 2
    for k, ay in enumerate(AISLES_Y):
        n = int((x1 - x0) // 2.0)
        for i in range(n):
            parts.append(_paint(f'tiret_{k}_{i}', x0 + 1.0 + 2.0 * i, ay, 1.0, 0.12, YELLOW))
        d = 0.0 if k % 2 == 0 else math.pi                               # sens du parcours en S
        for j, ax in enumerate((XS[0] + SLOT_W / 2, XS[3] + SLOT_W / 2)):   # loin du passage piéton
            for side in (-1, 1):
                y = ay + side * 2.2
                tx, ty = ax + 0.6 * math.cos(d), y                         # pointe de la flèche
                parts.append(_paint(f'fleche_{k}_{j}_{side}', ax, y, 1.2, 0.18, WHITE, d))
                for h, a in (('a', d + math.pi - 0.6), ('b', d + math.pi + 0.6)):
                    parts.append(_paint(f'pointe_{k}_{j}_{side}_{h}', tx + 0.25 * math.cos(a), ty + 0.25 * math.sin(a),
                                        0.55, 0.15, WHITE, a))
    # --- passage piéton là où traverse le piéton (allée 1) ---
    ay = AISLES_Y[0]
    for i in range(7):
        parts.append(_paint(f'pieton_{i}', PED[0], ay - AISLE_HALF + 0.5 + i * (2 * AISLE_HALF - 1.0) / 6, 1.6, 0.45, WHITE))
    # --- numéros de place : petite plaque au sol devant chaque place (couleur par rangée) ---
    colors = {'A': '0.2 0.45 0.9 1', 'B': '0.2 0.7 0.35 1', 'C': '0.85 0.45 0.15 1', 'D': '0.6 0.3 0.8 1'}
    for r, info in ROWS.items():
        for i, x in enumerate(XS):
            parts.append(_paint(f'repere_{r}{i + 1}', x, info['stop_y'] + info['side'] * 0.45, 0.7, 0.4,
                                colors.get(r, WHITE)))
    # --- lampadaires (collision : le lidar les voit), le long des murs ---
    lamp = lambda n, x, y: _model(n, x, y, [
        _link('mat', '<cylinder><radius>0.07</radius><length>4.0</length></cylinder>', '0.3 0.3 0.32 1', 2.0),
        _link('tete', '<box><size>0.6 0.25 0.12</size></box>', '0.95 0.95 0.8 1', 4.0, collide=False,
              emissive='0.9 0.9 0.7 1')])
    for i, x in enumerate((-WALL_X + 0.6, 0.0, WALL_X - 0.6)):
        parts.append(lamp(f'lampadaire_h{i}', x, WALL_Y - 0.6))
        parts.append(lamp(f'lampadaire_b{i}', x, -WALL_Y + 0.6))
    # --- arbres dans les coins (tronc avec collision, feuillage décoratif) ---
    tree = lambda n, x, y: _model(n, x, y, [
        _link('tronc', '<cylinder><radius>0.15</radius><length>2.0</length></cylinder>', '0.4 0.27 0.15 1', 1.0),
        _link('feuillage', '<sphere><radius>1.1</radius></sphere>', '0.18 0.5 0.2 1', 2.8, collide=False)])
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(tree(f'arbre_{sx}_{sy}', sx * (WALL_X - 1.4), sy * (WALL_Y - 1.4)))
    # --- OBSTACLES (avec collision : le lidar les voit, le robot les contourne) ---
    def cone(n, x, y):                                                 # cône de chantier de 75 cm
        return _model(n, x, y, [
            _link('base', '<box><size>0.5 0.5 0.05</size></box>', '0.08 0.08 0.08 1', 0.025),
            _link('corps', '<cylinder><radius>0.17</radius><length>0.7</length></cylinder>', '1 0.35 0 1', 0.4,
                  emissive='0.35 0.1 0 1'),
            _link('bande1', '<cylinder><radius>0.175</radius><length>0.08</length></cylinder>', WHITE, 0.5, collide=False),
            _link('bande2', '<cylinder><radius>0.175</radius><length>0.06</length></cylinder>', WHITE, 0.3, collide=False)])

    def barriere(n, x, y, yaw=0.0):                                    # barrière de chantier rouge et blanche, 2,4 m
        links = [_link('pied_g', '<box><size>0.08 0.5 0.06</size></box>', '0.2 0.2 0.2 1', 0.03, pose_xy=(-1.1, 0)),
                 _link('pied_d', '<box><size>0.08 0.5 0.06</size></box>', '0.2 0.2 0.2 1', 0.03, pose_xy=(1.1, 0)),
                 _link('montant_g', '<box><size>0.06 0.06 0.9</size></box>', '0.85 0.85 0.85 1', 0.45, pose_xy=(-1.1, 0)),
                 _link('montant_d', '<box><size>0.06 0.06 0.9</size></box>', '0.85 0.85 0.85 1', 0.45, pose_xy=(1.1, 0))]
        for i in range(8):                                             # panneau à bandes alternées
            links.append(_link(f'p{i}', '<box><size>0.3 0.05 0.35</size></box>', '0.9 0.05 0.05 1' if i % 2 else WHITE,
                               0.62, pose_xy=(-1.05 + 0.3 * i, 0)))
        links.append(_link('bas', '<box><size>2.2 0.05 0.25</size></box>', '0.9 0.05 0.05 1', 0.2))   # vu par le lidar
        return _model(n, x, y, links, yaw)

    lane_in, lane_out = x1 + 0.8, WALL_X - 0.9                         # bords intérieur / extérieur des voies latérales
    for i, (x, y) in enumerate([(-lane_out, 5.0), (-lane_in, -5.0), (lane_out, -4.0), (lane_in, 4.5),
                                (-lane_out, -14.0), (lane_in, -15.0)]):
        parts.append(cone(f'cone_{i}', x, y))
    # zone de travaux au milieu de l'allée 2 : barrière + cônes, que le robot doit contourner
    ay2 = AISLES_Y[-1]
    parts.append(barriere('travaux_barriere', 1.75, ay2))
    for i, dx in enumerate((-1.6, 1.6)):
        parts.append(cone(f'travaux_cone_{i}', 1.75 + dx, ay2))
    # --- personnes immobiles sur le trottoir (obstacles) ---
    for i, (x, y, yaw) in enumerate([(-WALL_X + 0.55, -9.0, 0.0), (WALL_X - 0.55, 11.0, math.pi)]):
        parts.append(f"<include><name>personne_{i}</name><uri>model://person_standing</uri><static>true</static>"
                     f"<pose>{x} {y} 0 0 0 {yaw}</pose></include>")

    # --- DÉCOR EXTÉRIEUR (sans effet sur le robot) ---
    parts.append("<light type='directional' name='lumiere_douce'><cast_shadows>false</cast_shadows>"
                 "<pose>0 0 10 0 0 0</pose><diffuse>0.28 0.24 0.2 1</diffuse><specular>0 0 0 1</specular>"
                 "<direction>0.6 -0.4 -0.7</direction></light>")
    hedge = '0.16 0.4 0.15 1'                                          # haies le long des murs, côté extérieur
    for i, (x, y, sx, sy) in enumerate([(0, WALL_Y + 0.9, 2 * WALL_X + 2.4, 1.0), (0, -WALL_Y - 0.9, 2 * WALL_X + 2.4, 1.0),
                                         (WALL_X + 0.9, 0, 1.0, 2 * WALL_Y), (-WALL_X - 0.9, 0, 1.0, 2 * WALL_Y)]):
        parts.append(_model(f'haie_{i}', x, y, [_link('l', f'<box><size>{sx} {sy} 1.1</size></box>', hedge, 0.55,
                                                      collide=False)]))
    for i in range(8):                                                 # rangées d'arbres le long des longs côtés
        for sx in (-1, 1):
            parts.append(tree(f'arbre_ext_{i}_{sx}', sx * (WALL_X + 4.0), -WALL_Y + 3 + i * (2 * WALL_Y - 6) / 7))

    def building(n, x, y, w, d, h, rgba, face):                        # immeuble avec fenêtres sur la façade visible
        links = [_link('corps', f'<box><size>{w} {d} {h}</size></box>', rgba, h / 2, collide=False)]
        for fl in range(int(h // 3.2)):
            for c in range(int(w // 2.6)):
                fx = -w / 2 + 1.3 + c * 2.6
                links.append(_link(f'f{fl}_{c}', '<box><size>1.4 0.05 1.5</size></box>', '0.35 0.55 0.75 1',
                                   1.9 + fl * 3.2, collide=False, pose_xy=(fx, face * (d / 2 + 0.03))))
        return _model(n, x, y, links)

    parts.append(building('immeuble_nord', -6, WALL_Y + 17, 24, 10, 16, '0.82 0.78 0.7 1', -1))
    parts.append(building('immeuble_nord2', 16, WALL_Y + 19, 14, 10, 10, '0.7 0.72 0.78 1', -1))
    parts.append(building('immeuble_sud', 4, -WALL_Y - 17, 28, 10, 13, '0.85 0.75 0.65 1', 1))
    return parts
