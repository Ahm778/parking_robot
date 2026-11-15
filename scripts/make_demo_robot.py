#!/usr/bin/env python3
"""Crée un TurtleBot3 waffle_pi « habillé » pour les vidéos de démonstration (Gazebo Classic, ROS 2 Humble).

Ajoute au modèle officiel UNIQUEMENT des éléments visuels (aucune collision, aucune masse) :
  - un mât à l'arrière avec un gyrophare orange qui clignote ;
  - un fanion « ROBOT PARKING » (lisible des deux côtés) ;
  - des bandes orange sur les flancs et à l'arrière.
Le lidar de Gazebo Classic ne voit que les collisions, et rien n'est placé devant la caméra :
la navigation et la vision se comportent exactement comme avec le robot d'origine.

Usage : python3 scripts/make_demo_robot.py      (écrit ~/gazebo_models/turtlebot3_waffle_pi_demo)
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAME = 'turtlebot3_waffle_pi_demo'
ORANGE, ORANGE_DARK = '1 0.45 0 1', '0.35 0.12 0 1'


def find_tb3_sdf():
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('turtlebot3_gazebo'))
    except Exception:  # noqa: BLE001
        sys.exit('turtlebot3_gazebo introuvable : faites d\'abord « source /opt/ros/humble/setup.bash » '
                 'et sourcez le workspace qui contient turtlebot3_gazebo')
    sdf = share / 'models' / 'turtlebot3_waffle_pi' / 'model.sdf'
    if not sdf.exists():
        sys.exit(f'Modèle introuvable : {sdf}')
    return sdf


def font(size):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def write_flag(dst):
    """Fanion texturé : maillage OBJ à deux faces (texte lisible des deux côtés) + MTL + PNG."""
    W, H = 0.22, 0.07                                   # m
    (dst / 'meshes').mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (880, 280), (20, 60, 140))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 871, 271], outline=(255, 255, 255), width=10)
    for size in range(150, 20, -4):                     # plus grande taille qui tient dans le cadre
        f = font(size)
        box = d.textbbox((0, 0), 'ROBOT PARKING', font=f)
        if box[2] - box[0] < 800 and box[3] - box[1] < 200:
            break
    tw, th = box[2] - box[0], box[3] - box[1]
    d.text(((880 - tw) / 2 - box[0], (280 - th) / 2 - box[1]), 'ROBOT PARKING', font=f, fill=(255, 255, 255))
    img.save(dst / 'meshes' / 'flag.png')
    (dst / 'materials' / 'textures').mkdir(parents=True, exist_ok=True)
    img.save(dst / 'materials' / 'textures' / 'flag.png')          # emplacement lu par Gazebo Classic (comme les plaques)
    (dst / 'meshes' / 'flag.mtl').write_text(
        'newmtl flag\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd flag.png\n')
    # quadrilatère dans le plan x-z, face +y puis face -y (u inversé pour que le texte ne soit pas en miroir)
    x0, x1, z0, z1 = -W / 2, W / 2, -H / 2, H / 2
    (dst / 'meshes' / 'flag.obj').write_text(
        'mtllib flag.mtl\no flag\n'
        f'v {x0} 0.001 {z0}\nv {x1} 0.001 {z0}\nv {x1} 0.001 {z1}\nv {x0} 0.001 {z1}\n'
        f'v {x0} -0.001 {z0}\nv {x1} -0.001 {z0}\nv {x1} -0.001 {z1}\nv {x0} -0.001 {z1}\n'
        'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n'
        'vn 0 1 0\nvn 0 -1 0\nusemtl flag\n'
        'f 2/1/1 1/2/1 4/3/1 3/4/1\n'          # face +y : lue de ce côté de gauche à droite
        'f 5/1/2 6/2/2 7/3/2 8/4/2\n')         # face -y


def visual(name, pose, geometry, color=None, emissive=None, plugin=None):
    v = ET.Element('visual', name=name)
    ET.SubElement(v, 'pose').text = pose
    g = ET.SubElement(v, 'geometry')
    g.append(ET.fromstring(geometry))
    if color:
        m = ET.SubElement(v, 'material')
        ET.SubElement(m, 'ambient').text = color
        ET.SubElement(m, 'diffuse').text = color
        if emissive:
            ET.SubElement(m, 'emissive').text = emissive
    if plugin is not None:
        v.append(plugin)
    return v


def main():
    src = find_tb3_sdf()
    dst = Path('~/gazebo_models').expanduser() / NAME
    dst.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(src)
    root = tree.getroot()
    model = root.find('model')
    model.set('name', 'turtlebot3_waffle_pi')
    links = model.findall('link')
    base = next((ln for ln in links if ln.get('name') == 'base_link'), None) \
        or next((ln for ln in links if ln.find('visual') is not None), None)
    if base is None:
        sys.exit('Lien de base introuvable dans le modèle du TurtleBot3')

    for sensor in model.iter('sensor'):              # rayons du lidar (disque bleu) masqués en vidéo
        if sensor.get('type') in ('ray', 'gpu_ray'):
            vis = sensor.find('visualize')
            if vis is None:
                vis = ET.SubElement(sensor, 'visualize')
            vis.text = 'false'
    write_flag(dst)
    blink = ET.fromstring(
        '<plugin name="gyrophare" filename="libBlinkVisualPlugin.so">'
        '<color_a>1 0.5 0 1</color_a><color_b>0.25 0.08 0 1</color_b>'
        '<period>0.8</period><use_wall_time>true</use_wall_time></plugin>')
    mast_x = -0.115                                     # arrière du robot, derrière le lidar
    for v in [
        # mât + gyrophare
        visual('demo_mat', f'{mast_x} 0 0.29 0 0 0', '<cylinder><radius>0.008</radius><length>0.26</length></cylinder>',
               '0.75 0.75 0.75 1'),
        visual('demo_gyro_socle', f'{mast_x} 0 0.425 0 0 0', '<cylinder><radius>0.03</radius><length>0.012</length></cylinder>',
               '0.15 0.15 0.15 1'),
        visual('demo_gyrophare', f'{mast_x} 0 0.455 0 0 0', '<sphere><radius>0.034</radius></sphere>',
               ORANGE, '1 0.4 0 1', blink),
        # fanion, accroché au mât, vers l'arrière
        visual('demo_fanion', f'{mast_x - 0.115} 0 0.37 0 0 0',
               f'<mesh><uri>model://{NAME}/meshes/flag.obj</uri></mesh>'),
        # jupes orange sur les flancs, avec un liseré blanc
        visual('demo_jupe_g', '-0.02 0.154 0.075 0 0 0', '<box><size>0.27 0.004 0.05</size></box>', ORANGE),
        visual('demo_jupe_d', '-0.02 -0.154 0.075 0 0 0', '<box><size>0.27 0.004 0.05</size></box>', ORANGE),
        visual('demo_lisere_g', '-0.02 0.1565 0.09 0 0 0', '<box><size>0.27 0.002 0.01</size></box>', '1 1 1 1'),
        visual('demo_lisere_d', '-0.02 -0.1565 0.09 0 0 0', '<box><size>0.27 0.002 0.01</size></box>', '1 1 1 1'),
        # face arrière orange avec deux feux rouges
        visual('demo_arriere', '-0.153 0 0.08 0 0 0', '<box><size>0.004 0.30 0.06</size></box>', ORANGE),
        visual('demo_feu_g', '-0.157 0.11 0.085 0 0 0', '<sphere><radius>0.012</radius></sphere>', '1 0.05 0.05 1', '0.9 0 0 1'),
        visual('demo_feu_d', '-0.157 -0.11 0.085 0 0 0', '<sphere><radius>0.012</radius></sphere>', '1 0.05 0.05 1', '0.9 0 0 1'),
        # petite antenne
        visual('demo_antenne', '-0.13 0.11 0.2 0 0 0', '<cylinder><radius>0.003</radius><length>0.16</length></cylinder>',
               '0.1 0.1 0.1 1'),
        visual('demo_antenne_bout', '-0.13 0.11 0.28 0 0 0', '<sphere><radius>0.007</radius></sphere>', '0.9 0.1 0.1 1'),
    ]:
        base.append(v)

    # sans en-tête <?xml ... encoding?> : spawn_entity.py (lxml) refuse une chaîne qui en contient un
    (dst / 'model.sdf').write_text(ET.tostring(root, encoding='unicode'))
    (dst / 'model.config').write_text(
        '<?xml version="1.0"?>\n<model>\n  <name>TurtleBot3 Waffle Pi (démo)</name>\n  <version>1.0</version>\n'
        '  <sdf version="1.6">model.sdf</sdf>\n  <description>waffle_pi habillé pour les vidéos (visuels seulement)'
        '</description>\n</model>\n')
    print(f'Robot de démonstration écrit dans {dst} (copie de {src} + éléments visuels)')


if __name__ == '__main__':
    main()
