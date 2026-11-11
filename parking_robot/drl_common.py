"""Éléments DRL partagés par l'entraînement (drl_env.py) et le nœud ROS (drl_controller.py).

Ne dépend que de NumPy : le nœud ROS n'a besoin ni de PyTorch ni de Gymnasium.
"""
import numpy as np

# Actions discrètes (v en m/s, w en rad/s) : avancer, tourner à gauche, à droite, ralentir, s'arrêter
ACTIONS = np.array([[0.22, 0.0], [0.10, 0.8], [0.10, -0.8], [0.08, 0.0], [0.0, 0.0]])
N_RAYS, MAX_R, DT, ROBOT_R = 24, 3.5, 0.2, 0.2      # 24 secteurs de 15°, portée lidar, pas de temps, rayon robot
REL_ANGLES = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)   # angle de chaque secteur / avant du robot


OBS_SIZE = 2 * N_RAYS + 4
DELTA_SCALE = 0.3      # variation max d'un secteur entre deux pas (m) : piéton à 0.7 m/s + robot


def build_obs(rays, prev_rays, d, ang, v, w):
    """Observation de 52 valeurs dans [-1, 1] : 24 distances, leur variation depuis le pas précédent
    (pour voir si un obstacle approche : piéton), distance et angle au but, v, w."""
    rays = np.minimum(rays, MAX_R)
    delta = np.clip((rays - np.minimum(prev_rays, MAX_R)) / DELTA_SCALE, -1.0, 1.0)
    return np.concatenate([rays / MAX_R * 2 - 1, delta,
                           [min(d, 10.0) / 5 - 1, ang / np.pi, v / 0.22 * 2 - 1, w / 0.8]]).astype(np.float32)


class NumpyPolicy:
    """Politique PPO exportée (policy.npz) : 2 couches tanh puis logits ; action = argmax (déterministe)."""

    def __init__(self, path):
        z = np.load(path)
        self.layers = [(z['w0'], z['b0']), (z['w1'], z['b1'])]
        self.head = (z['w2'], z['b2'])

    def act(self, obs):
        h = np.asarray(obs, dtype=np.float32)
        for w, b in self.layers:
            h = np.tanh(w @ h + b)
        return int(np.argmax(self.head[0] @ h + self.head[1]))
