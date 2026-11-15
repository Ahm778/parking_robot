#!/usr/bin/env python3
"""Petite fenêtre de démonstration : affiche les véhicules garés (plaque tunisienne + type) et lance
la recherche du véhicule choisi par le robot.

Usage (Gazebo et Nav2 déjà lancés, PARKING_DEMO=1) :   python3 scripts/choisir_vehicule.py
Normalement lancée par scripts/demo_choix.sh, qui démarre tout le reste.
"""
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import yaml

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from parking_robot.plates import render_plate_png  # noqa: E402

NOMS = {'hatchback_red': 'Citadine rouge', 'hatchback_blue': 'Citadine bleue', 'hatchback': 'Citadine grise',
        'suv': 'SUV gris', 'suv_red': 'SUV rouge', 'pickup': 'Pick-up gris'}
METHODES = [('Hybride A* + DRL (M3)', 'hybrid'), ('A* + DWB (M1)', 'classic'), ('DRL seul (M2)', 'drl')]
CACHE = Path('/tmp/parking_plaques_ui')


def load_cars():
    with open(PKG / 'config' / 'parking_slots.yaml') as f:
        data = yaml.safe_load(f)
    cars = [s for s in data['slots'] if s.get('gt_car')]
    return sorted(cars, key=lambda s: s['gt_plate'])


class App:
    def __init__(self, root):
        self.root, self.proc, self.lines = root, None, queue.Queue()
        root.title('Robot Parking — choisir le véhicule')
        root.configure(bg='#1f2430')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background='#1f2430')
        style.configure('Card.TFrame', background='#2b3142')
        style.configure('TLabel', background='#1f2430', foreground='#e8e8e8', font=('DejaVu Sans', 11))
        style.configure('Card.TLabel', background='#2b3142', foreground='#e8e8e8', font=('DejaVu Sans', 10))
        style.configure('Title.TLabel', font=('DejaVu Sans', 16, 'bold'))
        style.configure('Status.TLabel', font=('DejaVu Sans', 12, 'bold'), foreground='#ffb347')
        style.configure('TRadiobutton', background='#1f2430', foreground='#e8e8e8')
        style.configure('TCheckbutton', background='#1f2430', foreground='#e8e8e8')
        style.configure('Go.TButton', font=('DejaVu Sans', 10, 'bold'))

        top = ttk.Frame(root, padding=12)
        top.pack(fill='x')
        ttk.Label(top, text='Quel véhicule le robot doit-il retrouver ?', style='Title.TLabel').pack(anchor='w')
        ttk.Label(top, text='Cliquez sur « Chercher » sous la plaque. Le robot ne connaît que la plaque, '
                            'pas l\'emplacement.').pack(anchor='w', pady=(2, 8))
        opts = ttk.Frame(top)
        opts.pack(anchor='w')
        ttk.Label(opts, text='Méthode de navigation :').pack(side='left', padx=(0, 8))
        self.method = tk.StringVar(value='hybrid')
        for label, val in METHODES:
            ttk.Radiobutton(opts, text=label, value=val, variable=self.method).pack(side='left', padx=4)
        self.show_slot = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='Montrer l\'emplacement (pour le présentateur)', variable=self.show_slot,
                        command=self.refresh_slots).pack(anchor='w', pady=(6, 0))

        grid = ttk.Frame(root, padding=(12, 4))
        grid.pack(fill='both', expand=True)
        CACHE.mkdir(exist_ok=True)
        self.images, self.slot_labels = [], []
        for i, car in enumerate(load_cars()):
            card = ttk.Frame(grid, style='Card.TFrame', padding=8)
            card.grid(row=i // 5, column=i % 5, padx=6, pady=6, sticky='nsew')
            png = CACHE / f"{car['gt_plate']}.png"
            if not png.exists():
                render_plate_png(car['gt_plate'], str(png), px_w=230, px_h=50)
            img = tk.PhotoImage(file=str(png))
            self.images.append(img)
            ttk.Label(card, image=img, style='Card.TLabel').pack()
            ttk.Label(card, text=f"Plaque {car['gt_plate']}", style='Card.TLabel',
                      font=('DejaVu Sans', 10, 'bold')).pack(pady=(4, 0))
            ttk.Label(card, text=NOMS.get(car['gt_car'], car['gt_car']), style='Card.TLabel').pack()
            slot = ttk.Label(card, text=f"place {car['id']}", style='Card.TLabel', foreground='#8fd18f')
            self.slot_labels.append(slot)
            ttk.Button(card, text='Chercher', style='Go.TButton',
                       command=lambda c=car: self.search(c)).pack(pady=(6, 0), fill='x')

        bottom = ttk.Frame(root, padding=12)
        bottom.pack(fill='x')
        self.status = tk.StringVar(value='En attente : choisissez un véhicule.')
        ttk.Label(bottom, textvariable=self.status, style='Status.TLabel', wraplength=900).pack(side='left')
        ttk.Button(bottom, text='Arrêter la recherche', command=self.stop).pack(side='right')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(200, self.poll)

    def refresh_slots(self):
        for lab in self.slot_labels:
            if self.show_slot.get():
                lab.pack(before=lab.master.winfo_children()[-1])
            else:
                lab.pack_forget()

    def search(self, car):
        self.stop()
        plate, method = car['gt_plate'], self.method.get()
        self.status.set(f'Recherche de la plaque {plate} ({NOMS.get(car["gt_car"], "")}) — méthode {method}…')
        cmd = ['ros2', 'launch', 'parking_robot', 'mission.launch.py', 'color:=aucune', f'plate:={plate}',
               f'method:={method}', 'pedestrian:=true', 'blocker:=true', 'pause_s:=1.0']
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     start_new_session=True)
        threading.Thread(target=self.read, args=(self.proc,), daemon=True).start()

    def read(self, proc):
        for line in proc.stdout:
            if '[mission_manager]:' in line:
                self.lines.put(line.split('[mission_manager]:', 1)[1].strip())

    def poll(self):
        while not self.lines.empty():
            msg = self.lines.get()
            if not msg.startswith(('Navigating', 'Nav2 is ready')):
                self.status.set(msg)
        self.root.after(200, self.poll)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.status.set('Recherche arrêtée.')
        self.proc = None

    def close(self):
        self.stop()
        self.root.destroy()


def main():
    if os.environ.get('PARKING_DEMO') != '1':
        print('Attention : PARKING_DEMO=1 n\'est pas défini (lancez plutôt scripts/demo_choix.sh)')
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
