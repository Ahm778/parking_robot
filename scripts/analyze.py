#!/usr/bin/env python3
"""Tableau récapitulatif des essais par scénario et par méthode (à partir de ~/results.csv).

Usage : python3 scripts/analyze.py [chemin/results.csv] [essais_prévus_par_case]
Les essais prévus mais absents du CSV (dépassement de temps) comptent comme des échecs.
"""
import os
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/results.csv')
planned = int(sys.argv[2]) if len(sys.argv) > 2 else 50
df = pd.read_csv(path)
for col in ('loc_success', 'false_positive', 'mission_success'):
    df[col] = df[col].astype(str).str.lower().eq('true')

g = df.groupby(['scenario', 'method'])
table = pd.DataFrame({
    'essais_enregistres': g.size(),
    'reussite_mission_%': 100 * g['mission_success'].sum() / planned,
    'reussite_localisation_%': 100 * g['loc_success'].sum() / planned,
    'faux_positifs_%': 100 * g['false_positive'].mean(),
    'temps_pour_trouver_s': g['time_to_find_s'].mean(),
    'ecart_type_temps_s': g['time_to_find_s'].std(),
    'distance_m': g['distance_m'].mean(),
    'collisions_total': g['collisions'].sum(),
}).round(1)
print(table.to_string())
table.to_csv(os.path.splitext(path)[0] + '_summary.csv')
