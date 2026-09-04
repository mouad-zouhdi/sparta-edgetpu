#!/usr/bin/env python3
"""
make_annexe_tables.py — met les tables de l'annexe A au format francais.

Les fragments de work/final_tables_and_plots/tables/ sont produits par un
script anterieur qui ecrit les decimales avec un point. Le reste du rapport
emploie la virgule. Ce script en produit des copies francisees dans
figs/tables/, sans toucher aux originaux, que d'autres analyses relisent.

Seuls les points encadres par deux chiffres sont convertis. « SqueezeNet 1.1 »
est un nom de modele et non un nombre : il est protege avant la substitution.
"""
from pathlib import Path
import re

SRC = Path("/home/a131/Desktop/Project/work/final_tables_and_plots/tables")
OUT = Path("figs/tables"); OUT.mkdir(parents=True, exist_ok=True)

GARDE = "\x00SQUEEZE\x00"          # jeton temporaire, absent des sources

# Les fragments nomment les criteres par leur identifiant de code, en
# chasse fixe. Le corps du rapport emploie des noms propres : on aligne.
CRITERES = {
    r"\texttt{magnitude\_l1}": "Magnitude L1",
    r"\texttt{magnitude\_l2}": "Magnitude L2",
    r"\texttt{bn\_scale}":     "BN-Scale",
    r"\texttt{fpgm}":          "FPGM",
    r"\texttt{taylor}":        "Taylor",
    r"\texttt{obdc}":          "OBD-C",
    r"\texttt{random}":        "Random",
}
n_fic = n_sub = 0
for f in sorted(SRC.glob("*.tex")):
    t = f.read_text()
    t = t.replace("SqueezeNet 1.1", GARDE)
    # En mode mathematique une virgule est une ponctuation et LaTeX lui ajoute
    # une espace : « 74, 37 ». Il faut l'accolader. On decoupe donc sur les $,
    # les fragments n'en contenant aucun d'echappe.
    morceaux = t.split("$")
    k = 0
    for i, m in enumerate(morceaux):
        virgule = "{,}" if i % 2 else ","
        morceaux[i], n = re.subn(r"(?<=\d)\.(?=\d)", virgule, m)
        k += n
    t = "$".join(morceaux)
    t = t.replace(GARDE, "SqueezeNet 1.1")
    for code, nom in CRITERES.items():
        t = t.replace(code, nom)
    (OUT / f.name).write_text(t)
    n_fic += 1; n_sub += k
print(f"{n_fic} fragments francises dans {OUT}, {n_sub} décimales converties")
