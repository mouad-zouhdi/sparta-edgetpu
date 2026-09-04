#!/usr/bin/env bash
# Compilation locale du rapport.
#
#   ./build.sh          compilation complete (texte + biblio + renvois)
#   ./build.sh fast     une seule passe, sans biblio  -- pour relire du texte
#   ./build.sh watch    recompile a chaque sauvegarde de rapport.txt
#   ./build.sh clean    efface les fichiers intermediaires
#
# La source est ../rapport.txt, atteinte via le lien rapport.tex.
set -euo pipefail
cd "$(dirname "$0")"

command -v latexmk >/dev/null || {
    echo "latexmk introuvable. Voir INSTALL.md pour la ligne d'installation." >&2
    exit 1
}

case "${1:-full}" in
  fast)
    # -interaction=nonstopmode : ne s'arrete pas sur les erreurs recuperables
    pdflatex -interaction=nonstopmode -halt-on-error rapport.tex >/dev/null || {
        echo "--- erreurs ---"; grep -A3 "^!" rapport.log | head -40; exit 1; }
    echo "rapport.pdf mis a jour (passe unique, renvois et biblio possiblement perimes)"
    ;;
  watch)
    latexmk -pdf -pvc -interaction=nonstopmode rapport.tex
    ;;
  clean)
    latexmk -C rapport.tex 2>/dev/null || true
    rm -f rapport.bbl rapport.run.xml rapport.synctex.gz
    echo "nettoye"
    ;;
  full|*)
    latexmk -pdf -interaction=nonstopmode rapport.tex
    echo
    echo "=== rapport.pdf genere ==="
    if grep -q "REFERENCE PROVISOIRE" reference.bib 2>/dev/null; then
        echo "  ! biblio provisoire : recuperer reference.bib depuis Overleaf"
    fi
    n=$(grep -c "FIGURE MANQUANTE" .placeholders 2>/dev/null || echo 0)
    [ "$n" -gt 0 ] && echo "  ! $n figures de remplacement (voir INSTALL.md)"
    ;;
esac
