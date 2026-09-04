# Compilation locale du rapport

Tout est prêt sauf **une** étape que je ne peux pas faire à ta place :
l'installation de LaTeX demande le mot de passe sudo.

## 1. Installer LaTeX (une seule fois, ~10 min, ~2 Go)

Colle cette ligne dans Claude Code en la préfixant de `!`, ou dans un
terminal :

```
sudo apt-get update && sudo apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-lang-french texlive-pictures texlive-bibtex-extra biber latexmk
```

Ce jeu couvre exactement ce que le préambule demande :

| Paquet apt | Fournit |
|---|---|
| `texlive-latex-recommended` | geometry, fancyhdr, setspace, booktabs, parskip |
| `texlive-latex-extra` | titlesec, multirow, chngcntr, float, biblatex |
| `texlive-fonts-recommended` | lmodern |
| `texlive-lang-french` | babel french |
| `texlive-pictures` | tikz / pgf |
| `texlive-bibtex-extra` + `biber` | la bibliographie (backend biber) |
| `latexmk` | l'enchaînement automatique des passes |

`texlive-full` (~6 Go) marche aussi mais n'apporte rien de plus ici.

## 2. Compiler

```
cd /home/a131/Desktop/Project/rapport_build
./build.sh              # compilation complète
./build.sh fast         # une passe, ~3 s, pour relire du texte
./build.sh watch        # recompile à chaque sauvegarde
./build.sh clean
```

`rapport.tex` est un **lien** vers `../rapport.txt` : tu continues d'éditer
`rapport.txt` comme aujourd'hui, il n'y a pas de copie à resynchroniser.

## 3. Ce qui manque encore

Le document compile dès maintenant, mais deux choses sont provisoires.

### La bibliographie

`reference.bib` est un **faux** fichier que j'ai généré pour que la
compilation aboutisse. Ses 15 entrées portent le titre
`[[REFERENCE PROVISOIRE : <clé>]]`, impossible à rater dans le PDF.

À remplacer par le vrai, depuis Overleaf : `Menu → Download → Source`,
puis extraire `reference.bib` ici.

### Les figures

18 figures sont référencées. Six pointent vers les vrais fichiers déjà
présents dans le projet (liens symboliques) :

| Figure du document | Source réelle |
|---|---|
| `resnet50_post_prune.png` | `work/plots/post_pruning_accuracy/resnet50_post_prune_acc.png` |
| `resnet50_ft_trajectory.png` | `work/plots/convergence/resnet50_convergence.png` |
| `resnet50_pareto.png` | `work/curves/resnet50/pareto_accuracy_vs_latency.png` |
| `latency_vs_ntpu_resolution.png` | `multi_tpu/synthetic_models/outputs/bench_full/analysis/latency_vs_ntpu_w32_26MB.png` |
| `precision_throughput_tradeoff.png` | `wave_bench/results/G_precision_perf/` |
| `precision_latency_tradeoff.png` | idem |

Les douze autres sont des **cartouches rouges « FIGURE MANQUANTE »**
portant le nom du fichier attendu, pour que la mise en page reste lisible :

- **Trois panneaux à composer** (`accuracy_vs_compression_panel`,
  `speedup_vs_compression_panel`, `pipeline_vs_parallel_panel`). Les PNG
  par modèle existent sous `work/curves/<modele>/` et
  `wave_bench/results/B_pipeline_vs_parallel/per_model/` ; il reste à les
  assembler. C'est ce que signalent les `% [À FAIRE]` du document.
- **Neuf illustrations et logos** (`conv`, `systolic`, `edge_tpu`,
  `cifar100`, `BDD100K`, `dependence`, `laas`, `LAAS`, `sorbonne`), qui
  n'existent que sur Overleaf. Elles arriveront avec le zip de l'étape 3.

Déposer un vrai fichier au même nom dans ce dossier remplace
automatiquement le cartouche.

## 4. Vitesse

C'était le point de départ. En local, sur ce document :

- `./build.sh fast` : une passe, quelques secondes — le bon mode pour
  vérifier une reformulation.
- `./build.sh` : quatre passes plus biber, quelques dizaines de secondes.
- `./build.sh watch` : recompile tout seul à chaque sauvegarde.

Aucune limite de temps, contrairement à Overleaf.
