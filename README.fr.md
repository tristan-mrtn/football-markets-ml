# Les données publiques du football peuvent-elles battre les bookmakers ?

**Une étude en évaluation glissante, auditée contre les fuites d'information, sur les 5 grands championnats européens (2021-2026)**

*English version: [README.md](README.md)*

J'ai construit une chaîne complète (collecte des données, 220 variables conçues à la main,
modèles de gradient boosting, transformer sur les compositions d'équipe) pour prédire l'issue des
matchs de football, puis je l'ai évaluée comme on doit évaluer un prévisionniste : **sur environ
3 800 matchs jamais vus pendant sa conception ni ses réglages, face au marché des paris qui
disposait exactement de la même information.**

La réponse courte est **non** : le modèle est bon, bien meilleur qu'un modèle Dixon-Coles
classique, mais le marché l'est davantage, et le modèle n'apporte aucune information que les
cotes ne contiennent déjà. Les réponses les plus intéressantes portent sur le *pourquoi*, et sur
les pièges rencontrés en chemin.

## Résultats principaux

| | |
|---|---|
| **Le modèle est solide…** | log-loss **0,978** contre **0,992** pour Dixon-Coles et **1,074** pour une prédiction constante |
| **…mais le marché fait mieux** | cotes d'ouverture **0,970**, de clôture **0,968** ; le modèle est significativement moins bon (+0,009, IC 95 % [+0,004 ; +0,014]) |
| **Il n'apporte rien au marché** | poids appris sur la période de réglage : **0,06** pour le modèle contre 0,98 pour les cotes ; le mélange égale le marché (−0,0002, IC 95 % [−0,0007 ; +0,0003]) |
| **Pas d'avantage pour parier** | le modèle seul a une valeur de clôture (CLV) négative (−2 % à −9 %) et des rendements négatifs à tous les seuils |
| **Les compositions comptent peu** | connaître le onze de départ n'améliore la log-loss que de 0,001 (non significatif) |
| **Ce qui contient du signal** | seul le **contexte du match** (enjeu, Coupes d'Europe, repos, changement d'entraîneur, âge de l'effectif) aide significativement à lui seul (+0,007 de log-loss) ; tactique, attaque du onze, automatismes, défense et stats de match apportent ~0,002 chacun mais se recoupent ; forme, face-à-face, chance, disponibilité et xG à score égal n'apportent rien |
| **Biais du marché** | un net **biais favori-outsider** : les issues cotées sous 10 % n'arrivent que 6 % du temps (−24 % de rendement), les favoris ne perdent que 1 à 3 % |
| **Piège méthodologique** | mesurer la CLV avec un retrait de marge au prorata fait apparaître un faux avantage de **+10,7 %** ; la méthode de la puissance donne **−0,4 %** |

<p align="center"><img src="docs/figures/market_gap.png" width="820"></p>

## Pourquoi ce projet est rigoureux

* **Discipline temporelle stricte.** Les variables sont produites en rejouant tous les matchs dans
  l'ordre chronologique ; les matchs joués à la même heure sont tous calculés avant qu'aucun ne
  mette à jour les historiques.
* **Évaluation glissante.** Réentraînement tous les 2 mois sur le seul passé ; réglages,
  calibration et poids des mélanges choisis sur une période distincte (janv. 2023 à juin 2024) ;
  la période d'évaluation (juil. 2024 à oct. 2026) ne sert à aucune décision.
* **Comparaison équitable au marché.** Les modèles sans compositions sont comparés aux cotes
  d'*ouverture*, ceux avec compositions aux cotes de *clôture*.
* **Audit automatique des fuites** (`make leakage`) : recalculer les variables sur des données
  tronquées, ou sans un match joué en même temps, ne doit rien changer, et c'est le cas sur
  14 080 lignes et 154 857 jetons joueurs.
* **Chaque comparaison a un intervalle de confiance** (bootstrap apparié sur les matchs).
* **Les bugs sont documentés, pas cachés.** Cinq bugs de fuite ou de mesure ont été trouvés et
  corrigés ; ils sont listés dans [docs/METHODOLOGY.md](docs/METHODOLOGY.md#bugs-found-and-fixed-during-the-project).

## Résultats détaillés

### Qualité des prévisions : période d'évaluation, 3 799 matchs avec cotes complètes

| Prévisionniste | Information | Log-loss ↓ | Brier ↓ |
|---|---|---|---|
| Constante (fréquences) | aucune | 1,0740 | 0,6500 |
| Dixon-Coles, buts seuls | buts passés | 0,9923 | 0,5913 |
| LightGBM, 164 variables | avant compositions | 0,9784 | 0,5825 |
| LightGBM, 220 variables | avec compositions | 0,9774 | 0,5818 |
| LightGBM + cotes d'ouverture en variables | avant compositions | 0,9705 | 0,5771 |
| Bookmakers, ouverture | avant compositions | 0,9698 | 0,5770 |
| Mélange modèle ⊕ marché d'ouverture | avant compositions | 0,9696 | 0,5769 |
| Bookmakers, clôture | avec compositions | 0,9677 | 0,5756 |
| Mélange modèle ⊕ marché de clôture | avec compositions | 0,9675 | 0,5755 |

### Une piste exploratoire, pas un résultat

Une règle de pari se distingue : le mélange modèle ⊕ marché, en pariant à la **meilleure cote
d'ouverture disponible** quand elle dépasse la cote juste du mélange de 2 %, donne +9,2 % de
rendement (IC 95 % [+1,6 % ; +17,1 %]) et +2,6 % de CLV sur 622 paris. Je n'en tire **pas**
d'avantage : des dizaines de règles ont été testées (une ou deux doivent paraître significatives
par hasard), la « meilleure cote » inclut probablement Betfair Exchange dont la commission n'est
pas modélisée, et le poids du modèle dans le mélange est minime. C'est la première chose qu'un
test « à blanc » sur des matchs futurs devra vérifier.

### Quelles statistiques comptent ?

Chaque famille est ajoutée à la seule force des équipes (bleu) ou retirée du modèle complet
(orange), dans le protocole glissant, sur la période de réglage uniquement. L'ensemble des 220
variables améliore la force des équipes seule de 0,006 (0,988 → 0,982), mais seul le contexte
apporte une information qu'aucune autre famille ne fournit déjà.

<p align="center"><img src="docs/figures/feature_families.png" width="820"></p>

### Le marché

<p align="center">
<img src="docs/figures/favourite_longshot.png" width="420">
<img src="docs/figures/by_league.png" width="820">
</p>

Chiffres complets : [`results/wf_rapport.txt`](results/wf_rapport.txt) et
[`results/market_study.txt`](results/market_study.txt). Méthodologie détaillée (en anglais) :
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Reproduire

```bash
git clone https://github.com/tristan-mrtn/football-markets-ml && cd football-markets-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test        # 18 tests unitaires, sans données
make data        # téléchargement des données (~1-2 h, volontairement lent)
make features leakage evaluate report market ablation figures
```

Les données brutes ne sont pas redistribuées ; `make data` les télécharge. Tout tourne sur le
processeur d'un ordinateur portable, sans GPU.

## Limites et suites

* Pas de données défensives (tacles, interceptions) ; coupes nationales absentes de la fatigue.
* Bookmakers français absents des cotes ; commission des bourses de paris non modélisée.
* Suites : analyse bookmaker par bookmaker avec Betfair Exchange comme référence, test « à blanc »
  sur des matchs joués après le gel du modèle, petit modèle de langage lisant l'actualité d'avant-match.

Projet de recherche : rien ici ne constitue un conseil de pari. Code sous [licence MIT](LICENSE).

## Transparence

Ce projet a été développé avec l'aide de **Claude Opus 5.5** (Anthropic) qui, par souci
d'optimisation, a écrit l'essentiel du code et de la documentation sous ma direction. J'ai défini
le projet et ses objectifs, proposé les hypothèses footballistiques à tester (blessures,
automatismes entre joueurs, formations, retour à la moyenne…), exigé le protocole d'évaluation
rigoureux hors échantillon et la comparaison au marché, et relu les résultats.
