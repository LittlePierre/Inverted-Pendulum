# Inverted-Pendulum

## Vue d'ensemble

Ce projet implémente un **contrôle du pendule inversé** via deux approches :
1. **pendule.py** : Simulation simple avec interface graphique (Pygame)
2. **pendule_agent.py** : Contrôle intelligent par un agent d'apprentissage par renforcement (DQN - Deep Q-Network)

---

## 1. Architecture générale

### Objectif physique
Maintenir un bâton (pendule) en équilibre vertical en contrôlant le mouvement horizontal d'une base. Le bâton commence toujours penché et doit être relevé puis maintenu debout.

### Paramètres physiques
- **Longueur du pendule (l)** : 0.35 m
- **Masse (m)** : 1 kg
- **Gravité (g)** : 9.81 m/s²
- **Pas de temps (Δt)** : 0.01 s (100 FPS)
- **Échelle d'affichage** : 100 pixels pour 0.5 m

---

## 2. pendule.py - Simulation basique avec interface Pygame

### Structure MVC (Modèle-Vue-Contrôleur)

#### **Classe `model`** - Le modèle physique
```
Propriétés d'état :
  - theta (θ) : angle du pendule (radians)
  - x : position horizontale du chariot (m)
  - vx : vitesse horizontale du chariot
  - ax : accélération horizontale
  - vTheta : vitesse angulaire du pendule
  - aTheta : accélération angulaire
```

**Méthode `ApplyMove(newX)`** : Simule le mouvement physique
- Calcule la nouvelle vélocité horizontale : `vx = (newX - x) / Δt`
- Calcule l'accélération : `ax = (vx - vx_prev) / Δt`
- Applique la **dynamique du pendule inversé** :
  - La position du chariot produit une force qui fait basculer le pendule
  - Formule : `aTheta = (g/2 × sin(θ) - 0.5 × ax × cos(θ)) × 3/l`
  - Avec amortissement : `vTheta = vTheta × 0.99 + aTheta × Δt`

#### **Classe `view`** - L'interface graphique (Pygame)
- Affiche une fenêtre 800×600 pixels
- Charge l'image `Baton.png` (le bâton du pendule)
- Positionne le bâton selon `x` et `theta`
- Gère les événements clavier (Q pour quitter)
- Affiche les informations d'état

#### **Classe `controller`** - Le contrôleur
- Boucle principale : lie le modèle à la vue
- Applique une **formule de contrôle simple** :
  ```
  deltax = θ + 0.1×vx + 0.05×x
  newx = x + deltax × 0.1
  ```
  Cette formule empirique essaie de stabiliser le pendule en réagissant à l'angle, la vitesse et la position.

### Utilisation
```bash
python pendule.py
```
Lance une simulation avec le contrôleur empirique (ne fonctionne que partiellement).

---

## 3. pendule_agent.py - Agent RL avec Deep Q-Network

### Architecture complète

#### **3.1 Réseau de neurones (classe `NeuralNetwork`)**

Réseau feedforward **implémenté from scratch en Python pur** (sans TensorFlow/PyTorch).

**Architecture :**
- Couches cachées avec activation **ReLU** : `max(0, z)`
- Couche de sortie **linéaire** (pas d'activation)
- Initialisation des poids : **Xavier** pour la convergence

**Entraînement par batch (rétropropagation) :**
1. **Passe avant** : propage les données à travers le réseau
2. **Calcul de l'erreur** : MSE (Mean Squared Error) entre sortie et cible
3. **Rétropropagation** : calcule les gradients de chaque poids
4. **Clipping de gradients** : limite les gradients à ±1.0 pour éviter l'instabilité
5. **Mise à jour** : `poids -= lr × gradient`

#### **3.2 Replay Buffer (classe `ReplayBuffer`)**

Buffer circulaire stockant les expériences : `(état, action, récompense, état_suivant, fin_episode)`

- **Capacité** : 20 000 expériences
- **Permet** : d'entraîner le réseau sur des mini-batches tirés aléatoirement
- **Bénéfice** : casse la corrélation temporelle et stabilise l'apprentissage

#### **3.3 Agent DQN (classe `DQNAgent`)**

**Deep Q-Network** : apprend à évaluer la qualité des actions.

**Composants clés :**
- **Q-network (`q`)** : évalue les actions Q(s,a) → entrainé
- **Target network (`qt`)** : copie de `q` utilisée pour le calcul de cible (stabilité)
- **Epsilon-greedy** : exploration vs exploitation
  - ε = 1.0 (exploration totale) → 0.02 (exploitation maximale)
  - ε décroît par 0.995 chaque épisode
  - P(action aléatoire) = ε

**Actions discrètes** : 9 mouvements horizontaux
```
[-0.05, -0.02, -0.01, -0.003, 0.0, 0.003, 0.01, 0.02, 0.05] m/pas
```

**Entraînement (Q-Learning) :**
- Cible : `Q_target = r + γ × max_a Q_target(s', a)` si pas terminal, sinon `Q_target = r`
- Mise à jour : poids du Q-network entraîné sur la différence `Q(s,a) - Q_target`
- Gamma (γ) = 0.99 (facteur de discount : les récompenses futures comptent)

**Synchronisation** :
- Chaque 100 étapes : `Q_target ← Q` (actualise le réseau cible)

#### **3.4 Modèle physique (classe `model`)**

Identique à `pendule.py`, avec méthode `reset()` pour initialiser l'angle.

#### **3.5 Environnement RL (classe `PendulumEnv`)**

Enveloppe le modèle physique pour la boucle RL.

**État** : `[x, vx, ax, θ, vθ]` (5 dimensions)

**Encodage de l'état** (normalisation pour le réseau) :
```python
sin(θ), cos(θ),           # Position angulaire circulaire
vθ/10, x/2, vx/5          # Vitesses et position normalisées
```

**Récompense** :
- **Pendule renversé** (|θ| > 60°) : -2.0 → fin
- **Hors limites** (|x| > 2.5 m) : -2.0 → fin
- **Timeout** (200 pas) : reward = cos(θ) → fin
- **Normal** : `1.0 + cos(θ) - 0.01×x²` (bonus rester debout + stabilité)

#### **3.6 Apprentissage avec curriculum**

Augmente progressivement la difficulté (angle initial plus grand) pour mieux apprendre :
- Episodes 0-80% : angle initial de 10° à 55°
- Episodes 80%-100% : angle jusqu'à 55°
- 20% d'episodes "faciles" (10°) pour éviter l'oubli catastrophique

#### **3.7 Fonctions d'entraînement**

**`train(num_episodes)` :**
1. Crée l'environnement et l'agent
2. Pour chaque épisode :
   - Tire un angle initial selon le curriculum
   - Boucle : action → transitions → entraînement du réseau
   - Décroît ε (exploration)
3. Tous les 50 épisodes : évalue les performances (ε=0 pour tester)
4. Sauvegarde le meilleur modèle

**`evaluate(agent, n_episodes)` :**
- Teste l'agent sans exploration (ε=0)
- Retourne le reward moyen

**`test(num_episodes)` :**
- Charge le modèle entraîné
- Teste sur 10 épisodes avec angles aléatoires (±50°)
- Compte succès (reste debout > 100 pas sur 500)

#### **3.8 Interface graphique (classes `view`, `controller`)**

**`view`** : Affichage Pygame identique à `pendule.py`

**`controller`** :
- Charge les poids du modèle entraîné
- Boucle : état → encodage → agent prend action → affichage

---

## 4. Modes d'utilisation

### Entraînement (DQN)
```bash
python pendule_agent.py train 2000
```
- Entraîne l'agent pendant 2000 épisodes
- Sauvegarde le meilleur modèle dans `pendulum_weights.json`
- Affiche la progression tous les 50 épisodes

### Test (évaluation)
```bash
python pendule_agent.py test
```
- Charge le modèle entraîné
- Teste 10 épisodes avec angles initials aléatoires
- Affiche taux de succès et récompenses

### Visualisation (GUI)
```bash
python pendule_agent.py play
```
ou
```bash
python pendule_agent.py
```
- Lance l'interface Pygame
- L'agent contrôle le pendule en temps réel
- Affiche l'angle et la position du chariot

---

## 5. Comparaison : pendule.py vs pendule_agent.py

| Aspect | pendule.py | pendule_agent.py |
|--------|-----------|-----------------|
| Contrôle | Heuristique simple (formule fixe) | Réseau de neurones appris |
| Flexibilité | Peu adaptable | S'adapte aux conditions |
| Apprentissage | Aucun | Par renforcement (DQN) |
| Performance | Mediocre | Excellente (si bien entraîné) |
| Poids | Fixes | Sauvegardables/chargeables |
| Complexité | Simple (~100 lignes) | Complète (~630 lignes) |

---

## 6. Fichiers

- **`pendule.py`** : Simulation simple + contrôleur heuristique
- **`pendule_agent.py`** : Agent DQN complet avec entraînement/test/visualisation
- **`pendulum_weights.json`** : Poids du meilleur modèle entraîné (généré après `train`)
- **`Baton.png`** : Image du bâton pour l'affichage Pygame

---

## 7. Résumé technique

**pendule_agent.py** implémente un **système d'apprentissage par renforcement complet** :
- Réseau de neurones from scratch (rétropropagation)
- Experience replay pour la stabilité
- Double network (Q-network + Target network)
- Epsilon-decay pour l'exploration
- Apprentissage avec curriculum pour progresser graduellement
- Évaluation périodique avec sauvegarde du meilleur modèle

**Objectif** : Le pendule devrait rester debout indéfiniment (ou très longtemps) une fois bien entraîné.
