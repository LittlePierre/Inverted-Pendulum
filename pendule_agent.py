#!/usr/bin/env python3
"""Pendule inverse controle par Deep Q-Network (DQN).

Reseau de neurones et agent RL implementes from scratch (aucune bibliotheque ML).

Usage:
    python pendule_agent.py train [episodes]           - Entrainer l'agent (sans GUI)
    python pendule_agent.py train_gui [episodes]       - Entrainer l'agent (avec GUI)
    python pendule_agent.py test                       - Tester l'agent entraine (sans GUI)
    python pendule_agent.py play                       - Visualiser avec l'agent entraine (GUI)
    python pendule_agent.py                            - Identique a 'play'
"""

import math
import sys
import time
import json
import os
from random import random, randint, uniform, sample

LONGUEUR = 0.5  # m
SCALE = 100.0 / LONGUEUR
DELTAT = 1.0 / 100.0  # seconds
WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pendulum_weights.json")


# ===========================================================================
# Reseau de neurones - implementation pure Python
# ===========================================================================

class NeuralNetwork:
    """Reseau feedforward avec couches cachees ReLU et sortie lineaire."""

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))  # Xavier init
            self.weights.append(
                [[uniform(-limit, limit) for _ in range(fan_out)]
                 for _ in range(fan_in)]
            )
            self.biases.append([0.0] * fan_out)

    def predict(self, x):
        """Passe avant, retourne uniquement la sortie."""
        a = list(x)
        for i in range(self.n_layers):
            w, b = self.weights[i], self.biases[i]
            n_out = len(b)
            z = list(b)  # commence avec le biais
            for k in range(len(a)):
                ak, wk = a[k], w[k]
                for j in range(n_out):
                    z[j] += ak * wk[j]
            a = [max(0.0, v) for v in z] if i < self.n_layers - 1 else z
        return a

    def train_batch(self, batch_states, batch_targets, lr=0.001):
        """Entrainement sur un mini-batch par retropropagation. Retourne le MSE moyen."""
        bs = len(batch_states)
        if bs == 0:
            return 0.0

        # Accumulateurs de gradients
        gw = [[[0.0] * len(self.biases[i]) for _ in range(self.layer_sizes[i])]
              for i in range(self.n_layers)]
        gb = [[0.0] * len(self.biases[i]) for i in range(self.n_layers)]
        total_loss = 0.0

        for si in range(bs):
            inp, tgt = batch_states[si], batch_targets[si]

            # --- Passe avant ---
            acts = [list(inp)]
            pres = []
            a = list(inp)
            for i in range(self.n_layers):
                w, b = self.weights[i], self.biases[i]
                n_out = len(b)
                z = list(b)
                for k in range(len(a)):
                    ak, wk = a[k], w[k]
                    for j in range(n_out):
                        z[j] += ak * wk[j]
                pres.append(z)
                a = [max(0.0, v) for v in z] if i < self.n_layers - 1 else list(z)
                acts.append(a)

            # --- Gradient de sortie (MSE par action) ---
            out = acts[-1]
            n_o = len(out)
            delta = [0.0] * n_o
            for j in range(n_o):
                d = out[j] - tgt[j]
                total_loss += d * d
                # Pas de division par n_o : en DQN seule l'action prise a un
                # gradient non-nul, diviser affaiblirait le signal inutilement.
                delta[j] = 2.0 * d

            # --- Retropropagation ---
            for i in range(self.n_layers - 1, -1, -1):
                act_i = acts[i]
                nd = len(delta)
                ni = len(act_i)
                gwi, gbi = gw[i], gb[i]
                for r in range(ni):
                    ar = act_i[r]
                    gwir = gwi[r]
                    for c in range(nd):
                        gwir[c] += ar * delta[c]
                for c in range(nd):
                    gbi[c] += delta[c]

                if i > 0:
                    wi = self.weights[i]
                    np_ = len(wi)
                    nd_new = [0.0] * np_
                    for r in range(np_):
                        s = 0.0
                        wr = wi[r]
                        for c in range(nd):
                            s += wr[c] * delta[c]
                        nd_new[r] = s
                    pre = pres[i - 1]
                    delta = [nd_new[j] if pre[j] > 0 else 0.0 for j in range(np_)]

        # --- Application des gradients avec clipping ---
        scale = lr / bs
        clip = 1.0
        for i in range(self.n_layers):
            wi, bi = self.weights[i], self.biases[i]
            gwi, gbi = gw[i], gb[i]
            for r in range(len(wi)):
                wr, gwr = wi[r], gwi[r]
                for c in range(len(wr)):
                    g = gwr[c] * scale
                    if g > clip:
                        g = clip
                    elif g < -clip:
                        g = -clip
                    wr[c] -= g
            for c in range(len(bi)):
                g = gbi[c] * scale
                if g > clip:
                    g = clip
                elif g < -clip:
                    g = -clip
                bi[c] -= g

        n_outputs = len(batch_targets[0])
        return total_loss / (bs * n_outputs)

    def copy_from(self, other):
        """Copie les poids depuis un autre reseau."""
        for i in range(self.n_layers):
            for r in range(len(self.weights[i])):
                self.weights[i][r] = list(other.weights[i][r])
            self.biases[i] = list(other.biases[i])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'s': self.layer_sizes, 'w': self.weights, 'b': self.biases}, f)

    @classmethod
    def load_from(cls, path):
        with open(path, 'r') as f:
            d = json.load(f)
        nn = cls(d['s'])
        nn.weights = d['w']
        nn.biases = d['b']
        return nn


# ===========================================================================
# Replay Buffer
# ===========================================================================

class ReplayBuffer:
    """Buffer circulaire pour l'experience replay."""

    def __init__(self, capacity=20000):
        self.cap = capacity
        self.buf = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        item = (s, a, r, s2, done)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.cap

    def sample_batch(self, n):
        return sample(self.buf, min(n, len(self.buf)))

    def __len__(self):
        return len(self.buf)


# ===========================================================================
# Agent DQN
# ===========================================================================

class DQNAgent:
    """Agent Deep Q-Network avec experience replay et target network."""

    # Actions discretes : deplacement delta-x par pas de temps
    ACTIONS = [-0.05, -0.02, -0.01, -0.003, 0.0, 0.003, 0.01, 0.02, 0.05]

    def __init__(self, state_size=5, hidden=32, lr=0.0005):
        self.num_actions = len(self.ACTIONS)
        self.q = NeuralNetwork([state_size, hidden, hidden, self.num_actions])
        self.qt = NeuralNetwork([state_size, hidden, hidden, self.num_actions])
        self.qt.copy_from(self.q)
        self.mem = ReplayBuffer(20000)
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.02
        self.eps_decay = 0.995
        self.lr = lr
        self.batch_size = 16
        self.target_sync = 100
        self.train_every = 4
        self.steps = 0

    @staticmethod
    def encode(state):
        """Encode l'etat du modele [x,vx,ax,theta,vtheta] en entree reseau."""
        x, vx, _ax, theta, vtheta = state
        th = math.atan2(math.sin(theta), math.cos(theta))
        return [
            math.sin(th),
            math.cos(th),
            max(-1.0, min(1.0, vtheta / 10.0)),
            max(-1.0, min(1.0, x / 2.0)),
            max(-1.0, min(1.0, vx / 5.0)),
        ]

    def act(self, enc_state):
        """Selection d'action epsilon-greedy."""
        if random() < self.eps:
            return randint(0, self.num_actions - 1)
        q = self.q.predict(enc_state)
        best, best_v = 0, q[0]
        for i in range(1, len(q)):
            if q[i] > best_v:
                best_v, best = q[i], i
        return best

    def remember(self, s, a, r, s2, done):
        self.mem.push(s, a, r, s2, done)

    def learn(self):
        """Un pas d'entrainement avec experience replay."""
        if len(self.mem) < self.batch_size * 4:
            return 0.0
        self.steps += 1
        if self.steps % self.train_every != 0:
            return 0.0

        batch = self.mem.sample_batch(self.batch_size)
        states, targets = [], []
        for s, a, r, s2, done in batch:
            q = self.q.predict(s)
            if done:
                q_t = r
            else:
                q2 = self.qt.predict(s2)
                q_t = r + self.gamma * max(q2)
            tgt = list(q)
            tgt[a] = q_t
            states.append(s)
            targets.append(tgt)

        loss = self.q.train_batch(states, targets, lr=self.lr)
        if self.steps % self.target_sync == 0:
            self.qt.copy_from(self.q)
        return loss

    def end_episode(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def save(self, path):
        self.q.save(path)

    def load(self, path):
        self.q = NeuralNetwork.load_from(path)
        self.qt = NeuralNetwork(self.q.layer_sizes)
        self.qt.copy_from(self.q)


# ===========================================================================
# Modele physique
# ===========================================================================

class model(object):
    def __init__(self):
        self.g = 9.81
        self.m = 1.0
        self.l = 0.35
        self.deltaT = DELTAT
        self.reset()

    def reset(self, theta=None, x=0.0):
        self.theta = theta if theta is not None else 10.0 * math.pi / 180.0
        self.x = x
        self.vx = 0.0
        self.ax = 0.0
        self.vTheta = 0.0
        self.aTheta = 0.0
        self.lastx = x
        self.t0 = time.time()

    def ApplyMove(self, newX, deltaT=True):
        if deltaT:
            self.deltaT = DELTAT
        else:
            self.deltaT = time.time() - self.t0
            self.t0 = time.time()
        vx = (newX - self.x) / self.deltaT
        ax = (vx - self.vx) / self.deltaT
        aTheta = (self.g / 2.0 * math.sin(self.theta)
                  - 0.5 * ax * math.cos(self.theta)) * 3.0 / self.l
        self.vTheta = self.deltaT * aTheta + self.vTheta * 0.99
        self.theta = self.vTheta * self.deltaT + self.theta
        self.vx = vx
        self.ax = ax
        self.lastx = self.x
        self.x = newX
        self.aTheta = aTheta

    def getState(self):
        return [self.x, self.vx, self.ax, self.theta, self.vTheta]


# ===========================================================================
# Environnement RL
# ===========================================================================

class PendulumEnv:
    """Enveloppe le modele physique en environnement RL."""

    def __init__(self, max_steps=200, fall_angle=math.radians(60)):
        self.mdl = model()
        self.max_steps = max_steps
        self.fall_angle = fall_angle
        self.t = 0

    def reset(self, theta=None):
        self.mdl.reset(theta=theta)
        self.t = 0
        return self.mdl.getState()

    def step(self, action_idx):
        dx = DQNAgent.ACTIONS[action_idx]
        newx = max(-2.5, min(2.5, self.mdl.x + dx))
        self.mdl.ApplyMove(newx)
        self.t += 1
        s = self.mdl.getState()
        th_norm = math.atan2(math.sin(s[3]), math.cos(s[3]))
        fell = abs(th_norm) > self.fall_angle
        out_of_bounds = abs(s[0]) > 2.5
        timeout = self.t >= self.max_steps

        if fell or out_of_bounds:
            reward = -2.0
            done = True
        elif timeout:
            reward = math.cos(th_norm)
            done = True
        else:
            # Bonus pour rester debout + bonus cos(theta) [0..2]
            reward = 1.0 + math.cos(th_norm)
            reward -= 0.01 * s[0] * s[0]  # rester au centre
            done = False
        return s, reward, done


# ===========================================================================
# Evaluation (eps=0)
# ===========================================================================

def evaluate(agent, n_episodes=5, max_angle_deg=30):
    """Evalue la politique sans exploration. Retourne le reward moyen."""
    env = PendulumEnv(max_steps=200)
    old_eps = agent.eps
    agent.eps = 0.0
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset(uniform(-math.radians(max_angle_deg),
                                   math.radians(max_angle_deg)))
        es = agent.encode(state)
        ep_r = 0.0
        while True:
            a = agent.act(es)
            state, r, done = env.step(a)
            es = agent.encode(state)
            ep_r += r
            if done:
                break
        total += ep_r
    agent.eps = old_eps
    return total / n_episodes


# ===========================================================================
# Entrainement
# ===========================================================================

def train(num_episodes=2000, weights_path=WEIGHTS_FILE):
    """Entraine l'agent DQN avec apprentissage par curriculum."""
    env = PendulumEnv(max_steps=200, fall_angle=math.radians(70))
    agent = DQNAgent(lr=0.0005)

    if os.path.exists(weights_path):
        print(f"[train] Chargement des poids existants depuis {weights_path}")
        agent.load(weights_path)
        agent.eps = 0.3

    best_eval = -float('inf')
    history = []
    t_start = time.time()

    for ep in range(num_episodes):
        # Curriculum lineaire : angle max de 10 deg a 55 deg
        frac = ep / max(1, num_episodes - 1)
        max_a = math.radians(10 + 45 * min(1.0, frac / 0.8))

        # 20% d'episodes faciles pour eviter l'oubli catastrophique
        if random() < 0.2:
            theta0 = uniform(-math.radians(10), math.radians(10))
        else:
            theta0 = uniform(-max_a, max_a)
        state = env.reset(theta0)
        es = agent.encode(state)
        ep_r = 0.0

        while True:
            a = agent.act(es)
            state2, r, done = env.step(a)
            es2 = agent.encode(state2)
            agent.remember(es, a, r, es2, done)
            agent.learn()
            ep_r += r
            es = es2
            if done:
                break

        agent.end_episode()
        history.append(ep_r)

        # Evaluation periodique (performance sans exploration)
        if (ep + 1) % 50 == 0:
            eval_r = evaluate(agent, n_episodes=5, max_angle_deg=45)
            if eval_r > best_eval:
                best_eval = eval_r
                agent.save(weights_path)
            elapsed = time.time() - t_start
            avg = sum(history[-50:]) / min(50, len(history))
            print(f"  Ep {ep + 1:4d}/{num_episodes} | Train50: {avg:7.1f} | "
                  f"Eval(45deg): {eval_r:7.1f} | Eps: {agent.eps:.3f} | "
                  f"Best: {best_eval:7.1f} | Time: {elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n[train] Termine en {elapsed:.0f}s.")
    print(f"[train] Meilleur modele -> {weights_path}")

    # Test automatique
    print("\n--- Test automatique ---")
    test(10, weights_path)
    return agent


# ===========================================================================
# Entrainement avec GUI
# ===========================================================================

def train_gui(num_episodes=2000, weights_path=WEIGHTS_FILE):
    """Entraine l'agent DQN avec visualisation en temps reel (GUI)."""
    view_obj = view()
    model_obj = model()
    env = PendulumEnv(max_steps=200, fall_angle=math.radians(70))
    agent = DQNAgent(lr=0.0005)

    if os.path.exists(weights_path):
        print(f"[train_gui] Chargement des poids existants depuis {weights_path}")
        agent.load(weights_path)
        agent.eps = 0.3

    best_eval = -float('inf')
    history = []
    t_start = time.time()

    for ep in range(num_episodes):
        if not view_obj.continuer:
            break

        # Curriculum lineaire : angle max de 10 deg a 55 deg
        frac = ep / max(1, num_episodes - 1)
        max_a = math.radians(10 + 45 * min(1.0, frac / 0.8))

        # 20% d'episodes faciles pour eviter l'oubli catastrophique
        if random() < 0.2:
            theta0 = uniform(-math.radians(10), math.radians(10))
        else:
            theta0 = uniform(-max_a, max_a)
        state = env.reset(theta0)
        es = agent.encode(state)
        ep_r = 0.0

        while view_obj.continuer:
            a = agent.act(es)
            state2, r, done = env.step(a)
            es2 = agent.encode(state2)
            agent.remember(es, a, r, es2, done)
            agent.learn()
            ep_r += r
            es = es2

            # Mise a jour de la GUI chaque 4 pas
            if env.t % 4 == 0:
                model_obj.x = state2[0]
                model_obj.theta = state2[3]
                theta_deg = -model_obj.theta * 180.0 / math.pi
                view_obj.action(model_obj.x, theta_deg)
                view_obj.processFrame()

            if done:
                break

        agent.end_episode()
        history.append(ep_r)

        # Evaluation periodique (performance sans exploration)
        if (ep + 1) % 50 == 0:
            eval_r = evaluate(agent, n_episodes=5, max_angle_deg=45)
            if eval_r > best_eval:
                best_eval = eval_r
                agent.save(weights_path)
            elapsed = time.time() - t_start
            avg = sum(history[-50:]) / min(50, len(history))
            status_text = (f"Ep {ep + 1:4d}/{num_episodes} | Train50: {avg:7.1f} | "
                          f"Eval(45deg): {eval_r:7.1f} | Eps: {agent.eps:.3f} | "
                          f"Best: {best_eval:7.1f}")
            print(f"  {status_text} | Time: {elapsed:.0f}s")
            view_obj.dspText(status_text)

    elapsed = time.time() - t_start
    print(f"\n[train_gui] Termine en {elapsed:.0f}s.")
    print(f"[train_gui] Meilleur modele -> {weights_path}")
    view_obj.quit()


# ===========================================================================
# Test
# ===========================================================================

def test(num_episodes=10, weights_path=WEIGHTS_FILE):
    """Teste l'agent entraine sur des positions de depart aleatoires."""
    if not os.path.exists(weights_path):
        print(f"[test] Fichier de poids introuvable : {weights_path}")
        print("[test] Lancez d'abord : python pendule_agent.py train")
        return

    env = PendulumEnv(max_steps=500, fall_angle=math.radians(70))
    agent = DQNAgent()
    agent.load(weights_path)
    agent.eps = 0.0

    successes = 0
    total_r = 0.0

    for ep in range(num_episodes):
        theta0 = uniform(-math.radians(50), math.radians(50))
        state = env.reset(theta0)
        es = agent.encode(state)
        ep_r = 0.0
        upright = 0

        while True:
            a = agent.act(es)
            state, r, done = env.step(a)
            es = agent.encode(state)
            ep_r += r
            th = math.atan2(math.sin(state[3]), math.cos(state[3]))
            if abs(th) < math.radians(15):
                upright += 1
            if done:
                break

        total_r += ep_r
        ok = upright > 100
        if ok:
            successes += 1
        steps_done = env.t
        print(f"  Test {ep + 1:2d} | theta0={math.degrees(theta0):7.1f} deg | "
              f"Reward: {ep_r:7.1f} | Steps: {steps_done:3d} | Debout: {upright:3d} | "
              f"{'OK' if ok else 'ECHEC'}")

    print(f"\n[test] Succes: {successes}/{num_episodes} | "
          f"Reward moyen: {total_r / num_episodes:.1f}")


# ===========================================================================
# Vue (pygame) - charge uniquement pour le mode GUI
# ===========================================================================

class view(object):
    def __init__(self):
        import pygame
        from pygame.locals import QUIT
        self._pg = pygame
        self._QUIT = QUIT
        pygame.init()
        self.fenetre = pygame.display.set_mode((800, 600))
        self.BLANC = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.myfont = pygame.font.SysFont("Comic Sans MS", 30)
        self.label = self.myfont.render(" ", 1, self.BLACK)
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Baton.png")
        self.perso_surf = self.perso_rotated_surf = pygame.image.load(img_path).convert_alpha()
        self.perso_rect = self.perso_surf.get_rect(center=(400, 300))
        self.perso_angle = -175
        self.center_x = 400
        self.continuer = True

    def action(self, x, theta):
        self.center_x = int(400 + x * SCALE)
        self.perso_angle = int(theta) % 360
        self.perso_rotated_surf = self._pg.transform.rotate(self.perso_surf, self.perso_angle)
        self.perso_rect = self.perso_rotated_surf.get_rect(center=(self.center_x, 300))

    def processFrame(self):
        for event in self._pg.event.get():
            if event.type == self._QUIT:
                self.continuer = False
            if event.type == self._pg.KEYDOWN:
                if event.key == self._pg.K_q:
                    self.continuer = False
        self.fenetre.fill(self.BLANC)
        self.fenetre.blit(self.perso_rotated_surf, self.perso_rect)
        self.fenetre.blit(self.label, (10, 10))
        self._pg.display.flip()

    def dspText(self, textToDisp):
        self.label = self.myfont.render(textToDisp, 1, self.BLACK)

    def quit(self):
        self._pg.quit()


# ===========================================================================
# Controleur GUI (utilise l'agent entraine)
# ===========================================================================

class controller(object):
    def __init__(self):
        self.view = view()
        self.model = model()
        self.agent = DQNAgent()
        if os.path.exists(WEIGHTS_FILE):
            self.agent.load(WEIGHTS_FILE)
            print("[play] Poids charges. Le reseau controle le pendule.")
        else:
            print("[play] Aucun poids trouve. L'agent agit aleatoirement.")
        self.agent.eps = 0.0

    def mainLoop(self):
        while self.view.continuer:
            state = self.model.getState()
            time.sleep(DELTAT)

            es = self.agent.encode(state)
            a = self.agent.act(es)
            dx = DQNAgent.ACTIONS[a]
            newx = max(-2.5, min(2.5, self.model.x + dx))
            self.model.ApplyMove(newx)

            theta_deg = -self.model.theta * 180.0 / math.pi
            self.view.action(newx, theta_deg)
            th_norm = math.atan2(math.sin(self.model.theta), math.cos(self.model.theta))
            self.view.dspText(
                f"theta={math.degrees(th_norm):.1f} deg  x={self.model.x:.2f} m"
            )
            self.view.processFrame()
        self.view.quit()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "train":
            eps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
            train(eps)
        elif cmd == "train_gui":
            eps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
            train_gui(eps)
        elif cmd == "test":
            test()
        elif cmd == "play":
            controller().mainLoop()
        else:
            print(__doc__)
    else:
        controller().mainLoop()
