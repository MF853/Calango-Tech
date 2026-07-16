"""
treinar_robo_6rodas.py  -  Treinamento PPO do robo de 6 RODAS (Calango_6Roda).

Cópia fiel de treinar_robo.py (versao caterpillar de 100% sucesso), com APENAS
as mudancas necessarias pro robo de 6 rodas. O objetivo e comparacao JUSTA:
reward, observacao (7-D) e action shaping [0.6, 1.0] sao IDENTICOS ao caterpillar.
A diferenca de desempenho/energia deve vir da GEOMETRIA (rodas vs esteira),
nao de parametros diferentes.

Diferencas frente ao caterpillar:
  1. Handles: /Cuboid + JuntaEsquerda1-3 / JuntaDireita1-3 (6 motores, nao 8).
  2. Sentido dos motores: POSITIVO = pra frente (caterpillar era negativo).
  3. Auto-calibracao da direcao "frente": este robo foi montado do zero, entao
     o eixo X local do chassi pode NAO apontar pra frente fisica. Em vez de
     adivinhar (e arriscar inverter o sinal do reward de progresso, o bug que
     atormentou o caterpillar), damos um leve empurrao no reset e MEDIMOS a
     direcao real do deslocamento.
  4. Nomes de saida separados (cerebro_6rodas, checkpoints_6rodas, tb PPO_6rodas)
     -> NAO toca no modelo caterpillar de 100%.

IMPORTANTE: salve a cena com o robo de 6 rodas posicionado no PE da rampa 1,
de frente pra subida (mesmo ponto de partida do caterpillar). O reset volta o
robo pra posicao salva na cena -- e dela que saem os milestones de altura.
"""

import os
import time
from collections import deque

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# CONSTANTES DE CONFIGURACAO
# ==========================================
ALTURA_TOPO        = 4.45    # m, ganho de altura que define "chegou ao topo".
                             # Mesmo terreno (Terrain[0..2]) do caterpillar -> mesmo valor.
TILT_CAPOTAR       = 75.0    # graus, inclinacao do eixo-Z do robo vs eixo-Z do mundo.
TILT_TOPO          = 35.0    # graus, inclinacao maxima no topo para considerar estavel.
PASSOS_NO_TOPO_OK  = 50      # passos consecutivos acima de ALTURA_TOPO -> sucesso fallback.
TILT_PENALIDADE    = 55.0    # graus, acima disso comeca a penalidade progressiva.
GRACE_PERIOD       = 30      # passos iniciais onde capot e detectado mas NAO termina.
LIMITE_LATERAL_M   = 3.0     # m, drift lateral total da posicao inicial.
LIMITE_QUEDA_M     = 1.0     # m, queda abaixo da altura inicial que conta como capot.
LIMITE_ESTATICO_PASSOS = 200 # passos antes de considerar "travou estatico" (vel < 0.03).
PASSOS_ESTAGNACAO  = 600     # passos sem nova altura maxima -> trunca episodio.
MILESTONES_ALTURA  = [        # bonus de altitude alcancada -> incentiva ultrapassar
    (0.50, 15.0),             # meio rampa 1
    (1.30, 30.0),             # TOPO rampa 1
    (1.80, 15.0),             # progresso entrando na rampa 2
    (2.30, 25.0),             # meio rampa 2
    (2.95, 60.0),             # TOPO rampa 2
    (3.50, 30.0),             # progresso entrando na rampa 3
    (3.85, 40.0),             # meio rampa 3
    (4.20, 100.0),            # TOPO rampa 3
]
PROGRESSO_MIN_M    = 0.002   # ganho minimo (m) acima do maximo do episodio.
PAUSA_TOPO_PASSOS  = 40      # passos de simulacao com motores parados apos topo.
PITCH_CAPOTAR      = TILT_CAPOTAR  # alias retrocompatibilidade (importado por testar_robo)
PITCH_TOPO         = TILT_TOPO     # idem
ROLL_CAPOTAR       = TILT_CAPOTAR  # idem
# === ACTION SPACE RESTRICTION (identico ao caterpillar) ===
ACTION_FLOOR       = 0.6     # throttle minimo
ACTION_RANGE       = 0.4     # max - min, totaliza throttle ate 1.0
MAX_PASSOS         = 2400    # passos por episodio antes de truncar.
VELOCIDADE_MAXIMA  = 7.0     # rad/s (MESMO do caterpillar -> comparacao justa)
TORQUE_MAXIMO      = 1000.0  # N*m, forca maxima do motor (garante subida)
SENTIDO_FRENTE     = +1.0    # POSITIVO = frente neste robo (confirmado no GO/NO-GO)

# Treinamento
TOTAL_TIMESTEPS    = 200_000     # teto; com early-stop normalmente para muito antes
META_RECOMPENSA    = 45.0        # ep_rew_mean alvo (KPI: +35 a +50)
META_TAXA_SUCESSO  = 0.85        # exige tambem >=85% de sucesso para parar (anti-hacking)
JANELA_MEDIA       = 20          # n. de episodios para considerar a media
CHECKPOINT_FREQ    = 2_048       # passos entre checkpoints (= 1 iteracao PPO).

# Diretorios (SEPARADOS do caterpillar -- nao sobrescreve o modelo de 100%)
DIR_CHECKPOINT  = "./checkpoints_6rodas/"
DIR_TENSORBOARD = "./tensorboard_log/"
NOME_MODELO     = "cerebro_6rodas"
TB_LOG_NAME     = "PPO_6rodas"


class CrawlerEnv(gym.Env):
    """
    Ambiente Gymnasium para treinar o robo de 6 RODAS a subir as 3 rampas.

    Acao (1-D): [0,1] -> action shaping -> throttle [0.6, 1.0] -> VELOCIDADE_MAXIMA.
    Observacao (7-D): identica ao caterpillar (climb, tilt, altura, ganho, vel,
                      acao_anterior, vel_forward).
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.client = RemoteAPIClient()
        self.sim    = self.client.getObject("sim")
        self.client.setStepping(True)

        # Silencia qualquer script Lua que cheque 'python_no_op' (ex.: o
        # caterpillar tambem presente na cena). Mantem os outros robos quietos
        # enquanto treinamos o de 6 rodas. O Cuboid nao tem script Lua proprio.
        self.sim.setInt32Signal('python_no_op', 1)

        # Alcas dos objetos na cena -- ROBO DE 6 RODAS
        self.robot        = self.sim.getObject("/Cuboid")
        self.left_motors  = [self.sim.getObject(f"/Cuboid/JuntaEsquerda{i}") for i in range(1, 4)]
        self.right_motors = [self.sim.getObject(f"/Cuboid/JuntaDireita{i}")  for i in range(1, 4)]
        self.all_motors   = self.left_motors + self.right_motors

        # Espacos (identicos ao caterpillar)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low  = np.array([-1.0, 0.0, -0.5, -1.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0, 2.0,  1.5,  1.0, 1.0, 1.0,  1.0], dtype=np.float32),
        )

        self.velocidade_maxima = VELOCIDADE_MAXIMA
        self.altura_base       = 0.0
        self.altura_anterior   = 0.0
        self.altura_max_ep     = 0.0
        self.passos_sem_subir  = 0
        self.passos_no_topo    = 0
        self.acao_anterior     = 0.0
        self.passos_atuais     = 0
        self.forward_inicial   = np.array([1.0, 0.0], dtype=np.float32)
        self.pos_inicial_xy    = np.array([0.0, 0.0], dtype=np.float32)
        self.pos_anterior_xy   = np.array([0.0, 0.0], dtype=np.float32)
        self.milestones_pagos  = set()

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.stopSimulation()
        deadline = time.time() + 2.0
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            if time.time() > deadline:
                break
            time.sleep(0.01)

        self.sim.startSimulation()
        deadline = time.time() + 2.0
        while self.sim.getSimulationState() == self.sim.simulation_stopped:
            if time.time() > deadline:
                break
            time.sleep(0.01)

        # Garante torque maximo nos 6 motores (caso a cena nao tenha persistido).
        for m in self.all_motors:
            try:
                self.sim.setJointTargetForce(m, TORQUE_MAXIMO)
            except Exception:
                pass

        # Assentamento: motores parados por ~0.3 s de fisica.
        for m in self.all_motors:
            self.sim.setJointTargetVelocity(m, 0.0)
        for _ in range(6):
            self.client.step()

        # ----- AUTO-CALIBRACAO DA DIRECAO "FRENTE" -----
        # O eixo X local deste chassi (montado do zero) pode nao apontar pra
        # frente fisica. Damos um leve empurrao (vel positiva = frente) e
        # medimos a direcao real do deslocamento no plano XY. Isso elimina o
        # risco de inverter o sinal do reward de progresso.
        pos_a = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
        for m in self.all_motors:
            self.sim.setJointTargetVelocity(m, SENTIDO_FRENTE * 0.4 * self.velocidade_maxima)
        for _ in range(5):
            self.client.step()
        pos_b = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
        for m in self.all_motors:
            self.sim.setJointTargetVelocity(m, 0.0)

        fwd = np.array([pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]], dtype=np.float32)
        norma = float(np.linalg.norm(fwd))
        if norma > 1e-4:
            self.forward_inicial = fwd / norma
        else:
            # fallback: eixo X local projetado no plano XY
            mat_inicial = self.sim.getObjectMatrix(self.robot, self.sim.handle_world)
            fxy = np.array([mat_inicial[0], mat_inicial[4]], dtype=np.float32)
            n = float(np.linalg.norm(fxy))
            self.forward_inicial = fxy / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        # Estado inicial do episodio (pos pos-empurrao)
        pos = pos_b
        self.altura_base      = pos[2]
        self.altura_anterior  = pos[2]
        self.altura_max_ep    = pos[2]
        self.passos_sem_subir = 0
        self.passos_no_topo   = 0
        self.acao_anterior    = 0.0
        self.milestones_pagos = set()
        self.passos_atuais    = 0
        self.pos_inicial_xy   = np.array([pos[0], pos[1]], dtype=np.float32)
        self.pos_anterior_xy  = self.pos_inicial_xy.copy()

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action):
        acao_normalizada = float(np.clip(action[0], 0.0, 1.0))
        # ACTION SHAPING (identico ao caterpillar):
        # acao=0 -> 0.6, acao=0.5 -> 0.8, acao=1.0 -> 1.0
        throttle = ACTION_FLOOR + acao_normalizada * ACTION_RANGE
        velocidade_alvo = throttle * self.velocidade_maxima

        # Aplica velocidade nos 6 motores. SENTIDO_FRENTE = +1 (positivo = frente).
        for m in self.all_motors:
            self.sim.setJointTargetVelocity(m, SENTIDO_FRENTE * velocidade_alvo)

        self.client.step()

        # Inclinacao via matriz de rotacao (yaw-invariante)
        mat = self.sim.getObjectMatrix(self.robot, self.sim.handle_world)
        cos_tilt = float(np.clip(mat[10], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        climb_component = float(np.clip(mat[8], -1.0, 1.0))

        pos          = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
        altura_atual = pos[2]
        ganho_altura = altura_atual - self.altura_anterior

        vel_linear, _ = self.sim.getObjectVelocity(self.robot)
        vel_norma     = float(np.linalg.norm(vel_linear))

        # PROGRESSO NA DIRECAO DA ENCOSTA
        pos_xy             = np.array([pos[0], pos[1]], dtype=np.float32)
        delta_xy           = pos_xy - self.pos_anterior_xy
        delta_forward      = float(np.dot(delta_xy, self.forward_inicial))
        delta_lateral_vec  = delta_xy - delta_forward * self.forward_inicial
        delta_lateral      = float(np.linalg.norm(delta_lateral_vec))

        vel_xy             = np.array([vel_linear[0], vel_linear[1]], dtype=np.float32)
        vel_forward        = float(np.dot(vel_xy, self.forward_inicial))

        # FUNCAO DE RECOMPENSA (identica ao caterpillar)
        progresso_altura = altura_atual - self.altura_base
        terminated = False

        reward = -0.02

        if delta_forward > 0.0 and tilt_deg < 75.0:
            reward += 15.0 * delta_forward
        elif delta_forward < 0.0:
            reward -= 10.0 * abs(delta_forward)

        reward -= 40.0 * delta_lateral

        if ganho_altura > 0.0 and tilt_deg < 75.0:
            reward += 50.0 * ganho_altura
            reward += 0.15

        if tilt_deg > 25.0 and acao_normalizada > 0.6 and vel_norma > 0.2:
            reward += 0.25

        if tilt_deg > TILT_PENALIDADE:
            reward -= 0.05 * (tilt_deg - TILT_PENALIDADE)

        if velocidade_alvo > 0.4 * self.velocidade_maxima and vel_norma < 0.05 and tilt_deg < 25.0:
            reward -= 0.5

        reward -= 0.002 * (velocidade_alvo ** 2)

        reward -= 0.05 * abs(acao_normalizada - self.acao_anterior)

        # Capotamento: deteccao multi-condicao + grace period
        delta_xy_total      = pos_xy - self.pos_inicial_xy
        forward_total       = float(np.dot(delta_xy_total, self.forward_inicial))
        lateral_vec_total   = delta_xy_total - forward_total * self.forward_inicial
        drift_lateral_total = float(np.linalg.norm(lateral_vec_total))

        em_grace        = self.passos_atuais < GRACE_PERIOD
        capotou_tilt    = tilt_deg > TILT_CAPOTAR
        caiu_do_mapa    = altura_atual < self.altura_base - LIMITE_QUEDA_M
        travou_estatico = (vel_norma < 0.03) and (self.passos_atuais > LIMITE_ESTATICO_PASSOS)
        drift_excessivo = drift_lateral_total > LIMITE_LATERAL_M
        capotou = (capotou_tilt or caiu_do_mapa or travou_estatico or drift_excessivo) and not em_grace

        if capotou_tilt and not em_grace:
            reward -= 40.0
            terminated = True
        elif drift_excessivo and not em_grace:
            reward -= 30.0
            terminated = True
        else:
            if caiu_do_mapa and not em_grace:
                reward -= 0.15
            if travou_estatico:
                reward -= 0.05

        # Topo
        if progresso_altura > ALTURA_TOPO:
            self.passos_no_topo += 1
        else:
            self.passos_no_topo = 0

        topo_instantaneo = progresso_altura > ALTURA_TOPO and tilt_deg < TILT_TOPO
        topo_persistente = self.passos_no_topo >= PASSOS_NO_TOPO_OK
        atingiu_topo = topo_instantaneo or topo_persistente
        if atingiu_topo:
            reward += 200.0
            terminated = True
            for m in self.all_motors:
                self.sim.setJointTargetVelocity(m, 0.0)
            for _ in range(PAUSA_TOPO_PASSOS):
                self.client.step()

        # Watchdog de estagnacao + milestones
        if altura_atual > self.altura_max_ep + PROGRESSO_MIN_M:
            self.altura_max_ep    = altura_atual
            self.passos_sem_subir = 0
            for idx, (alt_threshold, bonus) in enumerate(MILESTONES_ALTURA):
                if (progresso_altura > alt_threshold) and (idx not in self.milestones_pagos):
                    reward += bonus
                    self.milestones_pagos.add(idx)
        else:
            self.passos_sem_subir += 1

        estagnou = self.passos_sem_subir >= PASSOS_ESTAGNACAO
        if estagnou:
            reward -= 0.08

        self.passos_atuais += 1
        truncated = self.passos_atuais >= MAX_PASSOS

        self.altura_anterior = altura_atual
        self.acao_anterior   = acao_normalizada
        self.pos_anterior_xy = pos_xy

        obs = self._build_obs(
            climb_component, tilt_deg, progresso_altura, ganho_altura,
            vel_norma, acao_normalizada, vel_forward,
        )
        info = {
            "tilt_deg": tilt_deg,
            "climb_component": climb_component,
            "progresso_altura": progresso_altura,
            "delta_forward": delta_forward,
            "delta_lateral": delta_lateral,
            "vel_forward": vel_forward,
            "capotou": capotou,
            "atingiu_topo": atingiu_topo,
            "capotou_tilt":    bool(capotou_tilt    and not em_grace),
            "caiu_do_mapa":    bool(caiu_do_mapa    and not em_grace),
            "travou_estatico": bool(travou_estatico and not em_grace),
            "drift_excessivo": bool(drift_excessivo and not em_grace),
            "estagnou":        bool(estagnou),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # OBSERVACAO
    # ------------------------------------------------------------------
    def _get_obs(self):
        mat = self.sim.getObjectMatrix(self.robot, self.sim.handle_world)
        cos_tilt = float(np.clip(mat[10], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        climb_component = float(np.clip(mat[8], -1.0, 1.0))

        pos          = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
        altura_atual = pos[2]
        progresso_altura = altura_atual - self.altura_base
        ganho_altura     = altura_atual - self.altura_anterior

        vel_linear, _ = self.sim.getObjectVelocity(self.robot)
        vel_norma     = float(np.linalg.norm(vel_linear))

        vel_xy        = np.array([vel_linear[0], vel_linear[1]], dtype=np.float32)
        vel_forward   = float(np.dot(vel_xy, self.forward_inicial))

        return self._build_obs(
            climb_component, tilt_deg, progresso_altura, ganho_altura,
            vel_norma, self.acao_anterior, vel_forward,
        )

    def _build_obs(self, climb_component, tilt_deg, progresso_altura, ganho_altura,
                   vel_norma, acao, vel_forward):
        climb_norm   = float(np.clip(climb_component, -1.0, 1.0))
        tilt_norm    = float(np.clip(tilt_deg / 90.0, 0.0, 2.0))
        altura_norm  = float(np.clip(progresso_altura / ALTURA_TOPO, -0.5, 1.5))
        ganho_norm   = float(np.clip(ganho_altura / 0.01, -1.0, 1.0))
        vel_norm     = float(np.clip(vel_norma / 2.0, 0.0, 1.0))
        acao_norm    = float(np.clip(acao, 0.0, 1.0))
        vel_fwd_norm = float(np.clip(vel_forward / 2.0, -1.0, 1.0))

        return np.array(
            [climb_norm, tilt_norm, altura_norm, ganho_norm, vel_norm, acao_norm, vel_fwd_norm],
            dtype=np.float32,
        )

    def close(self):
        try:
            self.sim.stopSimulation()
        except Exception:
            pass


# ==========================================
# CALLBACK - PARADA AUTOMATICA POR DESEMPENHO (identico ao caterpillar)
# ==========================================
class PararQuandoBomCallback(BaseCallback):
    def __init__(self, meta_recompensa: float, meta_sucesso: float = 0.70,
                 janela: int = 30, verbose: int = 1):
        super().__init__(verbose)
        self.meta_recompensa = meta_recompensa
        self.meta_sucesso = meta_sucesso
        self.janela = janela
        self.recompensas    = deque(maxlen=janela)
        self.flags_sucesso  = deque(maxlen=janela)
        self.flags_tilt     = deque(maxlen=janela)
        self.flags_queda    = deque(maxlen=janela)
        self.flags_estatico = deque(maxlen=janela)
        self.flags_drift    = deque(maxlen=janela)
        self.flags_estag    = deque(maxlen=janela)
        self.episodios = 0
        self.ultima_impressao = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.recompensas.append(info["episode"]["r"])
                self.flags_sucesso.append(1 if info.get("atingiu_topo") else 0)
                self.flags_tilt.append(    1 if info.get("capotou_tilt")    else 0)
                self.flags_queda.append(   1 if info.get("caiu_do_mapa")    else 0)
                self.flags_estatico.append(1 if info.get("travou_estatico") else 0)
                self.flags_drift.append(   1 if info.get("drift_excessivo") else 0)
                self.flags_estag.append(   1 if info.get("estagnou")        else 0)
                self.episodios += 1

        if self.episodios - self.ultima_impressao >= 10 and len(self.recompensas) > 0:
            self.ultima_impressao = self.episodios
            media         = float(np.mean(self.recompensas))
            taxa_tilt     = float(np.mean(self.flags_tilt))
            taxa_queda    = float(np.mean(self.flags_queda))
            taxa_estatico = float(np.mean(self.flags_estatico))
            taxa_drift    = float(np.mean(self.flags_drift))
            taxa_estag    = float(np.mean(self.flags_estag))
            taxa_suc      = float(np.mean(self.flags_sucesso))

            print(
                f"  [ep {self.episodios:4d}] rew={media:+7.2f}  suc={taxa_suc*100:4.1f}%  | "
                f"tilt={taxa_tilt*100:4.1f}% queda={taxa_queda*100:4.1f}% "
                f"estatico={taxa_estatico*100:4.1f}% drift={taxa_drift*100:4.1f}% "
                f"estag={taxa_estag*100:4.1f}%"
            )

            self.logger.record("metricas/taxa_capot_tilt",   taxa_tilt)
            self.logger.record("metricas/taxa_queda",        taxa_queda)
            self.logger.record("metricas/taxa_estatico",     taxa_estatico)
            self.logger.record("metricas/taxa_drift",        taxa_drift)
            self.logger.record("metricas/taxa_estagnacao",   taxa_estag)
            self.logger.record("metricas/taxa_sucesso",      taxa_suc)
            self.logger.record("metricas/rew_mean_janela",   media)

            if (len(self.recompensas) >= self.janela
                    and media    >= self.meta_recompensa
                    and taxa_suc >= self.meta_sucesso):
                print(
                    f"\n[OK] Meta atingida! "
                    f"rew_mean={media:+.2f}, sucesso={taxa_suc*100:.0f}%."
                )
                print("    Encerrando treinamento antecipadamente.")
                return False

        return True


# ==========================================
# PONTO DE ENTRADA - TREINAMENTO
# ==========================================
if __name__ == "__main__":
    os.makedirs(DIR_CHECKPOINT, exist_ok=True)
    os.makedirs(DIR_TENSORBOARD, exist_ok=True)

    print("Iniciando o Ambiente de Simulacao (ROBO 6 RODAS)...")
    raw_env = CrawlerEnv()
    env     = Monitor(raw_env)

    print("Criando a Rede Neural (Algoritmo PPO)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.015,
        tensorboard_log=DIR_TENSORBOARD,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq  = CHECKPOINT_FREQ,
        save_path  = DIR_CHECKPOINT,
        name_prefix = NOME_MODELO,
    )
    parada_cb = PararQuandoBomCallback(
        meta_recompensa=META_RECOMPENSA,
        meta_sucesso=META_TAXA_SUCESSO,
        janela=JANELA_MEDIA,
    )
    callbacks = CallbackList([checkpoint_cb, parada_cb])

    print(f"Iniciando o Treinamento ({TOTAL_TIMESTEPS:,} passos, early-stop em rew_mean>={META_RECOMPENSA})")
    print(f"  Logs TensorBoard: tensorboard --logdir {DIR_TENSORBOARD}  (run: {TB_LOG_NAME})")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, tb_log_name=TB_LOG_NAME)
    except KeyboardInterrupt:
        print("\n[!] Treino interrompido pelo usuario. Salvando estado atual...")

    print("Salvando modelo final...")
    model.save(NOME_MODELO)
    print(f"Modelo salvo em: {NOME_MODELO}.zip")

    raw_env.sim.stopSimulation()
    env.close()
