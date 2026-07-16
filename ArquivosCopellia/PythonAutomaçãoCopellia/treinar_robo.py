"""
treinar_robo.py  -  Treinamento PPO do Crawler (versao especializada na encosta unica)

Mudancas principais frente a versao anterior:
- Observacao 7-D: adicionada velocidade na direcao forward inicial (signed).
  A rede agora "sente" se esta avancando, recuando ou indo de lado.
- Recompensa baseada em DIRECAO inicial do robo (especializa pra esta cena):
  o vetor "forward" no inicio do episodio define a direcao da encosta;
  progresso nessa direcao = reward forte, lateral = penalidade, re = penalidade.
- Bonus de topo aumentado para 200 e mantida penalidade alta de capotamento.
- Early-stop mais exigente: rew_mean >= 45 + 85% sucesso (KPI da defesa).
- Total de passos reduzido a 200k (cabe em algumas horas, suficiente p/ converger
  nesta cena especifica).
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
ALTURA_TOPO        = 4.45    # m, ganho de altura que define "chegou ao topo"
                             # (explorar_terreno apos suavizar o terreno: ganho maximo
                             # = 1.672 m, ~70% = 1.17 m. Setado em 1.15 pra garantir
                             # que a regiao de sucesso bate com o plato real do topo
                             # onde tilt cai pra ~18 graus.)
TILT_CAPOTAR       = 75.0    # graus, inclinacao do eixo-Z do robo vs eixo-Z do mundo.
                             # Apos suavizar o joelho do terreno (tilt max estavel no
                             # explorer caiu de 56 -> 49 graus), nao precisamos de 80
                             # de margem pra transientes. 75 graus = 26 graus acima
                             # do max estavel, ainda lenient o suficiente pra picos
                             # transitorios mas pega capots mais cedo. Capots laterais
                             # parados continuam sendo pegos por travou_estatico.
TILT_TOPO          = 35.0    # graus, inclinacao maxima no topo para considerar estavel
                             # (era 25; explorer mostrou tilt=18 no plato, 35 da
                             # margem pra oscilacoes naturais quando o robo chega)
PASSOS_NO_TOPO_OK  = 50      # passos consecutivos acima de ALTURA_TOPO -> sucesso
                             # fallback (independente de tilt) pra capturar casos
                             # onde o robo chegou mas o tilt oscila momentaneamente
TILT_PENALIDADE    = 55.0    # graus, acima disso comeca a penalidade progressiva
GRACE_PERIOD       = 30      # passos iniciais onde capot e detectado mas NAO termina.
                             # Aumentado de 10 -> 30 (1.5s) porque o novo terreno
                             # multi-rampa tem o robo nascendo ja inclinado (tilt 14
                             # inicial), e a fisica precisa de mais tempo pra estabilizar
                             # sem fechar episodio prematuramente.
LIMITE_LATERAL_M   = 3.0     # m, drift lateral total da posicao inicial.
                             # Reduzido de 4.0 -> 3.0 pra terminar mais cedo quando
                             # o robo perdeu o caminho. Com acao 1-D, ele NAO consegue
                             # corrigir drift (nao tem steering diferencial), entao
                             # nao adianta deixar rolar -- melhor cortar e tentar
                             # outro episodio. Drift natural de 1.2m + 1.8m de margem.
LIMITE_QUEDA_M     = 1.0     # m, queda abaixo da altura inicial que conta como capot.
                             # Aumentado de 0.3 -> 1.0: no terreno multi-rampa o robo
                             # comeca numa inclinacao, e qualquer freada faz ele escorregar
                             # 30-50 cm pra tras (capot falso). Com 1m fica claro que e
                             # queda real, nao escorregada normal.
LIMITE_ESTATICO_PASSOS = 200 # passos antes de considerar "travou estatico" (vel < 0.03).
                             # Aumentado de 100 -> 200 (10s) porque o usuario reportou
                             # que esse watchdog dominava como "capot" (100%) sem
                             # capot fisico real. Em multi-rampa, o robo pode pausar
                             # brevemente entre rampas pra "decidir" sem ter travado.
PASSOS_ESTAGNACAO  = 600     # passos sem nova altura maxima -> trunca episodio.
                             # Aumentado de 400 -> 600 (30s) pra cobrir o caso pior:
                             # PPO sub-otimo levando 15-25s pra encontrar o caminho
                             # da proxima rampa (plateau natural so dura 3s na exploracao,
                             # mas com policy ruim pode wandering um bocado).
MILESTONES_ALTURA  = [        # bonus de altitude alcancada -> incentiva ultrapassar
    (0.50, 15.0),             # meio rampa 1
    (1.30, 30.0),             # TOPO rampa 1
    (1.80, 15.0),             # progresso entrando na rampa 2
    (2.30, 25.0),             # meio rampa 2
    (2.95, 60.0),             # TOPO rampa 2
    (3.50, 30.0),             # progresso entrando na rampa 3
    (3.85, 40.0),             # meio rampa 3
    (4.20, 100.0),            # TOPO rampa 3
]                             # Milestones intermediarios dao sinal continuo durante
                              # a subida, em vez de PPO so receber +60 ao chegar no
                              # pico longinquo. Total possivel: +315 + topo final.
PROGRESSO_MIN_M    = 0.002   # ganho minimo (m) acima do maximo do episodio pra
                             # resetar o contador de estagnacao. Reduzido de 5mm -> 2mm
                             # pra que progresso lento (2-3mm/passo) ainda conte.
PAUSA_TOPO_PASSOS  = 40      # passos de simulacao com motores parados apos atingir
                             # o topo, antes do reset. 40 passos = 2s simulados.
                             # Pausa visual pra ver o sucesso antes do reset abrupto.
PITCH_CAPOTAR      = TILT_CAPOTAR  # alias retrocompatibilidade (importado por testar_robo)
PITCH_TOPO         = TILT_TOPO     # idem
ROLL_CAPOTAR       = TILT_CAPOTAR  # idem
# === ACTION SPACE RESTRICTION (curriculum / policy initialization) ===
# Restringe action_normalizada [0,1] -> throttle real [FLOOR, FLOOR+RANGE].
# Justificativa empirica: experimentos C e J mostraram que throttles < 0.5
# levam a drift > 50% e falha sistematica. Restringimos a IA a operar em
# torno do otimo manual conhecido (0.8) com margem pra explorar [0.6, 1.0].
# Isso e action shaping informado por dominio (legitimo em RL aplicado).
# Mapeamento: acao=0 -> throttle 0.6, acao=0.5 -> 0.8, acao=1.0 -> 1.0.
ACTION_FLOOR       = 0.6     # throttle minimo (=conservador, ainda viavel)
ACTION_RANGE       = 0.4     # max - min, totaliza throttle ate 1.0
MAX_PASSOS         = 2400    # passos por episodio antes de truncar (120s = 2min sim).
                             # Em tempo REAL com fps ~5-6 isso vira ~8-10min/episodio.
                             # E o PRINCIPAL watchdog: apenas capot fisico real
                             # (tilt > 75) encerra cedo. Estatico/drift/queda viraram
                             # penalidades sem terminar episodio -- PPO precisa ver
                             # as consequencias completas das acoes pra aprender.
                             # 120s sim e ~3.5x o tempo da exploracao a v_max (34.5s),
                             # margem confortavel pra PPO sub-otimo atravessar tudo.
VELOCIDADE_MAXIMA  = 7.0     # rad/s, velocidade angular maxima dos motores

# Treinamento
TOTAL_TIMESTEPS    = 200_000     # teto; com early-stop normalmente para muito antes
META_RECOMPENSA    = 45.0        # ep_rew_mean alvo (KPI: +35 a +50)
META_TAXA_SUCESSO  = 0.85        # exige tambem >=85% de sucesso para parar (anti-hacking)
JANELA_MEDIA       = 20          # n. de episodios para considerar a media
CHECKPOINT_FREQ    = 2_048       # passos entre checkpoints (= 1 iteracao PPO).
                                 # Salvar a cada update permite recuperar o melhor
                                 # modelo caso o treino colapse depois (catastrophic
                                 # forgetting). Sem isso, perdemos picos de qualidade.

# Diretorios
DIR_CHECKPOINT = "./checkpoints/"
DIR_TENSORBOARD = "./tensorboard_log/"
NOME_MODELO = "cerebro_crawler_otimizado"


class CrawlerEnv(gym.Env):
    """
    Ambiente Gymnasium para treinar o robo lagarta a subir rampa ingreme no CoppeliaSim.

    Acao (1-D): velocidade-alvo dos motores [0.0 .. 1.0] -> escala VELOCIDADE_MAXIMA
    Observacao (7-D):
        0 - climb_component (eixo-X local . eixo-Z mundo) em [-1, 1]
            +1 = nariz pra cima, -1 = nariz pra baixo, 0 = horizontal
        1 - tilt_norm = inclinacao_da_vertical / 90 em [0, ~1]
            0 = robo de pe, 1 = robo de lado, >1 = pendurado
        2 - altura_relativa / ALTURA_TOPO em [-0.5, 1.5]
        3 - ganho_altura / 0.01 em [-1, 1]
        4 - |velocidade| / 2 m/s em [0, 1]
        5 - acao_anterior em [0, 1]
        6 - vel_forward / 2 m/s em [-1, 1]  (velocidade na direcao inicial do robo;
            negativo = recuando, ~0 = parado ou indo de lado, positivo = avancando)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Conexao com o CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim    = self.client.getObject("sim")
        self.client.setStepping(True)

        # LIMPA o signal 'python_no_op' (caso tenha ficado setado por um run
        # anterior de explorar_terreno.py na mesma sessao do CoppeliaSim).
        # Signals em CoppeliaSim persistem entre simulacoes da mesma sessao,
        # entao sem essa limpeza a Lua continua silenciada e o robo nao anda.
        #
        # Justificativa de NAO setar = 1: o modelo atual foi treinado num
        # ambiente onde o script Lua dominava os motores. Setar = 1 agora
        # silenciaria a Lua e o modelo nao saberia controlar.
        self.sim.setInt32Signal('python_no_op', 1)

        # Alcas dos objetos na cena
        self.robot        = self.sim.getObject("/caterpillar")
        self.left_motors  = [self.sim.getObject(f"/caterpillar/dynamicLeftJoint{i}")  for i in range(1, 5)]
        self.right_motors = [self.sim.getObject(f"/caterpillar/dynamicRightJoint{i}") for i in range(1, 5)]

        # Espacos
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low  = np.array([-1.0, 0.0, -0.5, -1.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0, 2.0,  1.5,  1.0, 1.0, 1.0,  1.0], dtype=np.float32),
        )

        self.velocidade_maxima = VELOCIDADE_MAXIMA
        self.altura_base       = 0.0
        self.altura_anterior   = 0.0
        self.altura_max_ep     = 0.0   # maior altura ATINGIDA no episodio
        self.passos_sem_subir  = 0     # contador de estagnacao vertical
        self.passos_no_topo    = 0     # contador de passos consecutivos acima do topo
        self.acao_anterior     = 0.0
        self.passos_atuais     = 0
        # Direcao "para frente" capturada no inicio do episodio (apos assentamento).
        # Vetor unitario no plano XY do mundo, define qual eixo conta como progresso.
        self.forward_inicial   = np.array([1.0, 0.0], dtype=np.float32)
        self.pos_inicial_xy    = np.array([0.0, 0.0], dtype=np.float32)
        self.pos_anterior_xy   = np.array([0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Para a simulacao e espera o estado confirmar (polling > sleep fixo).
        # Corta ~300 ms por reset em relacao ao time.sleep(0.4) cego.
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

        # Periodo de assentamento: motores parados por ~0.3 s de fisica.
        # Diagnostico mostrou que o robo nasce com roll ~-45 e leva ~0.3 s
        # caindo ate ficar de pe. Sem isso, a IA "vive" 6 passos por episodio
        # tentando agir num estado tombado que nao reflete o problema real.
        for lm, rm in zip(self.left_motors, self.right_motors):
            self.sim.setJointTargetVelocity(lm, 0.0)
            self.sim.setJointTargetVelocity(rm, 0.0)
        for _ in range(6):    # 6 ticks * 50 ms = 300 ms de fisica
            self.client.step()

        pos = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
        self.altura_base      = pos[2]
        self.altura_anterior  = pos[2]
        self.altura_max_ep    = pos[2]
        self.passos_sem_subir = 0
        self.passos_no_topo   = 0
        self.acao_anterior    = 0.0
        # Tracker dos milestones de altura ja alcancados no episodio (set de
        # indices da MILESTONES_ALTURA). Cada milestone paga uma vez por episodio.
        self.milestones_pagos = set()
        self.passos_atuais    = 0

        # Captura o eixo X local do robo (=frente do chassi) e projeta no plano
        # XY do mundo. Esse vetor define a direcao da encosta nessa cena
        # especifica. Recompensar progresso nessa direcao da um sinal muito mais
        # forte que medir apenas ganho de altitude.
        mat_inicial = self.sim.getObjectMatrix(self.robot, self.sim.handle_world)
        forward_xy = np.array([mat_inicial[0], mat_inicial[4]], dtype=np.float32)
        norma = float(np.linalg.norm(forward_xy))
        if norma > 1e-6:
            self.forward_inicial = forward_xy / norma
        else:
            self.forward_inicial = np.array([1.0, 0.0], dtype=np.float32)

        self.pos_inicial_xy  = np.array([pos[0], pos[1]], dtype=np.float32)
        self.pos_anterior_xy = self.pos_inicial_xy.copy()

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action):
        acao_normalizada = float(np.clip(action[0], 0.0, 1.0))
        # ACTION SHAPING: acao [0,1] -> throttle [ACTION_FLOOR, FLOOR+RANGE].
        # Restringe espaco util por evidencia empirica (ver docstring no topo).
        # acao=0   -> throttle 0.6 (minimo seguro)
        # acao=0.5 -> throttle 0.8 (baseline manual otimo)
        # acao=1.0 -> throttle 1.0 (maximo)
        throttle = ACTION_FLOOR + acao_normalizada * ACTION_RANGE
        velocidade_alvo  = throttle * self.velocidade_maxima

        # Aplica velocidade em todos os motores
        for lm, rm in zip(self.left_motors, self.right_motors):
            self.sim.setJointTargetVelocity(lm, -velocidade_alvo)
            self.sim.setJointTargetVelocity(rm, -velocidade_alvo)

        self.client.step()

        # Leituras de sensores - inclinacao via MATRIZ DE ROTACAO (yaw-invariante)
        # Diagnostico mostrou que ler pitch/roll de Euler com yaw=-88 leva a
        # interpretacao errada: o robo subia ramps inclinadas lateralmente e o
        # Euler reportava roll=-56 (penalizado como capotamento) mesmo o robo
        # estando firme nas esteiras. mat[10] = cos(angulo entre Z_robo e Z_mundo)
        # da o tilt REAL independente de yaw.
        mat = self.sim.getObjectMatrix(self.robot, self.sim.handle_world)
        cos_tilt = float(np.clip(mat[10], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        climb_component = float(np.clip(mat[8], -1.0, 1.0))  # +1 nariz acima, -1 nariz abaixo

        pos          = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
        altura_atual = pos[2]
        ganho_altura = altura_atual - self.altura_anterior

        vel_linear, _ = self.sim.getObjectVelocity(self.robot)
        vel_norma     = float(np.linalg.norm(vel_linear))

        # ==========================================
        # PROGRESSO NA DIRECAO DA ENCOSTA
        # ==========================================
        # Decompoe o deslocamento horizontal do passo em (forward, lateral)
        # usando a direcao XY inicial do robo. Forward+ = avancando para a encosta;
        # forward- = recuando; lateral = drift perpendicular.
        pos_xy             = np.array([pos[0], pos[1]], dtype=np.float32)
        delta_xy           = pos_xy - self.pos_anterior_xy
        delta_forward      = float(np.dot(delta_xy, self.forward_inicial))
        delta_lateral_vec  = delta_xy - delta_forward * self.forward_inicial
        delta_lateral      = float(np.linalg.norm(delta_lateral_vec))

        vel_xy             = np.array([vel_linear[0], vel_linear[1]], dtype=np.float32)
        vel_forward        = float(np.dot(vel_xy, self.forward_inicial))

        # ==========================================
        # FUNCAO DE RECOMPENSA - especializada nesta encosta
        # ==========================================
        progresso_altura = altura_atual - self.altura_base
        terminated = False

        # 1. Penalidade de existencia (pune ficar parado)
        reward = -0.02

        # 2. Progresso na direcao da encosta - sinal PRIMARIO
        #    Recompensa avanco; pune recuo. Limite de tilt elevado pra 75 pra
        #    manter sinal positivo durante a parte ingreme (rampa do cenario
        #    chega a ~56 graus a v_max, mas picos transitorios sobem mais).
        if delta_forward > 0.0 and tilt_deg < 75.0:
            reward += 15.0 * delta_forward
        elif delta_forward < 0.0:
            reward -= 10.0 * abs(delta_forward)

        # 3. Penalidade por drift lateral - prioriza trajeto reto.
        #    Aumentado de 20 -> 40 porque o robo estava drifting muito pra direita
        #    no terreno multi-rampa. Com acao 1-D ele nao tem como ATIVAMENTE
        #    corrigir, mas a penalidade alta faz o PPO preferir velocidades que
        #    minimizam o drift (ex: velocidade constante em vez de oscilacoes
        #    que amplificam derrapagens laterais).
        reward -= 40.0 * delta_lateral

        # 4. Bonus de ganho de altura - sinal complementar. Multiplicador
        #    aumentado de 30 -> 50 e steady bonus 0.05 -> 0.15 pra dar gradiente
        #    mais forte durante a subida (o problema atual e o robo parar no pe da
        #    rampa 2 porque o reward de subir nao compensa o risco de capot).
        if ganho_altura > 0.0 and tilt_deg < 75.0:
            reward += 50.0 * ganho_altura
            reward += 0.15                      # bonus por subir estavel

        # 4b. Bonus de "SUBIR COM POTENCIA". Pune politica timida que reduz
        #     throttle no meio da rampa. Combina 3 condicoes: rampa real
        #     (tilt > 25), aceleracao alta (acao > 0.6) e robo de fato se
        #     movendo (vel > 0.2). +0.25/passo durante a subida = +25 ao
        #     longo de uma rampa de 100 passos.
        if tilt_deg > 25.0 and acao_normalizada > 0.6 and vel_norma > 0.2:
            reward += 0.25

        # 5. Penalidade SO em inclinacoes extremas
        if tilt_deg > TILT_PENALIDADE:
            reward -= 0.05 * (tilt_deg - TILT_PENALIDADE)

        # 6. Penalidade por patinar (motor alto, robo parado) - SO em terreno plano.
        #    Em rampa ingreme o robo pode ter motor alto + vel baixa LEGITIMAMENTE
        #    (subindo devagar contra gravidade). Detector original capturava isso
        #    como patinacao falsa e punia, ensinando a IA a reduzir throttle em
        #    rampa - exatamente o oposto do que queremos.
        if velocidade_alvo > 0.4 * self.velocidade_maxima and vel_norma < 0.05 and tilt_deg < 25.0:
            reward -= 0.5

        # 7. Penalidade de energia (proxy do KPI de consumo). Reduzida de
        #    0.005 -> 0.002: o valor antigo subtraia -0.245/passo em throttle
        #    cheio (v=7), praticamente anulando o ganho de subir rapido.
        #    Com -0.098/passo a IA mantem incentivo claro pra usar potencia.
        reward -= 0.002 * (velocidade_alvo ** 2)

        # 8. Penalidade por mudanca brusca de acao
        reward -= 0.05 * abs(acao_normalizada - self.acao_anterior)

        # 9. Capotamento: deteccao MULTI-CONDICAO + grace period.
        #    A versao anterior so olhava tilt > 85 e perdia situacoes onde
        #    o robo claramente estava caido mas o threshold nao disparava:
        #    - capot lateral (tilt 70-85 deitado, antes nao era capot)
        #    - caiu fora do mapa (z bem abaixo da base, antes nao era capot)
        #    - travou imovel em estado quebrado (vel ~ 0 por muitos passos)
        #    Durante GRACE_PERIOD passos iniciais, capot e detectado mas NAO
        #    termina o episodio -- da tempo da fisica estabilizar.
        # Drift lateral total: distancia perpendicular a forward_inicial.
        # Diferente de delta_lateral (que e por passo), este e ACUMULADO.
        delta_xy_total      = pos_xy - self.pos_inicial_xy
        forward_total       = float(np.dot(delta_xy_total, self.forward_inicial))
        lateral_vec_total   = delta_xy_total - forward_total * self.forward_inicial
        drift_lateral_total = float(np.linalg.norm(lateral_vec_total))

        em_grace        = self.passos_atuais < GRACE_PERIOD
        capotou_tilt    = tilt_deg > TILT_CAPOTAR
        caiu_do_mapa    = altura_atual < self.altura_base - LIMITE_QUEDA_M
        travou_estatico = (vel_norma < 0.03) and (self.passos_atuais > LIMITE_ESTATICO_PASSOS)
        drift_excessivo = drift_lateral_total > LIMITE_LATERAL_M
        # Flag agregada mantida pro callback de log breakdown.
        capotou = (capotou_tilt or caiu_do_mapa or travou_estatico or drift_excessivo) and not em_grace

        # Capot fisico (tilt > 75) E drift excessivo terminam o episodio.
        # Drift termina porque com acao 1-D o robo NAO consegue corrigir
        # (sem steering diferencial), entao nao adianta deixar o sim rodar
        # 2 minutos com o robo perdido pra direita -- melhor cortar e iniciar
        # nova tentativa que possa convergir.
        # Estatico e queda continuam sendo penalidades sem terminar -- nesses
        # casos o robo ainda PODE se recuperar.
        if capotou_tilt and not em_grace:
            reward -= 40.0
            terminated = True
        elif drift_excessivo and not em_grace:
            reward -= 30.0          # penalidade unica no termino (nao por passo)
            terminated = True
        else:
            if caiu_do_mapa and not em_grace:
                reward -= 0.15          # caiu fora do mapa: desincentivo forte por passo
            if travou_estatico:
                reward -= 0.05          # parado: leve (existencia ja desincentiva)

        # 10. Topo: duas condicoes possiveis (OR).
        #     (a) Altura > ALTURA_TOPO E tilt < TILT_TOPO no mesmo instante.
        #     (b) Altura > ALTURA_TOPO por PASSOS_NO_TOPO_OK passos consecutivos,
        #         independente de tilt. Fallback pra casos onde o robo chegou
        #         no topo mas o tilt oscila acima/abaixo do TILT_TOPO sem nunca
        #         coincidir com a leitura de altura.
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
            # Pausa visual no topo: para os motores e roda fisica por PAUSA_TOPO_PASSOS
            # ticks com o robo livre pra "descansar" no topo. Da tempo de ver o
            # sucesso na tela antes do stopSimulation reiniciar a cena.
            for lm, rm in zip(self.left_motors, self.right_motors):
                self.sim.setJointTargetVelocity(lm, 0.0)
                self.sim.setJointTargetVelocity(rm, 0.0)
            for _ in range(PAUSA_TOPO_PASSOS):
                self.client.step()

        # 11. Watchdog de estagnacao vertical. Se o robo nao alcanca uma altura
        #     nova por PASSOS_ESTAGNACAO passos consecutivos, trunca com penalidade.
        #     Threshold de PROGRESSO_MIN_M (2mm) significa que subidas lentas mas
        #     consistentes (2-3mm/passo) NAO disparam o watchdog -- so estagnacao
        #     real (robo parado ou andando em circulo).
        if altura_atual > self.altura_max_ep + PROGRESSO_MIN_M:
            self.altura_max_ep    = altura_atual
            self.passos_sem_subir = 0

            # Bonus de milestone: paga uma vez por episodio quando o robo
            # cruza o pico de cada rampa. Cria "degraus" de recompensa que
            # superam o ganho parcial do plateau anterior, incentivando
            # tentar a proxima rampa em vez de estagnar.
            for idx, (alt_threshold, bonus) in enumerate(MILESTONES_ALTURA):
                if (progresso_altura > alt_threshold) and (idx not in self.milestones_pagos):
                    reward += bonus
                    self.milestones_pagos.add(idx)
        else:
            self.passos_sem_subir += 1

        # Estagnacao: penalidade continua sem terminar episodio. Idem racional
        # do bloco de capot acima -- PPO precisa observar consequencias, nao
        # ser cortado quando o estado fica ruim.
        estagnou = self.passos_sem_subir >= PASSOS_ESTAGNACAO
        if estagnou:
            reward -= 0.08          # -0.08/passo enquanto estagnado

        # 12. Tempo esgotado -> truncado (nao e falha do agente)
        self.passos_atuais += 1
        truncated = self.passos_atuais >= MAX_PASSOS

        # Atualizacoes de estado para o proximo passo
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
            "capotou": capotou,                     # mantido pra compat
            "atingiu_topo": atingiu_topo,
            # Breakdown das 4 causas distintas de termino-por-falha.
            # Apenas capotou_tilt e tombamento "de verdade" (robo deitado).
            # As outras 3 sao falhas de comportamento, nao capotamentos fisicos.
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
        """Constroi a observacao do estado atual (usado por reset)."""
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
        # Normalizacoes
        climb_norm   = float(np.clip(climb_component, -1.0, 1.0))     # ja em [-1, 1]
        tilt_norm    = float(np.clip(tilt_deg / 90.0, 0.0, 2.0))      # 1.0 = robo de lado
        altura_norm  = float(np.clip(progresso_altura / ALTURA_TOPO, -0.5, 1.5))
        ganho_norm   = float(np.clip(ganho_altura / 0.01, -1.0, 1.0))
        vel_norm     = float(np.clip(vel_norma / 2.0, 0.0, 1.0))
        acao_norm    = float(np.clip(acao, 0.0, 1.0))
        vel_fwd_norm = float(np.clip(vel_forward / 2.0, -1.0, 1.0))   # signed

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
# CALLBACK - PARADA AUTOMATICA POR DESEMPENHO
# ==========================================
class PararQuandoBomCallback(BaseCallback):
    """
    Para o treino assim que a media movel de recompensa por episodio atinge a meta.
    Tambem imprime estatisticas de pitch/roll/capotamento periodicamente.
    """

    def __init__(self, meta_recompensa: float, meta_sucesso: float = 0.70,
                 janela: int = 30, verbose: int = 1):
        super().__init__(verbose)
        self.meta_recompensa = meta_recompensa
        self.meta_sucesso = meta_sucesso
        self.janela = janela
        self.recompensas    = deque(maxlen=janela)
        self.flags_sucesso  = deque(maxlen=janela)
        # Breakdown das causas de termino: cada modo de falha conta separado
        # pra ficar claro o que esta acontecendo (capot fisico vs estagnacao vs queda).
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

            # Capotamento REAL (so tilt fisico) - separado das outras falhas
            self.logger.record("metricas/taxa_capot_tilt",   taxa_tilt)
            self.logger.record("metricas/taxa_queda",        taxa_queda)
            self.logger.record("metricas/taxa_estatico",     taxa_estatico)
            self.logger.record("metricas/taxa_drift",        taxa_drift)
            self.logger.record("metricas/taxa_estagnacao",   taxa_estag)
            self.logger.record("metricas/taxa_sucesso",      taxa_suc)
            self.logger.record("metricas/rew_mean_janela",   media)

            # Parada antecipada SO se as duas condicoes valem juntas
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

    print("Iniciando o Ambiente de Simulacao...")
    raw_env = CrawlerEnv()
    env     = Monitor(raw_env)

    print("Criando a Rede Neural (Algoritmo PPO)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1.5e-4,          # reduzido de 3e-4. Atualizacoes menores =
                                       # menos chance de colapso quando value
                                       # function ainda nao convergiu.
        n_steps=2048,
        batch_size=64,
        n_epochs=5,                    # reduzido de 10. Menos passes pelos mesmos
                                       # dados = menos overfitting a noise.
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.015,                # AUMENTADO de 0.005 -> 0.015 pro terreno
                                       # multi-rampa. O terreno anterior era 1 rampa;
                                       # agora sao 3 com plateaus entre elas, e a IA
                                       # precisa explorar mais pra romper o local
                                       # optimum "subir rampa 1 e parar". Risco de
                                       # catastrophic forgetting compensado pela
                                       # densidade nova dos milestones.
        tensorboard_log=DIR_TENSORBOARD,
    )

    # Callbacks
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

    print(f"Iniciando o Treinamento ({TOTAL_TIMESTEPS:,} passos, com early-stop em rew_mean>={META_RECOMPENSA})")
    print(f"  Logs TensorBoard: tensorboard --logdir {DIR_TENSORBOARD}")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[!] Treino interrompido pelo usuario. Salvando estado atual...")

    print("Salvando modelo final...")
    model.save(NOME_MODELO)
    print(f"Modelo salvo em: {NOME_MODELO}.zip")

    raw_env.sim.stopSimulation()
    env.close()
