"""
testar_robo_6rodas.py  -  Missao autonoma do ROBO DE 6 RODAS em terreno multi-rampa.

Cópia fiel de testar_robo.py, adaptada pro robo de 6 rodas (Calango_6Roda):
  - Importa o ambiente/constantes de treinar_robo_6rodas (handles /Cuboid, 6 motores).
  - Carrega cerebro_6rodas.zip (checkpoints em ./checkpoints_6rodas/).
  - aplicar_velocidade NAO nega: neste robo motor POSITIVO = frente (confirmado
    no GO/NO-GO), ao contrario do caterpillar onde negativo = frente.
  - Maquina de estados da missao e calculo de energia: IDENTICOS (comparacao justa).

Sequencia da missao
-------------------
1. SUBINDO_R1 -> 2. MEDINDO_1 -> 3. SUBINDO_R2 -> 4. MEDINDO_2 ->
5. SUBINDO_R3 -> 6. MEDINDO_3 -> 7. RETORNANDO (contagem regressiva) -> 8. FINALIZADO

Modos de execucao
-----------------
    python testar_robo_6rodas.py                  -> Modo IA (PPO treinado)
    python testar_robo_6rodas.py --constante VAL  -> Modo baseline (acao fixa VAL)

Com action shaping (acao -> throttle [0.6, 1.0]):
    --constante 0.5 -> throttle 0.8 (= baseline manual otimo)
    --constante 0.0 -> throttle 0.6 (= conservador)
    --constante 1.0 -> throttle 1.0 (= maximo)

Comparacao energetica
---------------------
    1. python testar_robo_6rodas.py                -> anote E_ia
    2. python testar_robo_6rodas.py --constante 0.5 -> anote E_const (throttle 0.8)
    3. Economia% = (E_const - E_ia) / E_const * 100
"""

import argparse
import glob
import os
import time
import numpy as np
from stable_baselines3 import PPO
from treinar_robo_6rodas import (
    CrawlerEnv, ALTURA_TOPO, TILT_TOPO, TILT_CAPOTAR,
    NOME_MODELO, VELOCIDADE_MAXIMA, ACTION_FLOOR, ACTION_RANGE,
)


# ==========================================
# PARAMETROS DA MISSAO
# ==========================================
TEMPO_MEDICAO_S      = 15.0    # segundos parado coletando dados (cada platau)
VELOCIDADE_RE        = -3.0    # rad/s comando p/ marcha-re (positivo=frente neste robo,
                               # entao -3.0 -> motor -3.0 -> re fisico)
LIMITE_PASSOS_TOTAIS = 15000   # guarda-chuva geral do loop (~12.5min sim a dt=50ms)
TOLERANCIA_BASE_M    = 0.10    # margem (m) para considerar "voltou a base"
TOTAL_RAMPAS         = 3       # numero de plateaus que o sistema vai detectar

# Deteccao de plateau (transicao rampa -> superficie quase plana)
TILT_RAMPA           = 25.0    # graus, acima disso = robo na rampa
TILT_PLATEAU         = 15.0    # graus, abaixo disso = entrou em platau
GANHO_MIN_RAMPA_M    = 0.3     # ganho de altura minimo pra considerar "rampa cruzada"

# Nomes simbolicos das zonas de umidade
NOMES_ZONAS = ["Zona Seca", "Zona Umida", "Zona de Lama"]
UMIDADE_POR_ZONA = [(20, 35), (50, 65), (75, 90)]  # (min, max) % umidade simulada

# Diretorio de checkpoints do robo de 6 rodas
DIR_CHECKPOINT_6R = "./checkpoints_6rodas/"


# ==========================================
# UTILITARIOS
# ==========================================
def localizar_modelo():
    """Retorna o caminho do modelo a carregar. Tenta o final primeiro;
    cai pro checkpoint mais recente se nao existir."""
    if os.path.exists(f"{NOME_MODELO}.zip"):
        return NOME_MODELO
    checkpoints = sorted(glob.glob(f"{DIR_CHECKPOINT_6R}{NOME_MODELO}_*_steps.zip"))
    if checkpoints:
        caminho = checkpoints[-1][:-4]
        print(f"[!] {NOME_MODELO}.zip nao existe. Usando: {caminho}")
        return caminho
    zips = sorted(glob.glob(f"{NOME_MODELO}_*.zip"))
    if zips:
        caminho = zips[-1][:-4]
        print(f"[!] {NOME_MODELO}.zip nao existe. Usando: {caminho}")
        return caminho
    raise FileNotFoundError(
        "Nenhum modelo encontrado. Rode treinar_robo_6rodas.py primeiro."
    )


def parar_motores(env: CrawlerEnv) -> None:
    for m in (env.left_motors + env.right_motors):
        env.sim.setJointTargetVelocity(m, 0.0)


def aplicar_velocidade(env: CrawlerEnv, velocidade: float) -> None:
    """Aplica velocidade aos motores. CONVENCAO (igual env.step() do 6-rodas):
    - velocidade > 0  -> robo anda PRA FRENTE
    - velocidade < 0  -> robo anda PRA TRAS (marcha-re)
    Neste robo motor POSITIVO = frente, entao NAO negamos (ao contrario do caterpillar).
    """
    for m in (env.left_motors + env.right_motors):
        env.sim.setJointTargetVelocity(m, velocidade)


def ler_tilt_graus(env: CrawlerEnv) -> float:
    """Inclinacao do eixo-Z do robo vs eixo-Z do mundo (yaw-invariante)."""
    mat = env.sim.getObjectMatrix(env.robot, env.sim.handle_world)
    cos_tilt = float(np.clip(mat[10], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))


def energia_do_passo(action_value: float) -> float:
    """Energia consumida em um passo (proxy P = tau*omega).
    Formula do KPI: 0.005 * v_alvo^2. Unidade arbitraria (J).
    Aplica o MESMO action shaping do env: throttle = FLOOR + acao*RANGE."""
    throttle = ACTION_FLOOR + float(action_value) * ACTION_RANGE
    velocidade_alvo = throttle * VELOCIDADE_MAXIMA
    return 0.005 * (velocidade_alvo ** 2)


def simular_umidade(zona_idx: int) -> float:
    if zona_idx < len(UMIDADE_POR_ZONA):
        mn, mx = UMIDADE_POR_ZONA[zona_idx]
    else:
        mn, mx = (40, 60)
    return float(np.random.uniform(mn, mx))


# ==========================================
# PARSER DE ARGUMENTOS CLI
# ==========================================
parser = argparse.ArgumentParser(description="Missao autonoma do robo de 6 rodas")
parser.add_argument(
    "--constante",
    type=float,
    default=None,
    metavar="VAL",
    help="Modo baseline: usa acao fixa VAL (0.0-1.0) em vez da IA. Util pra comparar energia."
)
args = parser.parse_args()

MODO_IA = args.constante is None
ACAO_CONSTANTE = float(args.constante) if not MODO_IA else None
if ACAO_CONSTANTE is not None:
    ACAO_CONSTANTE = float(np.clip(ACAO_CONSTANTE, 0.0, 1.0))


# ==========================================
# INICIALIZACAO
# ==========================================
modo_str = "IA (PPO treinado)" if MODO_IA else f"BASELINE acao={ACAO_CONSTANTE:.2f}"
print(f"Carregando o Cerebro treinado... (modo: {modo_str})")
if MODO_IA:
    model = PPO.load(localizar_modelo())
else:
    model = None  # nao usamos modelo em modo baseline

print("Iniciando a Missao do Rover (6 RODAS)...")
env = CrawlerEnv()
obs, info = env.reset()

posicao_inicial = env.sim.getObjectPosition(env.robot, env.sim.handle_world)
altura_base     = posicao_inicial[2]
pos_inicial_xy  = np.array([posicao_inicial[0], posicao_inicial[1]], dtype=np.float32)

estado               = "SUBINDO"
num_rampas_cruzadas  = 0
tilt_alto_visto      = False
altura_no_inicio_rampa = altura_base
tempo_inicio_medicao = 0.0
medicoes             = []
aviso_drift_dado     = False

rampas_a_descer        = 0
tilt_alto_visto_re     = False
altura_no_inicio_descida = 0.0

energia_subida    = 0.0
energia_retorno   = 0.0
dist_acumulada    = 0.0
pos_anterior_xy   = pos_inicial_xy.copy()
passos_sub_total  = 0
passos_re_total   = 0
tempo_sim_inicio  = env.sim.getSimulationTime()

print("=" * 60)
print("       MISSAO INICIADA (ROBO 6 RODAS)")
print("=" * 60)
print(f"  Modo               : {modo_str}")
print(f"  Altitude da Base   : {altura_base:+.3f} m")
print(f"  Numero de Plateaus : {TOTAL_RAMPAS}")
print(f"  Tempo de Medicao   : {TEMPO_MEDICAO_S:.0f} s por platau")
print("=" * 60)


# ==========================================
# LOOP PRINCIPAL DA MISSAO
# ==========================================
for passo in range(LIMITE_PASSOS_TOTAIS):

    pos_atual    = env.sim.getObjectPosition(env.robot, env.sim.handle_world)
    pos_xy       = np.array([pos_atual[0], pos_atual[1]], dtype=np.float32)
    altura_atual = pos_atual[2]
    tilt_deg     = ler_tilt_graus(env)

    dist_acumulada += float(np.linalg.norm(pos_xy - pos_anterior_xy))
    pos_anterior_xy = pos_xy

    # -----------------------------------------------
    # ESTADO 1 - SUBINDO (IA ou baseline constante)
    # -----------------------------------------------
    if estado == "SUBINDO":
        if MODO_IA:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([ACAO_CONSTANTE], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        passos_sub_total += 1
        energia_subida += energia_do_passo(action[0])

        if tilt_deg > TILT_RAMPA and not tilt_alto_visto:
            tilt_alto_visto = True
            altura_no_inicio_rampa = altura_atual

        if tilt_alto_visto and tilt_deg < TILT_PLATEAU:
            ganho = altura_atual - altura_no_inicio_rampa
            if ganho >= GANHO_MIN_RAMPA_M:
                num_rampas_cruzadas += 1
                zona = num_rampas_cruzadas - 1
                print(f"\n[OK] Rampa {num_rampas_cruzadas}/{TOTAL_RAMPAS} cruzada! "
                      f"(altura={altura_atual:+.3f} m, ganho={ganho:+.2f} m)")
                print(f"     Parando para medicao em {NOMES_ZONAS[zona]}...")
                parar_motores(env)
                estado = "MEDINDO"
                tempo_inicio_medicao = env.sim.getSimulationTime()
            tilt_alto_visto = False

        if passo % 100 == 0:
            acao_str = f"{action[0]:.2f}"
            print(f"  [SUBINDO]  altura={altura_atual:+.3f} m | tilt={tilt_deg:4.1f} "
                  f"| rampa={num_rampas_cruzadas}/{TOTAL_RAMPAS} | acao={acao_str}")

        if info.get("capotou_tilt"):
            print(f"\n[!] Capotou (tilt={tilt_deg:.1f}). Tentando recuperar...")
            obs, info_reset = env.reset()
            tilt_alto_visto = False
            continue

        if terminated and not info.get("capotou_tilt"):
            if info.get("drift_excessivo") and not aviso_drift_dado:
                print(f"\n[i] Aviso: drift acumulado excedeu o limite de treino. "
                      f"Continuando missao (em teste o watchdog e ignorado).")
                aviso_drift_dado = True

    # -----------------------------------------------
    # ESTADO 2 - MEDINDO (parado, coleta de dados)
    # -----------------------------------------------
    elif estado == "MEDINDO":
        parar_motores(env)
        env.client.step()
        tempo_passado = env.sim.getSimulationTime() - tempo_inicio_medicao

        barra = int((tempo_passado / TEMPO_MEDICAO_S) * 20)
        zona  = num_rampas_cruzadas - 1
        print(f"  [MEDINDO {NOMES_ZONAS[zona]}] {'#'*barra}{'-'*(20-barra)} "
              f"{tempo_passado:4.1f}/{TEMPO_MEDICAO_S:.0f}s", end="\r")

        if tempo_passado >= TEMPO_MEDICAO_S:
            umidade = simular_umidade(zona)
            medicoes.append((NOMES_ZONAS[zona], umidade))
            print(f"\n[OK] Coleta {num_rampas_cruzadas}/{TOTAL_RAMPAS} concluida. "
                  f"{NOMES_ZONAS[zona]}: Umidade = {umidade:.1f}%")

            obs = env._get_obs()

            if num_rampas_cruzadas < TOTAL_RAMPAS:
                print(f"     Retomando subida para rampa {num_rampas_cruzadas + 1}...")
                estado = "SUBINDO"
            else:
                print(f"     Todas as medicoes coletadas! Iniciando retorno a base...")
                print(f"     Contagem regressiva: {TOTAL_RAMPAS} rampas a descer.")
                estado = "RETORNANDO"
                rampas_a_descer = TOTAL_RAMPAS
                tilt_alto_visto_re = False

    # -----------------------------------------------
    # ESTADO 3 - RETORNANDO (marcha-re com contagem regressiva de rampas)
    # -----------------------------------------------
    elif estado == "RETORNANDO":
        vel_re = VELOCIDADE_RE if tilt_deg < TILT_CAPOTAR * 0.7 else VELOCIDADE_RE * 0.5
        aplicar_velocidade(env, vel_re)
        env.client.step()
        passos_re_total += 1
        energia_retorno += 0.005 * (vel_re ** 2)

        if tilt_deg > TILT_RAMPA and not tilt_alto_visto_re:
            tilt_alto_visto_re = True
            altura_no_inicio_descida = altura_atual

        if tilt_alto_visto_re and tilt_deg < TILT_PLATEAU:
            descida = altura_no_inicio_descida - altura_atual
            if descida >= GANHO_MIN_RAMPA_M:
                rampas_a_descer -= 1
                print(f"\n[OK] Desceu rampa! Faltam {rampas_a_descer}/{TOTAL_RAMPAS}.")
            tilt_alto_visto_re = False

        if passo % 100 == 0:
            print(f"  [RETORNANDO] altura={altura_atual:+.3f} m | tilt={tilt_deg:4.1f} "
                  f"| rampas restantes={rampas_a_descer}")

        if rampas_a_descer <= 0:
            parar_motores(env)
            env.client.step()
            print(f"\n[OK] Retorno a base concluido! "
                  f"Todas as {TOTAL_RAMPAS} rampas descidas. (altura={altura_atual:+.3f} m)")
            estado = "FINALIZADO"

        elif altura_atual <= altura_base + TOLERANCIA_BASE_M:
            parar_motores(env)
            env.client.step()
            print(f"\n[OK] Retorno a base concluido por altura! "
                  f"({TOTAL_RAMPAS - rampas_a_descer} rampas descidas detectadas, "
                  f"altura={altura_atual:+.3f} m)")
            estado = "FINALIZADO"

        if tilt_deg > TILT_CAPOTAR:
            parar_motores(env)
            print(f"\n[!] Capotou no retorno (tilt={tilt_deg:.1f}). Encerrando.")
            break

    # -----------------------------------------------
    # ESTADO 4 - FIM DA MISSAO
    # -----------------------------------------------
    elif estado == "FINALIZADO":
        parar_motores(env)
        env.client.step()
        break

    time.sleep(0.001)


# ==========================================
# RELATORIO FINAL
# ==========================================
tempo_total_sim = env.sim.getSimulationTime() - tempo_sim_inicio
energia_total   = energia_subida + energia_retorno
ef_por_metro    = energia_total / max(dist_acumulada, 0.001)

print("\n" + "=" * 60)
print("       RELATORIO DA MISSAO (ROBO 6 RODAS)")
print("=" * 60)
print(f"  Modo                              : {modo_str}")
print(f"  Estado final                      : {estado}")
print(f"  Rampas cruzadas                   : {num_rampas_cruzadas}/{TOTAL_RAMPAS}")
print(f"  Tempo total simulado              : {tempo_total_sim:.1f} s")
print(f"  Distancia horizontal percorrida   : {dist_acumulada:.2f} m")
print()
print(f"  Passos SUBINDO                    : {passos_sub_total}")
print(f"  Passos RETORNANDO                 : {passos_re_total}")
print()
print(f"  Energia SUBINDO (Sigma 0.005*v^2) : {energia_subida:8.2f} J")
print(f"  Energia RETORNANDO                : {energia_retorno:8.2f} J")
print(f"  Energia TOTAL da missao           : {energia_total:8.2f} J")
print(f"  Eficiencia                        : {ef_por_metro:.3f} J/m percorrido")
print()
if medicoes:
    print(f"  Medicoes de umidade ({len(medicoes)}):")
    for nome, umid in medicoes:
        print(f"    - {nome:15s} : {umid:.1f}%")
print("=" * 60)

if estado == "FINALIZADO":
    print("  MISSAO CONCLUIDA COM SUCESSO!")
else:
    print(f"  MISSAO ENCERRADA INCOMPLETAMENTE NO ESTADO: {estado}")
print("=" * 60)

env.sim.stopSimulation()
env.close()
print("Sistema desligado.")
