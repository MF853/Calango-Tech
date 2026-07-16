"""
medir_tracao.py - Varredura de TRACAO (slip ratio) vs throttle no robo de 6 rodas.

Otimizacao de tracao = achar o throttle que transfere o maximo de forca pro chao
com o minimo de patinagem (slip). Este script aplica varios throttles fixos e mede:

    slip = 1 - v_robo / (omega_real * raio_da_roda)

    - slip ~ 0   -> tracao perfeita (todo giro vira avanco)
    - slip -> 1  -> roda girando no vazio (perdeu tracao)

Tambem mede ganho de altura e tilt em cada throttle, pra mostrar o trade-off
"andar rapido x patinar x subir".

Como usar:
    1. Abra a cena Calango_6Roda no CoppeliaSim.
    2. Posicione o robo no PE da rampa 1, de frente pra subida (mesmo ponto do treino).
    3. NAO inicie a simulacao manualmente (o script controla).
    4. Rode: python medir_tracao.py

Saida: tabela throttle x slip x subida + identificacao do throttle otimo,
e os dados prontos pra colar no slide (ou gerar grafico).
"""

import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ----------------------------------------------------------------------
# PARAMETROS
# ----------------------------------------------------------------------
NOME_CHASSI = "/Cuboid"
JUNTAS = [f"/Cuboid/JuntaEsquerda{i}" for i in range(1, 4)] + \
         [f"/Cuboid/JuntaDireita{i}"  for i in range(1, 4)]
RODA_REF = "/Cuboid/JuntaDireita1/Cylinder"   # 1 roda p/ medir o raio automaticamente

VELOCIDADE_MAXIMA = 7.0     # rad/s (mesma do treino)
SENTIDO_FRENTE    = +1.0    # positivo = frente neste robo
RAIO_RODA_PADRAO  = 0.0704  # m, fallback se a auto-medicao falhar

THROTTLES = [0.6, 0.7, 0.8, 0.9, 1.0]   # faixa do action shaping [0.6, 1.0]

TEMPO_ASSENTAMENTO_S = 0.6   # descarta o transitorio inicial (aceleracao)
TEMPO_MEDICAO_S      = 2.5   # janela de regime permanente onde medimos slip


def tilt_graus(sim, handle):
    mat = sim.getObjectMatrix(handle, sim.handle_world)
    return math.degrees(math.acos(max(-1.0, min(1.0, mat[10]))))


def pegar_handle(sim, caminho):
    try:
        return sim.getObject(caminho)
    except Exception:
        return sim.getObject("/" + caminho.split("/")[-1])


def medir_raio_roda(sim, handle):
    """Tenta medir o raio da roda pela bounding box (maior semi-extensao).
    Roda fina: o raio (maior) > metade da espessura (menor)."""
    try:
        params = [
            (sim.objfloatparam_objbbox_min_x, sim.objfloatparam_objbbox_max_x),
            (sim.objfloatparam_objbbox_min_y, sim.objfloatparam_objbbox_max_y),
            (sim.objfloatparam_objbbox_min_z, sim.objfloatparam_objbbox_max_z),
        ]
        semis = []
        for pmin, pmax in params:
            vmin = sim.getObjectFloatParam(handle, pmin)
            vmax = sim.getObjectFloatParam(handle, pmax)
            semis.append((vmax - vmin) / 2.0)
        return max(semis), True
    except Exception:
        return RAIO_RODA_PADRAO, False


def main():
    print("\n" + "=" * 66)
    print("  VARREDURA DE TRACAO (SLIP RATIO) - ROBO 6 RODAS")
    print("=" * 66)

    client = RemoteAPIClient()
    sim = client.getObject("sim")
    client.setStepping(True)
    sim.setInt32Signal('python_no_op', 1)   # silencia Lua de outros robos na cena

    robot   = pegar_handle(sim, NOME_CHASSI)
    motores = [pegar_handle(sim, c) for c in JUNTAS]
    print(f"  Chassi e {len(motores)} motores encontrados.")

    # Raio da roda (auto ou fallback)
    try:
        roda = pegar_handle(sim, RODA_REF)
        raio, ok = medir_raio_roda(sim, roda)
    except Exception:
        raio, ok = RAIO_RODA_PADRAO, False
    origem = "medido da cena" if ok else "PADRAO (nao consegui medir)"
    print(f"  Raio da roda usado : {raio:.4f} m  ({origem})")
    if not ok:
        print(f"  [AVISO] Se o raio real for diferente, edite RAIO_RODA_PADRAO no topo.")

    dt = sim.getSimulationTimeStep()
    n_assent = int(TEMPO_ASSENTAMENTO_S / max(dt, 0.001))
    n_medida = int(TEMPO_MEDICAO_S / max(dt, 0.001))

    resultados = []

    for thr in THROTTLES:
        omega_cmd = thr * VELOCIDADE_MAXIMA      # rad/s comandado
        v_roda_cmd = omega_cmd * raio            # m/s superficie da roda (comandado)

        # Reinicia a cena pra cada throttle comecar do mesmo ponto
        sim.stopSimulation()
        deadline = time.time() + 2.0
        while sim.getSimulationState() != sim.simulation_stopped:
            if time.time() > deadline:
                break
            time.sleep(0.01)
        sim.startSimulation()
        time.sleep(0.2)

        pos_ini = sim.getObjectPosition(robot, sim.handle_world)

        # Aplica throttle e deixa assentar (descarta transitorio)
        for m in motores:
            sim.setJointTargetVelocity(m, SENTIDO_FRENTE * omega_cmd)
        for _ in range(n_assent):
            client.step()

        pos_reg_ini = sim.getObjectPosition(robot, sim.handle_world)

        # Janela de regime permanente: amostra omega real e v real
        omegas = []
        v_robos = []
        tilts = []
        for _ in range(n_medida):
            client.step()
            # velocidade angular real media das 6 rodas
            w = np.mean([abs(sim.getJointVelocity(m)) for m in motores])
            omegas.append(float(w))
            v_lin, _ = sim.getObjectVelocity(robot)
            v_robos.append(float(math.hypot(v_lin[0], v_lin[1])))  # horizontal
            tilts.append(tilt_graus(sim, robot))

        for m in motores:
            sim.setJointTargetVelocity(m, 0.0)

        pos_fim = sim.getObjectPosition(robot, sim.handle_world)

        omega_real = float(np.mean(omegas))
        v_robo     = float(np.mean(v_robos))
        v_roda_real = omega_real * raio
        # slip baseado na velocidade real da roda (regime)
        slip = 1.0 - (v_robo / v_roda_real) if v_roda_real > 1e-6 else 1.0
        slip = max(0.0, min(1.0, slip))
        ganho_alt = pos_fim[2] - pos_reg_ini[2]
        tilt_med  = float(np.mean(tilts))
        energia   = 0.005 * (omega_cmd ** 2)   # proxy por passo

        resultados.append({
            "thr": thr, "omega_cmd": omega_cmd, "omega_real": omega_real,
            "v_roda_cmd": v_roda_cmd, "v_robo": v_robo, "slip": slip,
            "ganho": ganho_alt, "tilt": tilt_med, "energia": energia,
        })

        print(f"  throttle {thr:.1f}: v_roda={v_roda_real:.3f}  v_robo={v_robo:.3f}  "
              f"slip={slip*100:4.1f}%  ganho={ganho_alt:+.3f}m  tilt={tilt_med:4.1f}")

    sim.stopSimulation()

    # ------------------------------------------------------------------
    # TABELA E ANALISE
    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("  TABELA DE TRACAO")
    print("=" * 66)
    print(f"  {'thr':>4} {'w_cmd':>6} {'w_real':>7} {'v_roda':>7} {'v_robo':>7} "
          f"{'slip%':>6} {'ganho_m':>8} {'tilt':>6}")
    for r in resultados:
        print(f"  {r['thr']:4.1f} {r['omega_cmd']:6.2f} {r['omega_real']:7.2f} "
              f"{r['omega_real']*raio:7.3f} {r['v_robo']:7.3f} {r['slip']*100:6.1f} "
              f"{r['ganho']:+8.3f} {r['tilt']:6.1f}")

    # Throttle otimo: menor slip ENTRE os que efetivamente sobem (ganho > 0)
    sobem = [r for r in resultados if r["ganho"] > 0.02]
    print("\n" + "-" * 66)
    if sobem:
        melhor_slip = min(sobem, key=lambda r: r["slip"])
        # melhor "subida por energia"
        melhor_ef = max(sobem, key=lambda r: r["ganho"] / max(r["energia"], 1e-6))
        print(f"  Menor slip (mais tracao) entre os que sobem : "
              f"throttle {melhor_slip['thr']:.1f}  (slip={melhor_slip['slip']*100:.1f}%)")
        print(f"  Melhor subida-por-energia                   : "
              f"throttle {melhor_ef['thr']:.1f}  "
              f"(ganho/E = {melhor_ef['ganho']/max(melhor_ef['energia'],1e-6):.2f})")
        print(f"\n  >> O throttle de TRACAO OTIMA fica em torno de {melhor_slip['thr']:.1f}.")
        print(f"     Compare com o throttle medio que a IA escolheu (~0.8 = acao 0.5):")
        print(f"     se a IA convergiu perto desse ponto, ela APRENDEU tracao otima.")
    else:
        print("  Nenhum throttle gerou subida clara nesta posicao.")
        print("  Verifique se o robo esta de frente pra rampa.")
    print("=" * 66)
    print("\n  Manda essa tabela que eu te ajudo a montar o grafico slip x throttle.")


if __name__ == "__main__":
    main()
