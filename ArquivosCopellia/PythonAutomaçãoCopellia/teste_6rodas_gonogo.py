"""
teste_6rodas_gonogo.py - Teste GO/NO-GO do robo de 6 rodas (Calango_6Roda).

Objetivo: confirmar que a fisica fecha ANTES de gastar horas treinando.
Aplica velocidade fixa nos 6 motores e mede se o robo:
  - anda pra frente,
  - ganha altura na rampa,
  - NAO capota (tilt < 60 graus).

Como usar:
    1. Abra a cena Calango_6Roda no CoppeliaSim.
    2. Posicione o robo de 6 rodas no PE da rampa 1, de FRENTE pra subida.
    3. NAO inicie a simulacao manualmente (o script controla).
    4. Rode: python teste_6rodas_gonogo.py

Convencao deste robo: velocidade POSITIVA = pra frente (confirmado pelo usuario).
"""

import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ----------------------------------------------------------------------
# PARAMETROS
# ----------------------------------------------------------------------
NOME_CHASSI   = "/Cuboid"          # objeto-pai (le posicao/orientacao daqui)
JUNTAS_DIR    = ["/Cuboid/JuntaDireita1", "/Cuboid/JuntaDireita2", "/Cuboid/JuntaDireita3"]
JUNTAS_ESQ    = ["/Cuboid/JuntaEsquerda1", "/Cuboid/JuntaEsquerda2", "/Cuboid/JuntaEsquerda3"]

VELOCIDADE      = 7.0     # rad/s (mesma do caterpillar -> comparacao justa)
TORQUE_MAXIMO   = 1000.0  # N*m (garante forca pra subir)
SENTIDO_FRENTE  = +1.0    # positivo = frente neste robo
TEMPO_TESTE_S   = 6.0     # segundos de simulacao
TILT_CAPOTAR    = 60.0    # graus


def tilt_graus(sim, handle):
    """Inclinacao do robo em relacao a vertical, via matriz de rotacao.
    mat[10] = cosseno do angulo entre o eixo Z do robo e o Z do mundo.
    Invariante a yaw (nao confunde girar com tombar)."""
    mat = sim.getObjectMatrix(handle, sim.handle_world)
    cos_tilt = max(-1.0, min(1.0, mat[10]))
    return math.degrees(math.acos(cos_tilt))


def pegar_handle(sim, caminho):
    try:
        return sim.getObject(caminho)
    except Exception:
        # fallback: tenta so o ultimo nome (alias)
        alias = "/" + caminho.split("/")[-1]
        return sim.getObject(alias)


def main():
    print("\n" + "=" * 60)
    print("  TESTE GO/NO-GO - ROBO 6 RODAS")
    print("=" * 60)

    client = RemoteAPIClient()
    sim = client.getObject("sim")
    client.setStepping(True)

    # ---- handles ----
    try:
        robot = pegar_handle(sim, NOME_CHASSI)
    except Exception as e:
        print(f"  [ERRO] Nao encontrei o chassi {NOME_CHASSI}: {e}")
        print("         Confira o nome do objeto-pai na hierarquia da cena.")
        return

    motores = []
    faltando = []
    for c in JUNTAS_ESQ + JUNTAS_DIR:
        try:
            motores.append(pegar_handle(sim, c))
        except Exception:
            faltando.append(c)
    if faltando:
        print(f"  [ERRO] Juntas nao encontradas: {faltando}")
        print("         Confira os nomes na hierarquia.")
        return
    print(f"  [OK] Chassi e {len(motores)} motores encontrados.")

    # ---- garante torque alto via API (caso a cena esteja diferente) ----
    for m in motores:
        try:
            sim.setJointTargetForce(m, TORQUE_MAXIMO)
        except Exception:
            pass

    dt = sim.getSimulationTimeStep()
    print(f"  dt do simulador: {dt*1000:.0f} ms")

    # ---- inicia ----
    sim.startSimulation()
    time.sleep(0.3)

    pos0 = sim.getObjectPosition(robot, sim.handle_world)
    print(f"  Posicao inicial (x,y,z): ({pos0[0]:+.3f}, {pos0[1]:+.3f}, {pos0[2]:+.3f})")
    print(f"  Tilt inicial: {tilt_graus(sim, robot):.1f} graus\n")

    # ---- aplica velocidade nos 6 motores ----
    for m in motores:
        sim.setJointTargetVelocity(m, SENTIDO_FRENTE * VELOCIDADE)

    n_passos = int(TEMPO_TESTE_S / max(dt, 0.001))
    amostras = []
    capotou = False
    for k in range(n_passos):
        client.step()
        if k % max(1, n_passos // 12) == 0:
            p = sim.getObjectPosition(robot, sim.handle_world)
            t = tilt_graus(sim, robot)
            v_lin, _ = sim.getObjectVelocity(robot)
            amostras.append((k * dt, p[0], p[1], p[2], t, float(np.linalg.norm(v_lin))))
            if t > TILT_CAPOTAR:
                capotou = True
                break

    # ---- para motores ----
    for m in motores:
        sim.setJointTargetVelocity(m, 0.0)

    # ---- relatorio ----
    print(f"  {'t(s)':>5} {'x':>8} {'y':>8} {'z':>8} {'tilt':>6} {'|v|':>7}")
    for t, x, y, z, til, v in amostras:
        print(f"  {t:5.2f} {x:+8.3f} {y:+8.3f} {z:+8.3f} {til:6.1f} {v:7.3f}")

    posf = sim.getObjectPosition(robot, sim.handle_world)
    delta = np.array(posf) - np.array(pos0)
    dist_horiz = float(np.linalg.norm(delta[:2]))
    ganho_alt = float(delta[2])
    tilt_max = max(a[4] for a in amostras)

    print("\n" + "-" * 60)
    print(f"  Distancia horizontal : {dist_horiz:.3f} m")
    print(f"  Ganho de altura      : {ganho_alt:+.3f} m")
    print(f"  Tilt maximo          : {tilt_max:.1f} graus")
    print("-" * 60)

    # ---- veredito ----
    print("\n  VEREDITO:")
    if capotou or tilt_max > TILT_CAPOTAR:
        print("  >> NO-GO: CAPOTOU. Ajuste atrito/centro de massa/posicao das rodas")
        print("            ANTES de treinar. Nao adianta treinar um robo que tomba.")
    elif dist_horiz < 0.1:
        print("  >> NO-GO: nao saiu do lugar. Verifique torque, atrito do terreno,")
        print("            ou se os motores estao em velocity control.")
    elif ganho_alt < 0.05:
        print("  >> ATENCAO: andou mas nao subiu. Confirme que esta de FRENTE pra")
        print("             rampa. Se estava no plano, isso e esperado - reposicione.")
    else:
        print(f"  >> GO! Subiu {ganho_alt:.2f} m com tilt max {tilt_max:.1f} graus, sem capotar.")
        print("        A fisica fecha. Pode treinar com seguranca.")

    sim.stopSimulation()
    print("\n  Teste encerrado. Manda esse output que eu te digo o proximo passo.")


if __name__ == "__main__":
    main()
