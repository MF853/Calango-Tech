"""
grafico_tracao.py - Gera o grafico slip x throttle do robo de 6 rodas (pro slide).

Usa os dados ja medidos por medir_tracao.py. Plota:
  1. slip (%) x throttle  -> linha quase reta ~9% = "tracao de sobra"
  2. linha vertical no throttle onde a IA convergiu (~0.6)
  3. (eixo secundario) ganho de altura x throttle, pra mostrar o trade-off

Saida: salva 'grafico_tracao.png' (300 dpi, pronto pra colar no PowerPoint)
e tambem abre a janela na tela.

Como usar:
    python grafico_tracao.py
"""

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# DADOS MEDIDOS (de medir_tracao.py - cena Calango_6Roda)
# ----------------------------------------------------------------------
throttle = [0.6, 0.7, 0.8, 0.9, 1.0]
slip_pct = [8.3, 10.0, 9.4, 9.8, 8.8]    # %
ganho_m  = [0.686, 1.035, 1.136, 1.336, 1.443]  # m por janela de medicao

THROTTLE_IA = 0.60   # onde a IA convergiu (acao ~0.00-0.03 -> throttle 0.60-0.61)
SLIP_MEDIO  = sum(slip_pct) / len(slip_pct)


def main():
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Eixo 1: slip x throttle ---
    cor_slip = "#c0392b"
    ax1.plot(throttle, slip_pct, "o-", color=cor_slip, linewidth=2.5,
             markersize=8, label="Slip medido")
    ax1.axhline(SLIP_MEDIO, color=cor_slip, linestyle=":", alpha=0.5,
                label=f"Slip medio ({SLIP_MEDIO:.1f}%)")
    ax1.set_xlabel("Throttle (fracao da velocidade maxima)", fontsize=12)
    ax1.set_ylabel("Slip ratio (%)", color=cor_slip, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=cor_slip)
    ax1.set_ylim(0, 20)
    ax1.set_xticks(throttle)
    ax1.grid(True, alpha=0.3)

    # --- Linha vertical: onde a IA convergiu ---
    ax1.axvline(THROTTLE_IA, color="#27ae60", linestyle="--", linewidth=2)
    ax1.annotate("IA convergiu aqui\n(tracao otima)",
                 xy=(THROTTLE_IA, 15.5), xytext=(0.68, 16.5),
                 color="#27ae60", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#27ae60"))

    # --- Eixo 2: ganho de altura x throttle ---
    ax2 = ax1.twinx()
    cor_ganho = "#2980b9"
    ax2.plot(throttle, ganho_m, "s--", color=cor_ganho, linewidth=2,
             markersize=7, alpha=0.7, label="Ganho de altura")
    ax2.set_ylabel("Ganho de altura na janela (m)", color=cor_ganho, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=cor_ganho)
    ax2.set_ylim(0, 1.8)

    # --- Titulo e legenda combinada ---
    plt.title("Tracao do robo de 6 rodas: slip ~9% constante\n"
              "(tracao independente do throttle = tracao de sobra)",
              fontsize=12, fontweight="bold")

    linhas1, labels1 = ax1.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, labels1 + labels2,
               loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig("grafico_tracao.png", dpi=300, bbox_inches="tight")
    print("  Grafico salvo em: grafico_tracao.png")
    plt.show()


if __name__ == "__main__":
    main()
