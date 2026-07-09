/**
 * ============================================================
 *  Carrinho de Esteiras — ESP32 + L298N + Xbox One S
 *  *** COM SERVIDOR HTTP PARA DASHBOARD DE SENSORES ***
 * ============================================================
 *
 *  Hardware:
 *    - ESP32 DevKit V1
 *    - Ponte H L298N (controle de 2 motores DC)
 *    - Controle Xbox One S (Bluetooth Classic)
 *    - MPU-6050 (acelerômetro + giroscópio — I²C)
 *    - Micro Servo (qualquer servo padrão 5 V)
 *    - Sensor de umidade capacitivo HW-390 v2.0.0
 *
 *  Bibliotecas necessárias:
 *    • Bluepad32: instale pelo Boards Manager.
 *      URL: https://raw.githubusercontent.com/ricardoquesada/
 *           esp32-arduino-lib-builder/master/bluepad32_files/
 *           package_esp32_bluepad32_index.json
 *      Selecione a placa: ESP32 + Bluepad32 > ESP32 Dev Module
 *    • MPU6050 (jrowberg/i2cdevlib): instale pela Library Manager
 *      ou em https://github.com/jrowberg/i2cdevlib
 *    • ESP32Servo: instale pela Library Manager
 *      (busque por "ESP32Servo" de Kevin Harrington)
 *    • WiFi.h e WebServer.h: já inclusas no ESP32 Arduino Core
 *
 *  Servidor HTTP:
 *    • GET /       → Dashboard HTML (dark glassmorphism, tempo real)
 *    • GET /dados  → JSON com todos os dados dos sensores
 *
 * ============================================================
 *  PINAGEM
 * ============================================================
 *
 *   L298N          ESP32 DevKit V1
 *  --------        ---------------
 *  ENA        →   5V (fixo — velocidade máxima sempre)
 *  IN1        →   GPIO 27   Motor A direção +
 *  IN2        →   GPIO 26   Motor A direção -
 *
 *  ENB        →   5V (fixo — velocidade máxima sempre)
 *  IN3        →   GPIO 25   Motor B direção +
 *  IN4        →   GPIO 33   Motor B direção -
 *
 *  12V        →   Bateria (7.4 V a 12 V)
 *  GND        →   GND compartilhado com ESP32
 *  5V (saída) →   Alimenta ENA, ENB, Servo e ESP32 via VIN
 *
 * ============================================================
 *  PINAGEM — MPU-6050
 * ============================================================
 *
 *   MPU-6050       ESP32 DevKit V1
 *  ----------      ---------------
 *  VCC        →   3.3 V (ou 5 V dependendo do módulo)
 *  GND        →   GND
 *  SDA        →   GPIO 21   (I²C padrão ESP32)
 *  SCL        →   GPIO 22   (I²C padrão ESP32)
 *  AD0        →   GND       (endereço I²C = 0x68)
 *
 * ============================================================
 *  PINAGEM — Micro Servo & HW-390
 * ============================================================
 *
 *   Micro Servo    ESP32 DevKit V1
 *  ------------    ---------------
 *  Sinal (laranj)→ GPIO 14   (liberto do ENA)
 *  VCC  (verm.)  → 5 V (saída do L298N ou bateria)
 *  GND  (preto)  → GND
 *
 *   HW-390         ESP32 DevKit V1
 *  ----------      ---------------
 *  VCC        →   3.3 V
 *  GND        →   GND
 *  AOUT       →   GPIO 32   (ADC1 — liberto do ENB)
 *
 *  Saída: valor ADC bruto de 0 a 4095 (12 bits)
 *
 * ============================================================
 */

#include <Bluepad32.h>
#include <Wire.h>
#include <MPU6050.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebServer.h>

// ============================================================
//  CONFIGURAÇÃO WI-FI — preencha antes de compilar!
// ============================================================
const char* WIFI_SSID     = "MarioS23";      // Nome da rede Wi-Fi
const char* WIFI_PASSWORD = "12345678";     // Senha da rede Wi-Fi

// ── Pinos Motor A (Esquerdo) ──────────────────────────────────
// ENA → ligado diretamente ao 5 V na fiação (velocidade máxima fixa)
#define MOTOR_A_IN1  27   // Direção +
#define MOTOR_A_IN2  26   // Direção -

// ── Pinos Motor B (Direito) ───────────────────────────────────
// ENB → ligado diretamente ao 5 V na fiação (velocidade máxima fixa)
#define MOTOR_B_IN3  25   // Direção +
#define MOTOR_B_IN4  33   // Direção -

// ── Zona morta do analógico (evita drift em repouso) ──────────
// Bluepad32 retorna -512 a +511 para os eixos analógicos
#define DEADZONE  40

// ── Micro Servo ───────────────────────────────────────────────
#define SERVO_PIN        14    // Sinal do servo (GPIO liberto do ENA)
#define SERVO_REPOUSO     0    // Posição inicial / repouso (graus)
#define SERVO_MEDICAO    90    // Posição de coleta de umidade (graus)
#define SERVO_DELAY_MS  600    // Tempo (ms) para o servo atingir a posição

// ── Sensor HW-390 v2.0.0 (capacitivo de umidade) ──────────────
#define UMIDADE_PIN      32    // GPIO analógico ADC1 (liberto do ENB)

// ── Calibração do HW-390 (ajuste conforme o seu sensor) ───────
// Valor ADC quando o sensor está NO AR (completamente seco)
#define HW390_SECO      3200
// Valor ADC quando o sensor está NA ÁGUA (completamente molhado)
#define HW390_MOLHADO    800

// ─────────────────────────────────────────────────────────────
//  MPU-6050 — Variáveis globais
// ─────────────────────────────────────────────────────────────

MPU6050 mpu;

// Ângulos filtrados (graus)
float gPitch = 0.0f;   // inclinação frente/trás
float gRoll  = 0.0f;   // inclinação lateral

// Inclinação armazenada ao pressionar botão A
float gPitchSalvo = 0.0f;
float gRollSalvo  = 0.0f;
bool  gTemLeituraSalva = false;

// Leitura bruta do HW-390 armazenada ao pressionar botão A
int gUmidadeADC = 0;

// Controle do filtro complementar
unsigned long gTempoAnterior = 0;
const float ALPHA = 0.98f;  // 0.98 → favorece giroscópio (menos ruído)

// Controle de borda do botão A (evita leituras repetidas)
bool gBotaoAAnterior = false;

// Estado atual do servo (graus)
int gServoPosicao = SERVO_REPOUSO;

// Objeto do servo
Servo gServo;

// ── Ponteiro global para o gamepad conectado ──────────────────
GamepadPtr gGamepad = nullptr;

// ─────────────────────────────────────────────────────────────
//  Servidor HTTP
// ─────────────────────────────────────────────────────────────

WebServer server(80);

// ─────────────────────────────────────────────────────────────
//  HTML da Dashboard (armazenado na Flash via PROGMEM)
// ─────────────────────────────────────────────────────────────

const char DASHBOARD_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Calango-Tech | Dashboard de Sensores</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0a0e1a;
      --surface:   rgba(255,255,255,0.05);
      --border:    rgba(255,255,255,0.10);
      --accent1:   #00d4ff;
      --accent2:   #7c3aed;
      --accent3:   #10b981;
      --accent4:   #f59e0b;
      --text:      #e2e8f0;
      --muted:     #64748b;
      --danger:    #ef4444;
      --radius:    16px;
      --glow1:     0 0 30px rgba(0,212,255,0.2);
      --glow2:     0 0 30px rgba(124,58,237,0.2);
      --glow3:     0 0 30px rgba(16,185,129,0.2);
    }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      background-image:
        radial-gradient(ellipse at 10% 20%, rgba(0,212,255,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(124,58,237,0.06) 0%, transparent 50%);
    }

    /* ── Header ── */
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 20px 32px;
      border-bottom: 1px solid var(--border);
      background: rgba(10,14,26,0.8);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-icon {
      width: 40px; height: 40px;
      background: linear-gradient(135deg, var(--accent1), var(--accent2));
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px;
    }
    .logo h1 { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.5px; }
    .logo span { color: var(--accent1); }

    .status-pill {
      display: flex; align-items: center; gap: 8px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 50px;
      padding: 6px 16px;
      font-size: 0.8rem;
      color: var(--muted);
      transition: all 0.3s;
    }
    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--accent3);
      box-shadow: 0 0 8px var(--accent3);
      animation: pulse 2s infinite;
    }
    .status-dot.offline { background: var(--danger); box-shadow: 0 0 8px var(--danger); animation: none; }

    @keyframes pulse {
      0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
    }

    /* ── Layout principal ── */
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px;
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }

    /* ── Cards ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      backdrop-filter: blur(8px);
      transition: transform 0.2s, box-shadow 0.2s;
      position: relative;
      overflow: hidden;
    }
    .card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      border-radius: var(--radius) var(--radius) 0 0;
    }
    .card.cyan::before   { background: linear-gradient(90deg, var(--accent1), transparent); }
    .card.purple::before { background: linear-gradient(90deg, var(--accent2), transparent); }
    .card.green::before  { background: linear-gradient(90deg, var(--accent3), transparent); }
    .card.amber::before  { background: linear-gradient(90deg, var(--accent4), transparent); }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .card-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--muted);
      margin-bottom: 16px;
      display: flex; align-items: center; gap: 8px;
    }
    .card-label-icon { font-size: 1rem; }

    /* ── Gauge SVG ── */
    .gauge-wrap {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    .gauge-svg { width: 180px; height: 100px; overflow: visible; }
    .gauge-track { fill: none; stroke: rgba(255,255,255,0.08); stroke-width: 14; stroke-linecap: round; }
    .gauge-fill  { fill: none; stroke-width: 14; stroke-linecap: round; transition: stroke-dashoffset 0.6s cubic-bezier(0.4,0,0.2,1); }
    .gauge-value { font-size: 2rem; font-weight: 700; text-anchor: middle; dominant-baseline: middle; }
    .gauge-unit  { font-size: 0.75rem; fill: var(--muted); text-anchor: middle; }

    /* ── Umidade ── */
    .moisture-bar-wrap {
      margin-top: 8px;
    }
    .moisture-header {
      display: flex; justify-content: space-between; align-items: baseline;
      margin-bottom: 10px;
    }
    .moisture-value {
      font-size: 2.5rem;
      font-weight: 700;
      color: var(--accent3);
      line-height: 1;
    }
    .moisture-unit { font-size: 1rem; color: var(--muted); }
    .moisture-adc  { font-size: 0.8rem; color: var(--muted); }
    .bar-bg {
      width: 100%; height: 14px;
      background: rgba(255,255,255,0.08);
      border-radius: 99px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 99px;
      transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
      background: linear-gradient(90deg, var(--accent3), #6ee7b7);
      box-shadow: 0 0 10px rgba(16,185,129,0.4);
    }
    .moisture-labels {
      display: flex; justify-content: space-between;
      font-size: 0.7rem; color: var(--muted);
      margin-top: 6px;
    }

    /* ── Servo ── */
    .servo-wrap {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }
    .servo-visual {
      width: 120px; height: 120px;
      position: relative;
      display: flex; align-items: center; justify-content: center;
    }
    .servo-circle {
      width: 100%; height: 100%;
      border-radius: 50%;
      border: 3px solid rgba(245,158,11,0.3);
      position: relative;
      display: flex; align-items: center; justify-content: center;
    }
    .servo-arm {
      width: 4px; height: 44px;
      background: linear-gradient(to top, var(--accent4), #fcd34d);
      border-radius: 4px;
      transform-origin: bottom center;
      transition: transform 0.6s cubic-bezier(0.4,0,0.2,1);
      box-shadow: 0 0 12px rgba(245,158,11,0.5);
    }
    .servo-center {
      position: absolute;
      width: 14px; height: 14px;
      background: var(--accent4);
      border-radius: 50%;
      box-shadow: 0 0 12px rgba(245,158,11,0.7);
    }
    .servo-angle {
      font-size: 1.8rem; font-weight: 700;
      color: var(--accent4);
    }
    .servo-state {
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    /* ── Gamepad ── */
    .gamepad-wrap {
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 12px;
      padding: 8px 0;
    }
    .gamepad-icon {
      font-size: 3.5rem;
      filter: drop-shadow(0 0 12px rgba(0,212,255,0.5));
      transition: all 0.4s;
    }
    .gamepad-icon.offline { filter: grayscale(1) drop-shadow(none); opacity: 0.3; }
    .gamepad-text {
      font-size: 0.85rem;
      font-weight: 500;
    }
    .gamepad-text.online  { color: var(--accent3); }
    .gamepad-text.offline { color: var(--muted); }

    /* ── Gráfico ── */
    .chart-card {
      grid-column: 1 / -1;
    }
    .chart-wrap { position: relative; height: 220px; }

    /* ── Última coleta ── */
    .last-card {
      grid-column: 1 / -1;
    }
    .last-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 4px;
    }
    .last-item {
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
    }
    .last-item-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .last-item-value { font-size: 1.3rem; font-weight: 700; }

    /* ── Footer ── */
    footer {
      text-align: center;
      padding: 20px;
      font-size: 0.75rem;
      color: var(--muted);
      border-top: 1px solid var(--border);
    }
    footer span { color: var(--accent1); }
  </style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🦎</div>
    <h1>Calango<span>-Tech</span></h1>
  </div>
  <div class="status-pill" id="status-pill">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text">Conectando...</span>
  </div>
</header>

<main>

  <!-- Pitch -->
  <div class="card cyan">
    <div class="card-label">
      <span class="card-label-icon">📐</span> Pitch — Frente / Trás
    </div>
    <div class="gauge-wrap">
      <svg class="gauge-svg" viewBox="0 0 200 110">
        <!-- trilha semi-circular -->
        <path class="gauge-track"
          d="M 20,95 A 80,80 0 0 1 180,95"
          stroke-dasharray="251.2" stroke-dashoffset="0"/>
        <!-- preenchimento -->
        <path id="gauge-pitch-fill" class="gauge-fill"
          d="M 20,95 A 80,80 0 0 1 180,95"
          stroke="url(#grad-pitch)"
          stroke-dasharray="251.2" stroke-dashoffset="125.6"/>
        <defs>
          <linearGradient id="grad-pitch" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#00d4ff"/>
            <stop offset="100%" stop-color="#7c3aed"/>
          </linearGradient>
        </defs>
        <text id="gauge-pitch-val" class="gauge-value" x="100" y="72" fill="#e2e8f0">0.0°</text>
        <text class="gauge-unit" x="100" y="95">-90°          +90°</text>
      </svg>
    </div>
  </div>

  <!-- Roll -->
  <div class="card purple">
    <div class="card-label">
      <span class="card-label-icon">🔄</span> Roll — Lateral
    </div>
    <div class="gauge-wrap">
      <svg class="gauge-svg" viewBox="0 0 200 110">
        <path class="gauge-track"
          d="M 20,95 A 80,80 0 0 1 180,95"
          stroke-dasharray="251.2" stroke-dashoffset="0"/>
        <path id="gauge-roll-fill" class="gauge-fill"
          d="M 20,95 A 80,80 0 0 1 180,95"
          stroke="url(#grad-roll)"
          stroke-dasharray="251.2" stroke-dashoffset="125.6"/>
        <defs>
          <linearGradient id="grad-roll" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#7c3aed"/>
            <stop offset="100%" stop-color="#ec4899"/>
          </linearGradient>
        </defs>
        <text id="gauge-roll-val" class="gauge-value" x="100" y="72" fill="#e2e8f0">0.0°</text>
        <text class="gauge-unit" x="100" y="95">-90°          +90°</text>
      </svg>
    </div>
  </div>

  <!-- Umidade -->
  <div class="card green">
    <div class="card-label">
      <span class="card-label-icon">💧</span> Umidade do Solo — HW-390
    </div>
    <div class="moisture-bar-wrap">
      <div class="moisture-header">
        <div>
          <span class="moisture-value" id="moisture-pct">—</span>
          <span class="moisture-unit">%</span>
        </div>
        <div class="moisture-adc">ADC: <span id="moisture-adc">—</span></div>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" id="moisture-bar" style="width:0%"></div>
      </div>
      <div class="moisture-labels"><span>Seco</span><span>Úmido</span></div>
    </div>
  </div>

  <!-- Servo -->
  <div class="card amber">
    <div class="card-label">
      <span class="card-label-icon">⚙️</span> Micro Servo
    </div>
    <div class="servo-wrap">
      <div class="servo-visual">
        <div class="servo-circle">
          <div class="servo-arm" id="servo-arm" style="transform: rotate(0deg)"></div>
          <div class="servo-center"></div>
        </div>
      </div>
      <div class="servo-angle" id="servo-angle">0°</div>
      <div class="servo-state" id="servo-state">Repouso</div>
    </div>
  </div>

  <!-- Gamepad -->
  <div class="card cyan">
    <div class="card-label">
      <span class="card-label-icon">🎮</span> Controle Xbox
    </div>
    <div class="gamepad-wrap">
      <div class="gamepad-icon" id="gamepad-icon">🎮</div>
      <div class="gamepad-text online" id="gamepad-text">Aguardando...</div>
    </div>
  </div>

  <!-- Gráfico -->
  <div class="card chart-card">
    <div class="card-label">
      <span class="card-label-icon">📈</span> Histórico — Pitch & Roll (últimas 30 amostras)
    </div>
    <div class="chart-wrap">
      <canvas id="imu-chart"></canvas>
    </div>
  </div>

  <!-- Última coleta (botão A) -->
  <div class="card last-card">
    <div class="card-label">
      <span class="card-label-icon">📋</span> Última Coleta (Botão A)
    </div>
    <div class="last-grid">
      <div class="last-item">
        <div class="last-item-label">Pitch Salvo</div>
        <div class="last-item-value" id="last-pitch" style="color:var(--accent1)">—</div>
      </div>
      <div class="last-item">
        <div class="last-item-label">Roll Salvo</div>
        <div class="last-item-value" id="last-roll" style="color:var(--accent2)">—</div>
      </div>
      <div class="last-item">
        <div class="last-item-label">Umidade ADC</div>
        <div class="last-item-value" id="last-adc" style="color:var(--accent3)">—</div>
      </div>
      <div class="last-item">
        <div class="last-item-label">Umidade %</div>
        <div class="last-item-value" id="last-pct" style="color:var(--accent3)">—</div>
      </div>
    </div>
  </div>

</main>

<footer>
  Calango-Tech &nbsp;|&nbsp; ESP32 DevKit V1 &nbsp;|&nbsp;
  Atualização automática a cada <span>1 s</span> &nbsp;|&nbsp;
  Servidor rodando na porta <span>80</span>
</footer>

<script>
  // ── Configuração do Gráfico ──
  const MAX_PONTOS = 30;
  const labels   = Array(MAX_PONTOS).fill('');
  const pitchData = Array(MAX_PONTOS).fill(null);
  const rollData  = Array(MAX_PONTOS).fill(null);

  const ctx = document.getElementById('imu-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Pitch (°)',
          data: pitchData,
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0,212,255,0.08)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.4
        },
        {
          label: 'Roll (°)',
          data: rollData,
          borderColor: '#7c3aed',
          backgroundColor: 'rgba(124,58,237,0.08)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 }, boxWidth: 14 }
        }
      },
      scales: {
        x: { display: false },
        y: {
          min: -90, max: 90,
          grid:  { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#64748b', font: { family: 'Inter', size: 11 }, callback: v => v + '°' }
        }
      }
    }
  });

  // ── Funções de atualização da UI ──

  /**
   * Atualiza um gauge semi-circular.
   * @param {string} fillId  — id do <path> SVG de preenchimento
   * @param {string} valId   — id do <text> SVG de valor
   * @param {number} graus   — valor atual em graus (-90 a +90)
   * @param {number} minGrau — mínimo esperado (ex: -90)
   * @param {number} maxGrau — máximo esperado (ex: +90)
   */
  function atualizarGauge(fillId, valId, graus, minGrau, maxGrau) {
    const totalArc = 251.2;                             // comprimento do semi-arco SVG
    const pct = (graus - minGrau) / (maxGrau - minGrau);
    const offset = totalArc * (1 - Math.min(Math.max(pct, 0), 1));
    document.getElementById(fillId).style.strokeDashoffset = offset.toFixed(1);
    document.getElementById(valId).textContent = graus.toFixed(1) + '°';
  }

  function atualizarServoBraco(graus) {
    // O braço parte de -90° (visual) e vai até +90°
    // Mapeamos 0° → -90deg CSS e 90° → +90deg CSS (meia volta)
    const cssRot = (graus / 90) * 90 - 90;
    document.getElementById('servo-arm').style.transform = `rotate(${cssRot}deg)`;
    document.getElementById('servo-angle').textContent = graus + '°';
    document.getElementById('servo-state').textContent = graus === 0 ? 'Repouso' : 'Medição';
  }

  function atualizarGamepad(conectado) {
    const icon = document.getElementById('gamepad-icon');
    const text = document.getElementById('gamepad-text');
    if (conectado) {
      icon.classList.remove('offline');
      text.className = 'gamepad-text online';
      text.textContent = 'Conectado';
    } else {
      icon.classList.add('offline');
      text.className = 'gamepad-text offline';
      text.textContent = 'Desconectado';
    }
  }

  // ── Polling ──
  async function buscarDados() {
    try {
      const resp = await fetch('/dados');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const d = await resp.json();

      // Status pill
      document.getElementById('status-dot').classList.remove('offline');
      document.getElementById('status-text').textContent = 'Online';

      // Gauges IMU
      atualizarGauge('gauge-pitch-fill', 'gauge-pitch-val', d.pitch, -90, 90);
      atualizarGauge('gauge-roll-fill',  'gauge-roll-val',  d.roll,  -90, 90);

      // Umidade
      const pct = Math.min(Math.max(d.umidade_pct, 0), 100);
      document.getElementById('moisture-pct').textContent = pct.toFixed(1);
      document.getElementById('moisture-adc').textContent = d.umidade_adc;
      document.getElementById('moisture-bar').style.width = pct.toFixed(1) + '%';

      // Servo
      atualizarServoBraco(d.servo_graus);

      // Gamepad
      atualizarGamepad(d.gamepad);

      // Histórico gráfico
      pitchData.shift(); pitchData.push(d.pitch);
      rollData.shift();  rollData.push(d.roll);
      chart.update('none');

      // Última coleta (botão A)
      if (d.tem_leitura_salva) {
        document.getElementById('last-pitch').textContent = d.pitch_salvo.toFixed(2) + '°';
        document.getElementById('last-roll').textContent  = d.roll_salvo.toFixed(2) + '°';
        document.getElementById('last-adc').textContent   = d.umidade_adc_salvo;
        document.getElementById('last-pct').textContent   = d.umidade_pct_salvo.toFixed(1) + '%';
      }

    } catch (e) {
      document.getElementById('status-dot').classList.add('offline');
      document.getElementById('status-text').textContent = 'Sem sinal';
      console.warn('Erro ao buscar dados:', e.message);
    }
  }

  buscarDados();
  setInterval(buscarDados, 1000);
</script>

</body>
</html>
)rawliteral";

// ─────────────────────────────────────────────────────────────
//  Callbacks do Bluepad32
// ─────────────────────────────────────────────────────────────

void onConnectedGamepad(GamepadPtr gp) {
    gGamepad = gp;
    Serial.println("[BT] Controle conectado.");
}

void onDisconnectedGamepad(GamepadPtr gp) {
    gGamepad = nullptr;
    pararMotores();   // segurança: para tudo ao desconectar
    Serial.println("[BT] Controle desconectado.");
}

// ─────────────────────────────────────────────────────────────
//  MPU-6050 — Funções
// ─────────────────────────────────────────────────────────────

/**
 * Lê o MPU-6050, atualiza gPitch e gRoll com filtro complementar.
 * Deve ser chamada a cada iteração do loop.
 */
void atualizarIMU() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Converte para unidades físicas
    float Ax = ax / 16384.0f;  // g  (fundo de escala ±2 g)
    float Ay = ay / 16384.0f;
    float Az = az / 16384.0f;
    float Gx = gx / 131.0f;   // °/s (fundo de escala ±250 °/s)
    float Gy = gy / 131.0f;

    // Delta de tempo em segundos
    unsigned long agora = millis();
    float dt = (agora - gTempoAnterior) / 1000.0f;
    gTempoAnterior = agora;

    // Ângulos calculados apenas pelo acelerômetro
    float pitchAcel = atan2f(Ax, sqrtf(Ay*Ay + Az*Az)) * 180.0f / PI;
    float rollAcel  = atan2f(Ay, sqrtf(Ax*Ax + Az*Az)) * 180.0f / PI;

    // Filtro complementar: giroscópio (curto prazo) + acelerômetro (longo prazo)
    gPitch = ALPHA * (gPitch + Gx * dt) + (1.0f - ALPHA) * pitchAcel;
    gRoll  = ALPHA * (gRoll  + Gy * dt) + (1.0f - ALPHA) * rollAcel;
}

/**
 * Converte a leitura bruta do ADC do HW-390 em porcentagem de umidade (0–100%).
 * O sensor capacitivo inverte o sinal: valor ADC maior → solo mais seco.
 */
float adcParaPorcentagem(int adcVal) {
    float pct = (float)(HW390_SECO - adcVal) / (float)(HW390_SECO - HW390_MOLHADO) * 100.0f;
    return constrain(pct, 0.0f, 100.0f);
}

/**
 * Sequência disparada pelo botão A:
 *   1. Salva a inclinação atual (Pitch / Roll) do IMU.
 *   2. Gira o servo 90° até a posição de medição.
 *   3. Lê a umidade do sensor HW-390 e converte para porcentagem.
 *   4. Imprime todos os dados no monitor serial.
 *   5. Retorna o servo à posição de repouso.
 */
void capturarInclinacao() {

    // ── 1. Salva inclinação do IMU ────────────────────────────
    gPitchSalvo      = gPitch;
    gRollSalvo       = gRoll;
    gTemLeituraSalva = true;

    // ── 2. Gira o servo até a posição de medição ──────────────
    gServoPosicao = SERVO_MEDICAO;
    gServo.write(SERVO_MEDICAO);
    delay(SERVO_DELAY_MS);   // Aguarda o servo completar o movimento

    // ── 3. Lê o sensor HW-390 (média de 10 amostras) ─────────
    // Múltiplas leituras reduzem ruído do ADC do ESP32
    long somaADC = 0;
    for (int i = 0; i < 10; i++) {
        somaADC += analogRead(UMIDADE_PIN);
        delay(5);
    }
    gUmidadeADC = (int)(somaADC / 10);

    // ── 4. Imprime tudo no monitor serial ─────────────────────
    Serial.println("\n=====================================");
    Serial.println("  [BOTAO A] Coleta de Dados!");
    Serial.println("  --- Inclinacao (MPU-6050) ---");
    Serial.printf ("  Pitch (frente/tras): %+.2f graus\n", gPitchSalvo);
    Serial.printf ("  Roll  (lateral)    : %+.2f graus\n", gRollSalvo);
    Serial.println("  --- Umidade (HW-390) ---");
    Serial.printf ("  Leitura ADC        : %d\n", gUmidadeADC);
    Serial.printf ("  Porcentagem        : %.1f%%\n", adcParaPorcentagem(gUmidadeADC));
    Serial.println("=====================================");

    // ── 5. Retorna servo ao repouso ───────────────────────────
    delay(300);                    // Pausa breve antes de retrair
    gServoPosicao = SERVO_REPOUSO;
    gServo.write(SERVO_REPOUSO);
}

// ─────────────────────────────────────────────────────────────
//  Funções de controle do motor
// ─────────────────────────────────────────────────────────────

/**
 * Controla a direção de um motor (velocidade sempre máxima).
 *
 * @param pinIn1    Pino de direção positiva (frente)
 * @param pinIn2    Pino de direção negativa (ré)
 * @param valorEixo Valor bruto do eixo Y do joystick (-512..+511)
 */
void controlarMotor(uint8_t pinIn1, uint8_t pinIn2, int valorEixo) {
    if (abs(valorEixo) < DEADZONE) {
        // Dentro da zona morta → motor parado
        digitalWrite(pinIn1, LOW);
        digitalWrite(pinIn2, LOW);
    } else if (valorEixo > 0) {
        // Analógico para frente → motor para frente
        digitalWrite(pinIn1, HIGH);
        digitalWrite(pinIn2, LOW);
    } else {
        // Analógico para trás → motor em ré
        digitalWrite(pinIn1, LOW);
        digitalWrite(pinIn2, HIGH);
    }
}

/**
 * Para ambos os motores imediatamente.
 */
void pararMotores() {
    digitalWrite(MOTOR_A_IN1, LOW);
    digitalWrite(MOTOR_A_IN2, LOW);
    digitalWrite(MOTOR_B_IN3, LOW);
    digitalWrite(MOTOR_B_IN4, LOW);
}

// ─────────────────────────────────────────────────────────────
//  Handlers HTTP
// ─────────────────────────────────────────────────────────────

/**
 * GET /
 * Serve a página HTML da dashboard.
 */
void handleRoot() {
    server.send_P(200, "text/html", DASHBOARD_HTML);
}

/**
 * GET /dados
 * Retorna um JSON com os dados atuais de todos os sensores.
 * Formato:
 * {
 *   "pitch": float,           // ângulo frente/trás em graus
 *   "roll": float,            // ângulo lateral em graus
 *   "umidade_adc": int,       // leitura bruta ADC 0-4095
 *   "umidade_pct": float,     // porcentagem de umidade 0-100
 *   "servo_graus": int,       // posição atual do servo em graus
 *   "gamepad": bool,          // controle Xbox conectado?
 *   "tem_leitura_salva": bool,// existe coleta salva pelo botão A?
 *   "pitch_salvo": float,     // pitch da última coleta
 *   "roll_salvo": float,      // roll da última coleta
 *   "umidade_adc_salvo": int, // ADC da última coleta
 *   "umidade_pct_salvo": float// porcentagem da última coleta
 * }
 */
void handleDados() {
    // Leitura ao vivo da umidade
    int umidadeAoVivo = analogRead(UMIDADE_PIN);
    float umidadePct  = adcParaPorcentagem(umidadeAoVivo);

    String json = "{";
    json += "\"pitch\":"           + String(gPitch, 2)         + ",";
    json += "\"roll\":"            + String(gRoll, 2)          + ",";
    json += "\"umidade_adc\":"     + String(umidadeAoVivo)     + ",";
    json += "\"umidade_pct\":"     + String(umidadePct, 1)     + ",";
    json += "\"servo_graus\":"     + String(gServoPosicao)     + ",";
    json += "\"gamepad\":"         + (gGamepad ? "true" : "false") + ",";
    json += "\"tem_leitura_salva\":" + (gTemLeituraSalva ? "true" : "false") + ",";
    json += "\"pitch_salvo\":"     + String(gPitchSalvo, 2)    + ",";
    json += "\"roll_salvo\":"      + String(gRollSalvo, 2)     + ",";
    json += "\"umidade_adc_salvo\":" + String(gUmidadeADC)     + ",";
    json += "\"umidade_pct_salvo\":" + String(adcParaPorcentagem(gUmidadeADC), 1);
    json += "}";

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
}

/**
 * Handler para rotas não encontradas.
 */
void handleNotFound() {
    server.send(404, "text/plain", "404 - Rota nao encontrada");
}

// ─────────────────────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    Serial.println("\n=== Carrinho de Esteiras - ESP32 ===");

    // ── Configura pinos de direção ────────────────────────────
    pinMode(MOTOR_A_IN1, OUTPUT);
    pinMode(MOTOR_A_IN2, OUTPUT);
    pinMode(MOTOR_B_IN3, OUTPUT);
    pinMode(MOTOR_B_IN4, OUTPUT);

    // ENA e ENB estão ligados diretamente ao 5 V na fiação;
    // nenhuma configuração de software necessária.

    // Garante motores parados na inicialização
    pararMotores();

    // ── Inicializa MPU-6050 via I²C ───────────────────────────
    Wire.begin();
    delay(250);                    // aguarda barramento I2C estabilizar

    mpu.reset();                   // reinicia todos os registradores internos
    delay(100);                    // aguarda reset completar

    mpu.initialize();              // configura DLPF, clock e acorda o sensor
    mpu.setSleepEnabled(false);    // acorda explicitamente (sleep mode = off)
    mpu.setClockSource(MPU6050_CLOCK_PLL_XGYRO);  // clock estável via giroscópio X
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);   // ±250 °/s
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);   // ±2 g
    delay(100);                    // aguarda sensor estabilizar após config

    if (!mpu.testConnection()) {
        Serial.println("ERRO: MPU-6050 nao encontrado!");
    } else {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        Serial.println("--- MPU-6050 OK ---");
        Serial.printf("  AX=%d AY=%d AZ=%d  GX=%d GY=%d GZ=%d\n",
                      ax, ay, az, gx, gy, gz);
        Serial.println("-------------------");
    }
    gTempoAnterior = millis();

    // ── Inicializa Micro Servo ────────────────────────────────
    // ESP32Servo usa canais LEDC internamente (não conflita com L298N)
    gServo.setPeriodHertz(50);
    gServo.attach(SERVO_PIN, 500, 2400);
    gServo.write(SERVO_REPOUSO);
    gServoPosicao = SERVO_REPOUSO;

    // ── Configura pino do HW-390 ──────────────────────────────
    pinMode(UMIDADE_PIN, INPUT);

    // ── Inicializa Bluepad32 ──────────────────────────────────
    BP32.setup(&onConnectedGamepad, &onDisconnectedGamepad);
    BP32.enableNewBluetoothConnections(true);

    // ── Conecta ao Wi-Fi ──────────────────────────────────────
    Serial.printf("\nConectando ao Wi-Fi: %s ", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int tentativas = 0;
    while (WiFi.status() != WL_CONNECTED && tentativas < 30) {
        delay(500);
        Serial.print(".");
        tentativas++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Conectado!");
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
        Serial.println("[WiFi] Acesse a dashboard em: http://" + WiFi.localIP().toString());
    } else {
        Serial.println("\n[WiFi] FALHA na conexao. Verifique SSID e senha.");
        Serial.println("[WiFi] O carrinho continuara funcionando sem Wi-Fi.");
    }

    // ── Registra rotas e inicia servidor HTTP ─────────────────
    server.on("/",      HTTP_GET, handleRoot);
    server.on("/dados", HTTP_GET, handleDados);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("[HTTP] Servidor iniciado na porta 80.");
}

// ─────────────────────────────────────────────────────────────
//  Loop Principal
// ─────────────────────────────────────────────────────────────

void loop() {
    // ── Processa requisições HTTP ─────────────────────────────
    // Deve ser chamado o mais frequentemente possível para
    // responder clientes sem bloquear o restante da lógica.
    server.handleClient();

    // ── Atualiza IMU (sempre, independente do controle) ──────
    atualizarIMU();

    BP32.update();

    if (gGamepad && gGamepad->isConnected()) {

        // ── Leitura dos analógicos ────────────────────────────
        // axisY()  → analógico esquerdo Y (-512 = cima, +511 = baixo)
        // axisRY() → analógico direito  Y
        // Invertemos o sinal: empurrar para cima = valor positivo = frente
        int motorEsquerdo = -(gGamepad->axisY());
        int motorDireito  = -(gGamepad->axisRY());

        // ── Botão A → captura e imprime inclinação ────────────
        // Detecção de borda de subida: só dispara uma vez por pressão
        bool botaoAAtual = gGamepad->a();
        if (botaoAAtual && !gBotaoAAnterior) {
            capturarInclinacao();
        }
        gBotaoAAnterior = botaoAAtual;

        // ── Botão B → parada de emergência ───────────────────
        if (gGamepad->b()) {
            pararMotores();
            delay(10);
            return;
        }

        // ── Aplica direção nas pontes H ───────────────────────
        controlarMotor(MOTOR_A_IN1, MOTOR_A_IN2, motorEsquerdo);
        controlarMotor(MOTOR_B_IN3, MOTOR_B_IN4, motorDireito);

    } else {
        // Sem controle conectado → motores parados por segurança
        pararMotores();
    }

    delay(10);
}
