# 🤖 Calango-Tech

Controle de um carrinho com esteiras usando **Bluetooth Classic** do ESP32 e um **controle Xbox One S**, com ponte H **L298N**.

---

## 📦 Bibliotecas Necessárias

### 1. Bluepad32 (controle Xbox via Bluetooth)

> **Atenção:** O Xbox One S usa **Bluetooth Classic (BR/EDR)**, não BLE.
> A biblioteca correta é o **Bluepad32**.

**Passo a passo no Arduino IDE:**

1. Vá em `Sketch` → `Include Library` → `Manage Libraries...`
2. Pesquise: `Bluepad32`
3. Instale a versão do autor **Ricardo Quesada**

> Alternativa: https://github.com/ricardoquesada/bluepad32

### 2. Suporte ao ESP32 no Arduino IDE

1. Vá em `File` → `Preferences`
2. Em "Additional boards manager URLs", adicione:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Vá em `Tools` → `Board` → `Boards Manager`
4. Pesquise `esp32` e instale o pacote da **Espressif Systems**

**Configuração de upload:**
- Board: `ESP32 Dev Module`
- Upload Speed: `115200`
- Flash Size: `4MB`

---

## 🔌 Diagrama de Pinagem

```
ESP32 DevKit V1          L298N
─────────────────        ──────────────────────────
GPIO 14 (PWM)    ──────► ENA   (Motor A — velocidade)
GPIO 27          ──────► IN1   (Motor A — direção +)
GPIO 26          ──────► IN2   (Motor A — direção -)

GPIO 32 (PWM)    ──────► ENB   (Motor B — velocidade)
GPIO 25          ──────► IN3   (Motor B — direção +)
GPIO 33          ──────► IN4   (Motor B — direção -)

GND              ──────► GND   (terra comum)

                         12V ◄── Bateria (7.4V a 12V)
                         5V  ──► VIN do ESP32 (opcional)
```

---

## 🎮 Esquema de Controle (Tank Drive)

| Analógico | Motor Controlado | Direção |
|---|---|---|
| Esquerdo ↑ | Motor Esquerdo | Frente |
| Esquerdo ↓ | Motor Esquerdo | Ré |
| Direito ↑ | Motor Direito | Frente |
| Direito ↓ | Motor Direito | Ré |
| Botão B | Ambos | Parada de emergência |

**Manobras:**
- Ambos pra frente → andar reto
- Ambos pra ré → ré
- Esquerdo frente + Direito parado → curva direita
- Esquerdo frente + Direito pra ré → giro no eixo

---

## 📡 Como Parear o Controle Xbox

1. Carregue o código no ESP32 e abra o Monitor Serial (115200 baud)
2. No controle Xbox One S:
   - Ligue pressionando o botão Xbox (logo no centro)
   - Pressione e segure o **botão de pareamento** (pequeno botão na parte superior, acima do conector USB)
   - O LED piscará rapidamente
3. No Monitor Serial você verá a mensagem de conexão em alguns segundos

Após o primeiro pareamento, o controle reconecta automaticamente.

---

## 🔧 Ajustes no Código

**Alterar pinos:** edite as `#define` no topo do `.ino`

**Zona morta do joystick:**
`#define DEADZONE  40` — aumente se o carrinho se mover sem tocar no analógico

**Frequência PWM:**
`#define PWM_FREQ  1000` — valores entre 500 e 5000 Hz funcionam bem

---

## ⚡ Fonte de Alimentação

| Componente | Tensão | Corrente |
|---|---|---|
| Motores + L298N | 7.4V a 12V | 1A a 3A por motor |
| ESP32 | 5V via VIN | ~240mA |

O L298N tem saída de 5V regulada que pode alimentar o ESP32 pelo pino VIN (para motores menores). Para motores mais potentes, use fontes separadas.

---

## 🐛 Troubleshooting

| Problema | Causa provável | Solução |
|---|---|---|
| Controle não conecta | LED não piscando rápido | Pressione o botão de pareamento |
| Motor gira em uma direção só | IN1/IN2 invertidos | Troque os fios do motor |
| Motor treme sem mover | PWM muito baixo | Aumente DEADZONE ou a tensão |
| ESP32 reinicia ao ligar motor | Pico de corrente | Adicione capacitor 100µF na bateria |
| Serial sem output | Baud rate errado | Ajuste para 115200 |
