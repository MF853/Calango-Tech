/**
 * ============================================================
 *  Carrinho de Esteiras - ESP32 + L298N + Xbox One S
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
 *
 *  Esquema de Controle (Tank Drive — velocidade sempre máxima):
 *    - Analógico Esquerdo Y > zona morta → Motor A para frente
 *    - Analógico Esquerdo Y < zona morta → Motor A para ré
 *    - Analógico Direito  Y > zona morta → Motor B para frente
 *    - Analógico Direito  Y < zona morta → Motor B para ré
 *    - Botão B → parada de emergência
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

// Objeto do servo
Servo gServo;

// ── Ponteiro global para o gamepad conectado ──────────────────
GamepadPtr gGamepad = nullptr;

// ─────────────────────────────────────────────────────────────
//  Callbacks do Bluepad32
// ─────────────────────────────────────────────────────────────

void onConnectedGamepad(GamepadPtr gp) {
    gGamepad = gp;
}

void onDisconnectedGamepad(GamepadPtr gp) {
    gGamepad = nullptr;
    pararMotores();   // segurança: para tudo ao desconectar
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
    Serial.println("=====================================");

    // ── 5. Retorna servo ao repouso ───────────────────────────
    delay(300);                    // Pausa breve antes de retrair
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
        // Diagnóstico: imprime leitura bruta única para confirmar sensor ativo
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

    // ── Configura pino do HW-390 ──────────────────────────────
    pinMode(UMIDADE_PIN, INPUT);

    // ── Inicializa Bluepad32 ──────────────────────────────────
    BP32.setup(&onConnectedGamepad, &onDisconnectedGamepad);
    BP32.enableNewBluetoothConnections(true);
}

// ─────────────────────────────────────────────────────────────
//  Loop Principal
// ─────────────────────────────────────────────────────────────

void loop() {
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
