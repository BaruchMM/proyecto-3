// TinyML 2026 | Reconocimiento de actividad fisica | modelo HAR_acc_gir_int8
// Placa: Arduino Nano 33 BLE Sense (Rev1, IMU LSM9DS1) o Sense Rev2 (IMU BMI270).
// Bibliotecas: Chirale_TensorFlowLite 2.0.0 y Arduino_LSM9DS1 (Rev1) o Arduino_BMI270_BMM150 (Rev2).
// Generado por Proyecto_3.py junto con model.h y har_config.h. Todo se muestra en el Monitor Serie.
//
// Al encender: autoprueba con ventanas del dataset (debe dar lo mismo que Python) y
// cronometraje de 100 inferencias. Despues: lee la IMU a 50 Hz y clasifica cada 0.5 s
// la ultima ventana de 2 s.
#include <Arduino.h>
#include <math.h>
#include <stdint.h>

// 0 = Nano 33 BLE Sense (Rev1, LSM9DS1); 1 = Nano 33 BLE Sense Rev2 (BMI270).
// Debe ser la misma revision de placa con la que se grabo el dataset.
#ifndef SENSE_REV2
#define SENSE_REV2 0
#endif

#if SENSE_REV2
#include <Arduino_BMI270_BMM150.h>
#else
#include <Arduino_LSM9DS1.h>
#endif

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "har_config.h"

// Poner en 0 si la biblioteca no tiene arena_used_bytes().
#define USE_ARENA_USED 1

constexpr int kTensorArenaSize = 32768;
constexpr int kNumInferencias = 100;   // inferencias cronometradas en setup()
constexpr int kSuavizado = 3;          // promedio de las ultimas probabilidades
constexpr uint32_t kResumenCada = 40;     // repetir el resumen cada 40 predicciones (20 s)
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* net = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Buffer circular con las 6 columnas crudas, en el orden del CSV
float buffer_imu[kVentana][6];
float ventana[kVentana][6];
int pos_buffer = 0;
int muestras_en_buffer = 0;
int muestras_desde_inferencia = 0;
uint32_t proxima_muestra_us = 0;
uint32_t muestras_tarde = 0;
float ultima_acc[3] = {0.0f, 0.0f, 0.0f};
float ultimo_gir[3] = {0.0f, 0.0f, 0.0f};

float historial[kSuavizado][kNumClases];
int n_historial = 0;
int pos_historial = 0;
uint32_t n_predicciones = 0;

// Resultados de setup() que se repiten en loop() por si el monitor se abrio tarde
int autoprueba_ok = 0;
float autoprueba_dif_max = 0.0f;
uint32_t t_min = 0, t_max = 0;
float t_prom = 0.0f, t_desv = 0.0f, prep_prom = 0.0f;

void fatal(const char* mensaje) {
  // Se repite porque el monitor serie puede abrirse despues del error
  while (true) {
    Serial.print("[error] modelo=HAR_acc_gir_int8 ");
    Serial.println(mensaje);
    delay(2000);
  }
}

void leer_imu(float* fila) {
  float x, y, z;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
    ultima_acc[0] = x; ultima_acc[1] = y; ultima_acc[2] = z;
  }
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(x, y, z);
    ultimo_gir[0] = x; ultimo_gir[1] = y; ultimo_gir[2] = z;
  }
  for (int j = 0; j < 3; j++) {
    fila[j] = ultima_acc[j];
    fila[3 + j] = ultimo_gir[j];
  }
}

// Ventana cruda -> tensor de entrada. Repite preprocesar() de Proyecto_3.py:
// restar la media de la ventana a las columnas de kRestarMedia, elegir canales,
// (x - kMedia) / kDesv y cuantizar.
void preparar_entrada(const float v[kVentana][6]) {
  float media_ventana[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < 6; j++) {
    if (!kRestarMedia[j]) continue;
    double suma = 0.0;
    for (int t = 0; t < kVentana; t++) suma += (double)v[t][j];
    media_ventana[j] = (float)(suma / (double)kVentana);
  }
  const bool entero = input_tensor->type == kTfLiteInt8;
  const float scale = input_tensor->params.scale;
  const int zero = input_tensor->params.zero_point;
  for (int t = 0; t < kVentana; t++) {
    for (int c = 0; c < kCanales; c++) {
      const int col = kColumna[c];
      float x = v[t][col];
      if (kRestarMedia[col]) x = x - media_ventana[col];
      x = (x - kMedia[c]) / kDesv[c];
      const int i = t * kCanales + c;
      if (entero) {
        // q = round(x / scale) + zero_point, saturada a [-128, 127]
        int q = (int)lroundf(x / scale) + zero;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input_tensor->data.int8[i] = (int8_t)q;
      } else {
        input_tensor->data.f[i] = x;
      }
    }
  }
}

void leer_salida(float* p) {
  for (int k = 0; k < kNumClases; k++) {
    if (output_tensor->type == kTfLiteInt8) {
      p[k] = (float)(output_tensor->data.int8[k] - output_tensor->params.zero_point)
             * output_tensor->params.scale;
    } else {
      p[k] = output_tensor->data.f[k];
    }
  }
}

int argmax(const float* p) {
  int mejor = 0;
  for (int k = 1; k < kNumClases; k++) if (p[k] > p[mejor]) mejor = k;
  return mejor;
}

void autoprueba() {
  autoprueba_ok = 0;
  autoprueba_dif_max = 0.0f;
  float p[kNumClases];
  for (int i = 0; i < kNumPruebas; i++) {
    preparar_entrada(kPruebaX[i]);
    if (interpreter->Invoke() != kTfLiteOk) fatal("ERROR: fallo de inferencia en la autoprueba.");
    leer_salida(p);
    float dif = 0.0f;
    for (int k = 0; k < kNumClases; k++) {
      float d = fabsf(p[k] - kPruebaEsperada[i][k]);
      if (d > dif) dif = d;
    }
    if (dif > autoprueba_dif_max) autoprueba_dif_max = dif;
    const int pred = argmax(p);
    const int pred_py = argmax(kPruebaEsperada[i]);
    if (pred == pred_py && dif < 0.02f) autoprueba_ok++;
    Serial.print("[autoprueba] i="); Serial.print(i);
    Serial.print(" persona="); Serial.print(kPruebaPersona[i]);
    Serial.print(" real="); Serial.print(kClases[kPruebaClase[i]]);
    Serial.print(" placa="); Serial.print(kClases[pred]);
    Serial.print(" python="); Serial.print(kClases[pred_py]);
    Serial.print(" p="); Serial.print(p[pred], 4);
    Serial.print(" dif_max="); Serial.println(dif, 5);
  }
}

void cronometrar() {
  static uint32_t tiempos[kNumInferencias];
  uint32_t suma_prep = 0;
  for (int i = 0; i < kNumInferencias; i++) {
    uint32_t t0 = micros();
    preparar_entrada(kPruebaX[i % kNumPruebas]);
    uint32_t t1 = micros();
    TfLiteStatus st = interpreter->Invoke();
    uint32_t t2 = micros();
    if (st != kTfLiteOk) fatal("ERROR: fallo de inferencia al cronometrar.");
    suma_prep += t1 - t0;
    tiempos[i] = t2 - t1;
  }
  t_min = tiempos[0];
  t_max = tiempos[0];
  uint64_t suma = 0;
  for (int i = 0; i < kNumInferencias; i++) {
    if (tiempos[i] < t_min) t_min = tiempos[i];
    if (tiempos[i] > t_max) t_max = tiempos[i];
    suma += tiempos[i];
  }
  t_prom = (float)suma / (float)kNumInferencias;
  float var = 0.0f;
  for (int i = 0; i < kNumInferencias; i++) {
    float d = (float)tiempos[i] - t_prom;
    var += d * d;
  }
  t_desv = sqrtf(var / (float)kNumInferencias);
  prep_prom = (float)suma_prep / (float)kNumInferencias;
}

void imprimir_resumen() {
  Serial.print("[autoprueba] modelo=HAR_acc_gir_int8 coinciden="); Serial.print(autoprueba_ok);
  Serial.print("/"); Serial.print(kNumPruebas);
  Serial.print(" dif_max="); Serial.println(autoprueba_dif_max, 5);
  Serial.print("[tiempo] modelo=HAR_acc_gir_int8 n="); Serial.print(kNumInferencias);
  Serial.print(" min_us="); Serial.print(t_min);
  Serial.print(" max_us="); Serial.print(t_max);
  Serial.print(" prom_us="); Serial.print(t_prom, 1);
  Serial.print(" desv_us="); Serial.print(t_desv, 1);
  Serial.print(" prep_prom_us="); Serial.print(prep_prom, 1);
#if USE_ARENA_USED
  Serial.print(" arena_usada="); Serial.print((unsigned long)interpreter->arena_used_bytes());
#endif
  Serial.print(" arena="); Serial.println(kTensorArenaSize);
}

void clasificar() {
  // Ventana en orden temporal: pos_buffer apunta a la muestra mas antigua
  for (int t = 0; t < kVentana; t++) {
    const int idx = (pos_buffer + t) % kVentana;
    for (int j = 0; j < 6; j++) ventana[t][j] = buffer_imu[idx][j];
  }
  preparar_entrada(ventana);
  uint32_t t0 = micros();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t t_inf = micros() - t0;
  if (st != kTfLiteOk) fatal("ERROR: fallo de inferencia.");

  float p[kNumClases];
  leer_salida(p);
  for (int k = 0; k < kNumClases; k++) historial[pos_historial][k] = p[k];
  pos_historial = (pos_historial + 1) % kSuavizado;
  if (n_historial < kSuavizado) n_historial++;
  float prom[kNumClases];
  for (int k = 0; k < kNumClases; k++) {
    float s = 0.0f;
    for (int h = 0; h < n_historial; h++) s += historial[h][k];
    prom[k] = s / (float)n_historial;
  }
  const int suav = argmax(prom);
  const int inst = argmax(p);

  Serial.print("[pred] "); Serial.print(kClases[suav]);
  Serial.print(" p="); Serial.print(prom[suav], 2);
  Serial.print(" | ventana: "); Serial.print(kClases[inst]);
  Serial.print(" p="); Serial.print(p[inst], 2);
  Serial.print(" | t_inf_us="); Serial.print(t_inf);
  Serial.print(" muestras_tarde="); Serial.println(muestras_tarde);

  n_predicciones++;
  if (n_predicciones % kResumenCada == 0) imprimir_resumen();
}

void setup() {
  Serial.begin(115200);
  unsigned long inicio = millis();
  while (!Serial && millis() - inicio < 5000) { delay(10); }

  if (!IMU.begin()) {
    fatal("IMU.begin() fallo. Revisar SENSE_REV2: 0 = Sense Rev1 (LSM9DS1), 1 = Sense Rev2 (BMI270).");
  }

  net = tflite::GetModel(g_model);
  if (net->version() != TFLITE_SCHEMA_VERSION) {
    fatal("ERROR: version de schema incompatible.");
  }

  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddExpandDims();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddSoftmax();
  static tflite::MicroInterpreter static_interpreter(net, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fatal("ERROR: AllocateTensors. Revisar arena, operadores o versiones.");
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  if (input_tensor->bytes != (size_t)kVentana * kCanales * (input_tensor->type == kTfLiteInt8 ? 1 : 4)) {
    fatal("ERROR: la entrada del modelo no coincide con har_config.h.");
  }

  Serial.println();
  Serial.println("modelo: HAR_acc_gir_int8");
  Serial.print("tipo entrada (1=float32, 9=int8): "); Serial.println((int)input_tensor->type);
  Serial.print("scale/zero entrada: "); Serial.print(input_tensor->params.scale, 8);
  Serial.print(" / "); Serial.println(input_tensor->params.zero_point);
  Serial.print("scale/zero salida: "); Serial.print(output_tensor->params.scale, 8);
  Serial.print(" / "); Serial.println(output_tensor->params.zero_point);
  Serial.print("IMU: acelerometro "); Serial.print(IMU.accelerationSampleRate());
  Serial.print(" Hz, giroscopio "); Serial.print(IMU.gyroscopeSampleRate()); Serial.println(" Hz");

  autoprueba();
  cronometrar();
  imprimir_resumen();
  Serial.println("Clasificando en tiempo real (primera prediccion en 2 s)...");
  proxima_muestra_us = micros();
}

void loop() {
  const uint32_t ahora = micros();
  if ((int32_t)(ahora - proxima_muestra_us) < 0) return;

  // Si la inferencia o el puerto serie retrasaron la lectura, se cuenta la muestra como tarde.
  const uint32_t retraso = ahora - proxima_muestra_us;
  if (retraso >= kPeriodoUs) muestras_tarde++;
  if (retraso >= 5 * kPeriodoUs) proxima_muestra_us = ahora;   // resincronizar
  proxima_muestra_us += kPeriodoUs;

  leer_imu(buffer_imu[pos_buffer]);
  pos_buffer = (pos_buffer + 1) % kVentana;
  if (muestras_en_buffer < kVentana) muestras_en_buffer++;
  muestras_desde_inferencia++;

  if (muestras_en_buffer == kVentana && muestras_desde_inferencia >= kPaso) {
    muestras_desde_inferencia = 0;
    clasificar();
  }
}
