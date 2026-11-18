#%% [markdown]
# # TinyML 2026, proyecto 3: reconocimiento de actividad fisica (HAR) en Arduino Nano 33 BLE Sense
#
# Entrada: datos/*.csv con columnas accion, persona, t_us, ax, ay, az (g), gx, gy, gz (grados/s),
# muestreadas a 50 Hz. Una sesion continua de 60 s por persona y actividad.
#
# Los bloques 0.x preparan lo compartido. Cada bloque numerado corresponde a un apartado del
# reporte y guarda sus tablas y figuras en resultados/<apartado>_<tema>/.
# Los sketches de Arduino se generan en arduino/.
#
# Evaluacion: se deja una persona fuera (LOSO). Se entrena con dos personas y se prueba con la
# tercera, rotando. Ninguna ventana de una sesion de prueba se usa para entrenar ni validar.

#%% 0.1 Configuracion
import contextlib, csv, io, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Avisos conocidos que no afectan los resultados: tf.lite.Interpreter esta deprecado (funciona
# en TF 2.21) y el conversor avisa que no recibio estadisticas de la entrada int8.
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')
warnings.filterwarnings('ignore', message='.*Statistics for quantized inputs.*')

SEED = 2026
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Rutas
try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # celdas interactivas
    ROOT = Path.cwd()
DATOS = ROOT / 'datos'
OUTPUT = ROOT / 'resultados'
ARDUINO = ROOT / 'arduino'

APARTADOS = {
    1: '1_datos',
    2: '2_preprocesamiento',
    3: '3_modelo',
    4: '4_clasificacion',
    5: '5_sensores',
    6: '6_cuantizacion',
    7: '7_arduino',
}

# Datos
FS_HZ = 50
CLASES = ['reposo', 'caminar', 'correr', 'subir_escaleras', 'bajar_escaleras']
ETIQUETAS = ['reposo', 'caminar', 'correr', 'subir', 'bajar']   # para figuras
COLUMNAS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']                 # orden fijo, igual que en la placa
PERSONAS = [1, 2, 3]
RECORTE_MUESTRAS = 50    # se descarta 1 s al inicio y al final de cada sesion

# Ventanas
VENTANA = 100            # 2 s a 50 Hz
PASO = 25                # 0.5 s (75 % de traslape); la placa clasifica con el mismo paso
FRACCION_VAL = 0.2       # ultimo 20 % de cada sesion de entrenamiento, como bloque temporal aparte
# Columnas (indices de COLUMNAS) a las que se resta su media dentro de cada ventana. En el
# acelerometro la media es sobre todo la gravedad, que depende de como se coloco la placa;
# en el giroscopo es el giro sostenido (vueltas en los descansos de la escalera) y el sesgo.
RESTAR_MEDIA = [0, 1, 2, 3, 4, 5]

SENSORES = {
    'acc': [0, 1, 2],                  # indices dentro de COLUMNAS
    'acc_gir': [0, 1, 2, 3, 4, 5],
}
NOMBRE_SENSOR = {'acc': 'acelerometro', 'acc_gir': 'acelerometro + giroscopo'}
# Modelo de la demostracion. En la exploracion LOSO previa (2 semillas) el acelerometro solo
# generalizo mejor (0.82) que acelerometro + giroscopo (0.78); el apartado 5 los compara.
SENSOR_PRINCIPAL = 'acc'

# Aumento de datos (solo en entrenamiento): copias con una rotacion 3D pequena y aleatoria
# aplicada igual a la aceleracion y a la velocidad angular. Imita diferencias de colocacion.
USAR_AUMENTO = False
AUMENTO_COPIAS = 2
AUMENTO_ROT_GRADOS = 20.0

# Modelo y entrenamiento
# Capas convolucionales: (filtros, kernel, stride, max-pooling despues). Con pooling, la ultima
# convolucion ve 59 muestras (1.2 s), mas que un ciclo de paso.
CAPAS_CONV = [
    (16, 5, 2, 2),
    (32, 5, 1, 2),
    (32, 5, 1, 0),
]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 10
SEMILLAS_LOSO = [1, 2, 3]    # cada pliegue LOSO se entrena con estas semillas
N_REP = 300                  # ventanas de entrenamiento para calibrar INT8
SUAVIZADO = 3                # la placa promedia las probabilidades de las ultimas 3 ventanas

# Ablacion (apartado 2), con acelerometro + giroscopo y FP32:
# (nombre para la figura, columnas a las que se resta la media, aumento, capas convolucionales)
CORRER_ABLACION = True
SENSOR_ABLACION = 'acc_gir'
CAPAS_CONV_CORTA = [(16, 5, 2, 0), (32, 5, 2, 0)]   # sin pooling: la ultima convolucion ve 13 muestras
ABLACION = [
    ('sin restar\nmedia', [], False, CAPAS_CONV),
    ('restar media\nsolo giroscopo', [3, 4, 5], False, CAPAS_CONV),
    ('restar media\n6 canales\ncampo de 13', [0, 1, 2, 3, 4, 5], False, CAPAS_CONV_CORTA),
    ('restar media\n6 canales', [0, 1, 2, 3, 4, 5], False, CAPAS_CONV),
    ('restar media\n6 canales\n+ rotaciones', [0, 1, 2, 3, 4, 5], True, CAPAS_CONV),
]

# Arduino
# El kernel FULLY_CONNECTED int8 de Chirale 2.0.0 usa una sola escala de pesos por capa
# (hallazgo del proyecto 1). Las convoluciones si aceptan una escala por canal.
DENSE_POR_CANAL = False
TENSOR_ARENA_BYTES = 32 * 1024
SENSE_REV2 = 0               # 0 = Sense Rev1 (LSM9DS1), 1 = Sense Rev2 (BMI270). Igual que al grabar.
N_TIMING_RUNS = 100
SKETCHES = [                 # (sensores, modo); el primero es el de la demostracion
    (SENSOR_PRINCIPAL, 'int8'),
    (SENSOR_PRINCIPAL, 'fp32'),
] + [(s, m) for s in ('acc', 'acc_gir') if s != SENSOR_PRINCIPAL for m in ('int8', 'fp32')]
PENDIENTE = 'pendiente'

# Figuras (misma paleta que el proyecto 1)
MOSTRAR_FIGURAS = 'ipykernel' in sys.modules
DPI_FIG = 200
SUPERFICIE = '#ffffff'
TINTA = '#0b0b0b'
TINTA_2 = '#52514e'
GRIS = '#898781'
REJILLA = '#e1e0d9'
EJE = '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
COLOR_EJE = dict(zip(['x', 'y', 'z'], SERIES[:3]))
COLOR_SENSOR = {'acc': SERIES[1], 'acc_gir': SERIES[0]}
COLOR_FP32 = GRIS
COLOR_INT8 = SERIES[0]
RAMPA_AZUL = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
CMAP_CONF = LinearSegmentedColormap.from_list('azul', RAMPA_AZUL)

plt.rcParams.update({
    'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE, 'savefig.facecolor': SUPERFICIE,
    'axes.edgecolor': EJE, 'axes.labelcolor': TINTA_2, 'axes.titlecolor': TINTA,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
    'grid.color': REJILLA, 'grid.linewidth': 0.6,
    'xtick.color': TINTA_2, 'ytick.color': TINTA_2,
    'legend.frameon': False, 'lines.linewidth': 1.2,
})

T_INICIO = time.time()
print('Python:', sys.version.split()[0])
print('TensorFlow:', tf.__version__)
print('NumPy:', np.__version__)
print('pandas:', pd.__version__)
print('Archivos en:', OUTPUT)


#%% 0.2 Funciones
def carpeta(n, sub=None):
    d = OUTPUT / APARTADOS[n]
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def guardar_tab(n, nombre, encabezado, filas):
    ruta = carpeta(n) / f'{n}_tab_{nombre}.csv'
    with open(ruta, 'w', newline='', encoding='ascii') as f:
        w = csv.writer(f)
        w.writerow(encabezado)
        w.writerows(filas)
    print('Tabla:', ruta.relative_to(ROOT))
    return ruta


def guardar_fig(n, nombre, fig):
    ruta = carpeta(n) / f'{n}_fig_{nombre}.png'
    fig.savefig(ruta, dpi=DPI_FIG, bbox_inches='tight')
    print('Figura:', ruta.relative_to(ROOT))
    if MOSTRAR_FIGURAS:
        plt.show()
    plt.close(fig)


def guardar_modelo(n, nombre, blob):
    ruta = carpeta(n, 'modelos') / f'{nombre}.tflite'
    ruta.write_bytes(blob)
    return ruta


def r4(v):
    return round(float(v), 4)


# --- Datos y ventanas -------------------------------------------------------------------------
def cargar_datos():
    archivos = sorted(DATOS.glob('*.csv'))
    if not archivos:
        raise FileNotFoundError(f'No hay CSV en {DATOS}')
    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    faltan = set(['accion', 'persona', 't_us'] + COLUMNAS) - set(df.columns)
    if faltan:
        raise ValueError(f'Faltan columnas: {faltan}')
    desconocidas = set(df['accion']) - set(CLASES)
    if desconocidas:
        raise ValueError(f'Acciones no esperadas: {desconocidas}')
    return df


def ventanas(x, paso=PASO):
    inicios = range(0, len(x) - VENTANA + 1, paso)
    return np.stack([x[i:i + VENTANA] for i in inicios]).astype(np.float32)


def armar(personas, parte):
    """Ventanas crudas (n, VENTANA, 6), etiquetas y persona de cada ventana.

    parte = 'todo' usa la sesion completa; 'train' y 'val' la parten en dos bloques temporales
    (primero 80 %, ultimo 20 %) y cada bloque se ventanea por separado, asi que ninguna ventana
    de validacion comparte muestras con una de entrenamiento.
    """
    V, y, quien = [], [], []
    for p in personas:
        for c in range(len(CLASES)):
            s = SESIONES[(p, c)]
            corte = int(round(len(s) * (1 - FRACCION_VAL)))
            seg = {'todo': s, 'train': s[:corte], 'val': s[corte:]}[parte]
            v = ventanas(seg)
            V.append(v)
            y.append(np.full(len(v), c))
            quien.append(np.full(len(v), p))
    return np.concatenate(V), np.concatenate(y), np.concatenate(quien)


def rotacion_aleatoria(rng, max_grados):
    eje = rng.normal(size=3)
    eje /= np.linalg.norm(eje)
    ang = np.deg2rad(rng.uniform(-max_grados, max_grados))
    K = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]])
    return (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K).astype(np.float32)


def aumentar(V, y, rng, copias=AUMENTO_COPIAS, max_grados=AUMENTO_ROT_GRADOS):
    Vs, ys = [V], [y]
    for _ in range(copias):
        Va = V.copy()
        for i in range(len(Va)):
            R = rotacion_aleatoria(rng, max_grados)
            Va[i, :, 0:3] = Va[i, :, 0:3] @ R.T
            Va[i, :, 3:6] = Va[i, :, 3:6] @ R.T
        Vs.append(Va)
        ys.append(y)
    return np.concatenate(Vs), np.concatenate(ys)


def preprocesar(V, canales, mu=None, sd=None, restar_media=RESTAR_MEDIA):
    """Ventanas crudas -> entrada del modelo. preparar_entrada() del sketch repite estos pasos.

    1. A cada columna de restar_media se le resta su media en la ventana (calculada en float64).
    2. Elegir los canales del modelo.
    3. Normalizar cada canal con media y desviacion del conjunto de entrenamiento: (x - mu) / sd.
    Las restas y divisiones se hacen en float32, como en el microcontrolador.
    """
    X = V.astype(np.float32).copy()
    for col in restar_media:
        media = X[:, :, col].astype(np.float64).mean(axis=1, keepdims=True).astype(np.float32)
        X[:, :, col] = X[:, :, col] - media
    X = X[:, :, canales]
    if mu is None:
        plano = X.reshape(-1, X.shape[2]).astype(np.float64)
        mu, sd = plano.mean(axis=0).astype(np.float32), plano.std(axis=0).astype(np.float32)
    return ((X - mu) / sd).astype(np.float32), mu, sd


# --- Modelo ---------------------------------------------------------------------------------
def construir_modelo(n_canales, nombre='har', capas_conv=None):
    capas = [tf.keras.Input((VENTANA, n_canales))]
    for i, (filtros, kernel, stride, pool) in enumerate(capas_conv or CAPAS_CONV, start=1):
        capas.append(tf.keras.layers.Conv1D(filtros, kernel, strides=stride, activation='relu', name=f'conv{i}'))
        if pool:
            capas.append(tf.keras.layers.MaxPooling1D(pool, name=f'maxpool{i}'))
    capas += [
        tf.keras.layers.GlobalAveragePooling1D(name='promedio_global'),
        tf.keras.layers.Dropout(DROPOUT, name='dropout'),
        tf.keras.layers.Dense(len(CLASES), activation='softmax', name='salida'),
    ]
    return tf.keras.Sequential(capas, name=nombre)


def descripcion_arquitectura():
    partes = []
    for filtros, kernel, stride, pool in CAPAS_CONV:
        partes.append(f'Conv1D({filtros}, k={kernel}, s={stride}, relu)')
        if pool:
            partes.append(f'MaxPool({pool})')
    return ' - '.join(partes + ['promedio global', f'Dropout({DROPOUT})', f'Dense({len(CLASES)}, softmax)'])


def campo_receptivo(capas_conv=None):
    rf, salto = 1, 1
    for filtros, kernel, stride, pool in capas_conv or CAPAS_CONV:
        rf += (kernel - 1) * salto
        salto *= stride
        if pool:
            rf += (pool - 1) * salto
            salto *= pool
    return rf


def entrenar(Xtr, ytr, Xva, yva, semilla, nombre='har', capas_conv=None):
    tf.keras.utils.set_random_seed(semilla)
    model = construir_modelo(Xtr.shape[2], nombre, capas_conv)
    model.compile(tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                          restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                     batch_size=BATCH_SIZE, shuffle=True, verbose=0, callbacks=[es])
    return model, hist


def predecir_keras(model, X):
    # Llamada directa: model.predict() reconstruye su funcion en cada modelo nuevo
    return np.concatenate([model(X[i:i + 512], training=False).numpy() for i in range(0, len(X), 512)])


def macs_modelo(model):
    """Multiplicaciones-acumulaciones por inferencia (convoluciones y densa)."""
    filas, total = [], 0
    for capa in model.layers:
        params = capa.count_params()
        forma = tuple(capa.output.shape[1:])
        if isinstance(capa, tf.keras.layers.Conv1D):
            k, cin, cout = capa.kernel.shape
            macs = forma[0] * k * cin * cout
        elif isinstance(capa, tf.keras.layers.Dense):
            macs = capa.kernel.shape[0] * capa.kernel.shape[1]
        else:
            macs = 0
        total += macs
        filas.append([capa.name, type(capa).__name__, 'x'.join(str(v) for v in forma), params, macs])
    return filas, total


# --- TFLite ---------------------------------------------------------------------------------
def convertir(model, modo, X_rep=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if modo == 'int8':
        def rep():
            for i in range(len(X_rep)):
                yield [X_rep[i:i + 1]]
        converter._experimental_disable_per_channel_quantization_for_dense_layers = not DENSE_POR_CANAL
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif modo != 'fp32':
        raise ValueError(modo)
    with contextlib.redirect_stdout(io.StringIO()):
        return converter.convert()


def interprete(blob):
    # Kernels de referencia: los mismos calculos enteros que TFLite Micro
    it = tf.lite.Interpreter(
        model_content=blob,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    return it


def redondear(v):
    """Redondeo a la mitad lejos del cero, como lroundf() en el sketch."""
    v = np.asarray(v, dtype=np.float64)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def cuantizar_entrada(X, scale, zp):
    q = redondear(X.astype(np.float32) / np.float32(scale)) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def predecir_tflite(blob, X):
    """Probabilidades (n, clases) en float; maneja E/S float32 o int8 una ventana a la vez."""
    it = interprete(blob)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']
    probs = np.zeros((len(X), len(CLASES)), dtype=np.float32)
    for i in range(len(X)):
        x = X[i:i + 1]
        if inp['dtype'] == np.int8:
            x = cuantizar_entrada(x, in_scale, in_zp)
        it.set_tensor(inp['index'], x)
        it.invoke()
        r = it.get_tensor(out['index'])[0]
        probs[i] = (r.astype(np.float32) - out_zp) * np.float32(out_scale) if out['dtype'] == np.int8 else r
    return probs


def operadores(blob):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    return [op['op_name'] for op in it._get_ops_details() if op['op_name'] != 'DELEGATE']  # API privada


def tensores(blob):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    filas = []
    for t in it.get_tensor_details():
        q = t['quantization_parameters']
        filas.append([t['index'], t['name'][:70], np.dtype(t['dtype']).name,
                      'x'.join(str(v) for v in t['shape']), len(q['scales']),
                      '%.6g' % (float(np.max(q['scales'])) if len(q['scales']) else 0.0),
                      int(q['zero_points'][0]) if len(q['zero_points']) else 0])
    return filas


# --- Metricas -------------------------------------------------------------------------------
def matriz_confusion(y, pred, n=len(CLASES)):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def metricas(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {'exactitud': tp.sum() / cm.sum(), 'f1_macro': f1.mean(),
            'precision': prec, 'recall': rec, 'f1': f1}


def suavizar(probs, y, quien, k=SUAVIZADO):
    """Promedio de las probabilidades de la ventana actual y las k-1 anteriores de la misma sesion,
    como hace la placa. Las ventanas de cada sesion estan en orden temporal dentro del arreglo."""
    out = probs.copy()
    for i in range(len(probs)):
        j = i
        while j > 0 and i - (j - 1) < k and y[j - 1] == y[i] and quien[j - 1] == quien[i]:
            j -= 1
        out[i] = probs[j:i + 1].mean(axis=0)
    return out


# --- Figuras --------------------------------------------------------------------------------
def dibujar_confusion(ax, cm, titulo):
    filas = cm.sum(axis=1, keepdims=True)
    pct = 100.0 * cm / np.maximum(filas, 1)
    ax.imshow(pct, cmap=CMAP_CONF, vmin=0, vmax=100)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = '#ffffff' if pct[i, j] >= 55 else TINTA
            ax.text(j, i, f'{pct[i, j]:.0f}', ha='center', va='center', fontsize=8, color=color)
    ax.set_xticks(range(len(ETIQUETAS)), ETIQUETAS, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(ETIQUETAS)), ETIQUETAS, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xlabel('prediccion')
    ax.set_ylabel('actividad real')
    ax.set_title(titulo, fontsize=9, loc='left')


#%% 0.3 Carga de datos
DF = cargar_datos()
SESIONES = {}
for (p, a), g in DF.groupby(['persona', 'accion'], sort=False):
    g = g.sort_values('t_us')
    SESIONES[(int(p), CLASES.index(a))] = g[COLUMNAS].to_numpy(np.float32)[RECORTE_MUESTRAS:-RECORTE_MUESTRAS]
faltantes = [(p, c) for p in PERSONAS for c in CLASES if (p, CLASES.index(c)) not in SESIONES]
if faltantes:
    raise ValueError(f'Faltan sesiones: {faltantes}')
print('Muestras:', len(DF), '| sesiones:', len(SESIONES))


#%% 1 Datos
filas = []
for (p, a), g in DF.groupby(['persona', 'accion'], sort=False):
    dt = np.diff(g['t_us'].to_numpy())
    filas.append([p, a, len(g), r4((g['t_us'].iloc[-1] - g['t_us'].iloc[0]) / 1e6),
                  r4(1e6 / dt.mean()), int(dt.min()), int(dt.max()), r4(dt.std())])
filas.sort(key=lambda f: (f[0], CLASES.index(f[1])))
guardar_tab(1, 'sesiones', ['persona', 'accion', 'muestras', 'duracion_s', 'fs_medida_hz',
                            'dt_min_us', 'dt_max_us', 'dt_desv_us'], filas)

est = DF.groupby(['accion', 'persona'])[COLUMNAS].agg(['mean', 'std'])
filas = []
for a in CLASES:
    for p in PERSONAS:
        fila = [a, p]
        for c in COLUMNAS:
            fila += [r4(est.loc[(a, p), (c, 'mean')]), r4(est.loc[(a, p), (c, 'std')])]
        filas.append(fila)
guardar_tab(1, 'estadisticas', ['accion', 'persona'] + [f'{c}_{s}' for c in COLUMNAS for s in ('media', 'desv')], filas)

mag = np.sqrt((DF[['ax', 'ay', 'az']] ** 2).sum(axis=1))
guardar_tab(1, 'rangos', ['magnitud', 'min', 'max', 'muestras_saturadas'], [
    ['aceleracion por eje (g)', r4(DF[['ax', 'ay', 'az']].min().min()), r4(DF[['ax', 'ay', 'az']].max().max()),
     int((DF[['ax', 'ay', 'az']].abs() >= 3.99).to_numpy().sum())],
    ['velocidad angular por eje (grados/s)', r4(DF[['gx', 'gy', 'gz']].min().min()),
     r4(DF[['gx', 'gy', 'gz']].max().max()), int((DF[['gx', 'gy', 'gz']].abs() >= 1999).to_numpy().sum())],
    ['norma de la aceleracion (g)', r4(mag.min()), r4(mag.max()), ''],
])

# Senales de la persona 1: 4 s del centro de cada sesion
n_ver = 4 * FS_HZ
t = np.arange(n_ver) / FS_HZ
fig, axes = plt.subplots(len(CLASES), 2, figsize=(9, 9.5), sharex=True)
for c, nombre in enumerate(CLASES):
    s = SESIONES[(1, c)]
    i0 = len(s) // 2 - n_ver // 2
    for k, eje in enumerate('xyz'):
        axes[c, 0].plot(t, s[i0:i0 + n_ver, k], color=COLOR_EJE[eje], label=eje)
        axes[c, 1].plot(t, s[i0:i0 + n_ver, 3 + k], color=COLOR_EJE[eje], label=eje)
    axes[c, 0].set_ylabel(f'{ETIQUETAS[c]}\n(g)')
    axes[c, 1].set_ylabel('(grados/s)')
axes[0, 0].set_title('Aceleracion', loc='left', fontsize=10)
axes[0, 1].set_title('Velocidad angular', loc='left', fontsize=10)
axes[0, 1].legend(ncol=3, loc='upper right', fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('tiempo (s)')
fig.suptitle('Persona 1: 4 s de cada actividad', x=0.01, ha='left', fontsize=11, color=TINTA)
fig.tight_layout()
guardar_fig(1, 'senales', fig)

# Giro medio alrededor del eje vertical (gy) en escaleras: cambia de signo entre personas
fig, ax = plt.subplots(figsize=(6.5, 3))
ancho = 0.38
for k, (c, color) in enumerate([(CLASES.index('subir_escaleras'), SERIES[0]),
                                (CLASES.index('bajar_escaleras'), SERIES[1])]):
    medias = [SESIONES[(p, c)][:, 4].mean() for p in PERSONAS]
    ax.bar(np.arange(3) + (k - 0.5) * ancho, medias, ancho * 0.95, color=color, label=ETIQUETAS[c])
ax.axhline(0, color=EJE, linewidth=0.8)
ax.set_xticks(range(3), [f'persona {p}' for p in PERSONAS])
ax.set_ylabel('media de gy (grados/s)')
ax.set_title('Giro medio alrededor del eje vertical durante la sesion', loc='left', fontsize=10)
ax.legend()
fig.tight_layout()
guardar_fig(1, 'giro_escaleras', fig)


#%% 0.4 Experimento LOSO
def corrida_loso(sensor, persona_prueba, semilla, restar_media=RESTAR_MEDIA,
                 con_aumento=USAR_AUMENTO, capas_conv=None, con_int8=True):
    canales = SENSORES[sensor]
    entrenan = [p for p in PERSONAS if p != persona_prueba]
    Vtr, ytr, _ = armar(entrenan, 'train')
    Vva, yva, _ = armar(entrenan, 'val')
    Vte, yte, qte = armar([persona_prueba], 'todo')
    if con_aumento:
        Vtr, ytr = aumentar(Vtr, ytr, np.random.default_rng(semilla))
    Xtr, mu, sd = preprocesar(Vtr, canales, restar_media=restar_media)
    Xva, _, _ = preprocesar(Vva, canales, mu, sd, restar_media)
    Xte, _, _ = preprocesar(Vte, canales, mu, sd, restar_media)
    model, hist = entrenar(Xtr, ytr, Xva, yva, semilla, capas_conv=capas_conv)
    res = {'sensor': sensor, 'persona_prueba': persona_prueba, 'semilla': semilla,
           'epocas': len(hist.history['loss']), 'n_train': len(Xtr), 'n_val': len(Xva), 'n_test': len(Xte)}
    p_fp = predecir_keras(model, Xte)
    res['cm_fp32'] = matriz_confusion(yte, p_fp.argmax(1))
    res['cm_fp32_suav'] = matriz_confusion(yte, suavizar(p_fp, yte, qte).argmax(1))
    if con_int8:
        rng = np.random.default_rng(SEED)
        X_rep = Xtr[rng.choice(len(Xtr), size=min(N_REP, len(Xtr)), replace=False)]
        blob = convertir(model, 'int8', X_rep)
        p_q = predecir_tflite(blob, Xte)
        res['cm_int8'] = matriz_confusion(yte, p_q.argmax(1))
        res['cm_int8_suav'] = matriz_confusion(yte, suavizar(p_q, yte, qte).argmax(1))
        res['acuerdo_fp32_int8'] = float((p_q.argmax(1) == p_fp.argmax(1)).mean())
    tf.keras.backend.clear_session()
    return res


LOSO = []
for sensor in SENSORES:
    for persona in PERSONAS:
        for semilla in SEMILLAS_LOSO:
            t0 = time.time()
            r = corrida_loso(sensor, persona, semilla)
            LOSO.append(r)
            print(f"LOSO {sensor:8s} prueba=p{persona} semilla={semilla} epocas={r['epocas']:3d} "
                  f"fp32={metricas(r['cm_fp32'])['exactitud']:.3f} "
                  f"int8={metricas(r['cm_int8'])['exactitud']:.3f} ({time.time() - t0:.0f} s)", flush=True)


def loso_de(sensor, clave, persona=None):
    return [r[clave] for r in LOSO if r['sensor'] == sensor and (persona is None or r['persona_prueba'] == persona)]


#%% 2 Preprocesamiento
Vtr_all, ytr_all, _ = armar(PERSONAS, 'train')
Vva_all, yva_all, _ = armar(PERSONAS, 'val')
factor = 1 + (AUMENTO_COPIAS if USAR_AUMENTO else 0)
filas = [['modelo final', '1, 2, 3', 'entrenamiento (primer 80 % de cada sesion)', len(Vtr_all) * factor],
         ['modelo final', '1, 2, 3', 'validacion (ultimo 20 % de cada sesion)', len(Vva_all)]]
for persona in PERSONAS:
    entrenan = [p for p in PERSONAS if p != persona]
    filas += [[f'LOSO prueba p{persona}', ', '.join(map(str, entrenan)), 'entrenamiento',
               len(armar(entrenan, 'train')[0]) * factor],
              [f'LOSO prueba p{persona}', ', '.join(map(str, entrenan)), 'validacion',
               len(armar(entrenan, 'val')[0])],
              [f'LOSO prueba p{persona}', str(persona), 'prueba (sesiones completas)',
               len(armar([persona], 'todo')[0])]]
guardar_tab(2, 'ventanas', ['particion', 'personas', 'conjunto', 'ventanas'], filas)

guardar_tab(2, 'parametros', ['Parametro', 'Valor'], [
    ['Frecuencia de muestreo (Hz)', FS_HZ],
    ['Ventana (muestras / s)', f'{VENTANA} / {VENTANA / FS_HZ:g}'],
    ['Paso entre ventanas (muestras / s)', f'{PASO} / {PASO / FS_HZ:g}'],
    ['Traslape', f'{100 * (1 - PASO / VENTANA):g} %'],
    ['Recorte por extremo de sesion (muestras)', RECORTE_MUESTRAS],
    ['Validacion', f'ultimo {100 * FRACCION_VAL:g} % de cada sesion de entrenamiento'],
    ['Columnas a las que se resta su media en cada ventana',
     ' '.join(COLUMNAS[c] for c in RESTAR_MEDIA) or 'ninguna'],
    ['Normalizacion', 'z-score por canal con media y desviacion del entrenamiento'],
    ['Aumento', f'{AUMENTO_COPIAS} copias con rotacion 3D aleatoria de hasta {AUMENTO_ROT_GRADOS:g} grados'
     if USAR_AUMENTO else 'no'],
])

if CORRER_ABLACION:
    filas_abl, detalle_abl = [], []
    for nombre, cols, aum, capas in ABLACION:
        es_principal = sorted(cols) == sorted(RESTAR_MEDIA) and aum == USAR_AUMENTO and capas == CAPAS_CONV
        accs = {}
        for persona in PERSONAS:
            for semilla in SEMILLAS_LOSO:
                if es_principal:   # ya se corrio en el experimento LOSO
                    cm = [r['cm_fp32'] for r in LOSO if r['sensor'] == SENSOR_ABLACION
                          and r['persona_prueba'] == persona and r['semilla'] == semilla][0]
                else:
                    cm = corrida_loso(SENSOR_ABLACION, persona, semilla, cols, aum, capas, con_int8=False)['cm_fp32']
                accs.setdefault(persona, []).append(metricas(cm)['exactitud'])
                detalle_abl.append([nombre.replace(chr(10), ' '), persona, semilla, r4(metricas(cm)['exactitud'])])
        todas = [a for v in accs.values() for a in v]
        filas_abl.append([nombre.replace(chr(10), ' '), ' '.join(COLUMNAS[c] for c in cols) or 'ninguna',
                          'si' if aum else 'no', campo_receptivo(capas), 'si' if es_principal else 'no']
                         + [r4(np.mean(accs[p])) for p in PERSONAS] + [r4(np.mean(todas)), r4(np.std(todas))])
        print('Ablacion', filas_abl[-1], flush=True)
    guardar_tab(2, 'ablacion', ['variante', 'restar_media', 'aumento', 'campo_receptivo', 'elegida']
                + [f'exactitud_prueba_p{p}' for p in PERSONAS] + ['exactitud_media', 'desv'], filas_abl)
    guardar_tab(2, 'ablacion_detalle', ['variante', 'persona_prueba', 'semilla', 'exactitud'], detalle_abl)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    medias = [f[-2] for f in filas_abl]
    desv = [f[-1] for f in filas_abl]
    colores = [SERIES[0] if f[4] == 'si' else GRIS for f in filas_abl]
    ax.bar(range(len(medias)), medias, 0.6, yerr=desv, color=colores, ecolor=TINTA_2, capsize=3)
    for i, m in enumerate(medias):
        ax.text(i, 0.03, f'{100 * m:.1f} %', ha='center', va='bottom', fontsize=8, color='#ffffff')
    ax.set_xticks(range(len(medias)), [a[0] for a in ABLACION], fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('exactitud LOSO')
    ax.set_title(f'Ablacion LOSO, {NOMBRE_SENSOR[SENSOR_ABLACION]}, FP32 (3 personas x {len(SEMILLAS_LOSO)} '
                 'semillas; azul = configuracion elegida)', loc='left', fontsize=9)
    fig.tight_layout()
    guardar_fig(2, 'ablacion', fig)


#%% 3 Modelo final (entrenado con las tres personas)
MODELOS_FINALES = {}
for sensor in SENSORES:
    canales = SENSORES[sensor]
    Vtr, ytr = Vtr_all, ytr_all
    if USAR_AUMENTO:
        Vtr, ytr = aumentar(Vtr, ytr, np.random.default_rng(SEED))
    Xtr, mu, sd = preprocesar(Vtr, canales)
    Xva, _, _ = preprocesar(Vva_all, canales, mu, sd)
    model, hist = entrenar(Xtr, ytr, Xva, yva_all, SEED, nombre=f'har_{sensor}')
    model.save(carpeta(3, 'modelos') / f'har_{sensor}.keras')
    MODELOS_FINALES[sensor] = {'model': model, 'hist': hist, 'mu': mu, 'sd': sd,
                               'Xtr': Xtr, 'Xva': Xva}
    print(f'Modelo final {sensor}: {len(hist.history["loss"])} epocas, '
          f'val_acc={max(hist.history["val_accuracy"]):.3f}', flush=True)

final = MODELOS_FINALES[SENSOR_PRINCIPAL]
filas_capas, macs_total = macs_modelo(final['model'])
guardar_tab(3, 'arquitectura', ['capa', 'tipo', 'forma_salida', 'parametros', 'macs'],
            filas_capas + [['total', '', '', final['model'].count_params(), macs_total]])
guardar_tab(3, 'entrenamiento', ['Parametro', 'Valor'], [
    ['Entrada', f'{VENTANA} muestras x {len(SENSORES[SENSOR_PRINCIPAL])} canales'],
    ['Arquitectura', descripcion_arquitectura()],
    ['Campo receptivo de la ultima convolucion (muestras)', campo_receptivo()],
    ['Optimizador', 'Adam'],
    ['Learning rate', LEARNING_RATE],
    ['Tamano de lote', BATCH_SIZE],
    ['Perdida', 'entropia cruzada categorica'],
    ['Epocas maximas', EPOCHS],
    ['Paciencia de EarlyStopping (val_loss, restaura los mejores pesos)', PATIENCE],
    ['Semillas LOSO', ' '.join(map(str, SEMILLAS_LOSO))],
    ['Semilla del modelo final', SEED],
] + [[f'Epocas ejecutadas, modelo final {s}', len(MODELOS_FINALES[s]['hist'].history['loss'])] for s in SENSORES]
  + [['Python', sys.version.split()[0]], ['TensorFlow', tf.__version__], ['Keras', tf.keras.__version__],
     ['NumPy', np.__version__], ['pandas', pd.__version__]])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
h = final['hist'].history
ep = np.arange(1, len(h['loss']) + 1)
for ax, clave, titulo in [(axes[0], 'loss', 'Perdida'), (axes[1], 'accuracy', 'Exactitud')]:
    ax.plot(ep, h[clave], color=SERIES[0], label='entrenamiento')
    ax.plot(ep, h['val_' + clave], color=SERIES[1], label='validacion')
    ax.set_xlabel('epoca')
    ax.set_title(titulo, loc='left', fontsize=10)
axes[1].legend()
fig.suptitle(f'Modelo final, {NOMBRE_SENSOR[SENSOR_PRINCIPAL]} (validacion: ultimo 20 % de cada sesion)',
             x=0.01, ha='left', fontsize=10, color=TINTA)
fig.tight_layout()
guardar_fig(3, 'entrenamiento', fig)


#%% 4 Clasificacion (LOSO)
filas = []
for sensor in SENSORES:
    for persona in PERSONAS + [None]:
        fila = [sensor, f'p{persona}' if persona else 'todas']
        for clave in ('cm_fp32', 'cm_fp32_suav', 'cm_int8', 'cm_int8_suav'):
            ms = [metricas(cm) for cm in loso_de(sensor, clave, persona)]
            fila += [r4(np.mean([m['exactitud'] for m in ms])), r4(np.std([m['exactitud'] for m in ms]))]
        ms = [metricas(cm) for cm in loso_de(sensor, 'cm_fp32', persona)]
        fila += [r4(np.mean([m['f1_macro'] for m in ms]))]
        filas.append(fila)
guardar_tab(4, 'loso', ['sensores', 'persona_prueba',
                        'exactitud_fp32', 'desv_fp32', 'exactitud_fp32_suavizada', 'desv_fp32_suavizada',
                        'exactitud_int8', 'desv_int8', 'exactitud_int8_suavizada', 'desv_int8_suavizada',
                        'f1_macro_fp32'], filas)
guardar_tab(4, 'loso_detalle', ['sensores', 'persona_prueba', 'semilla', 'epocas', 'n_train', 'n_val', 'n_test',
                                'exactitud_fp32', 'exactitud_int8', 'acuerdo_fp32_int8'],
            [[r['sensor'], r['persona_prueba'], r['semilla'], r['epocas'], r['n_train'], r['n_val'], r['n_test'],
              r4(metricas(r['cm_fp32'])['exactitud']), r4(metricas(r['cm_int8'])['exactitud']),
              r4(r['acuerdo_fp32_int8'])] for r in LOSO])

CM_PRINCIPAL = sum(loso_de(SENSOR_PRINCIPAL, 'cm_fp32'))
m = metricas(CM_PRINCIPAL)
guardar_tab(4, 'por_clase', ['clase', 'precision', 'recall', 'f1', 'ventanas'],
            [[CLASES[i], r4(m['precision'][i]), r4(m['recall'][i]), r4(m['f1'][i]), int(CM_PRINCIPAL[i].sum())]
             for i in range(len(CLASES))] + [['macro', r4(m['precision'].mean()), r4(m['recall'].mean()),
                                              r4(m['f1_macro']), int(CM_PRINCIPAL.sum())]])
guardar_tab(4, 'confusion', ['real \\ prediccion'] + CLASES,
            [[CLASES[i]] + CM_PRINCIPAL[i].tolist() for i in range(len(CLASES))])

fig, axes = plt.subplots(1, len(PERSONAS) + 1, figsize=(13, 3.6))
for ax, persona in zip(axes, PERSONAS + [None]):
    cm = sum(loso_de(SENSOR_PRINCIPAL, 'cm_fp32', persona))
    titulo = f'prueba con persona {persona}' if persona else 'las tres personas'
    dibujar_confusion(ax, cm, f'{titulo}\nexactitud {100 * metricas(cm)["exactitud"]:.1f} %')
    if persona != PERSONAS[0]:
        ax.set_ylabel('')
fig.suptitle(f'Matriz de confusion LOSO, {NOMBRE_SENSOR[SENSOR_PRINCIPAL]}, FP32 '
             f'(% por fila, suma de {len(SEMILLAS_LOSO)} semillas)',
             x=0.01, ha='left', fontsize=10, color=TINTA)
fig.tight_layout()
guardar_fig(4, 'confusion', fig)


#%% 5 Acelerometro frente a acelerometro + giroscopo
filas = []
for sensor in SENSORES:
    cm = sum(loso_de(sensor, 'cm_fp32'))
    m = metricas(cm)
    accs = [metricas(c)['exactitud'] for c in loso_de(sensor, 'cm_fp32')]
    filas.append([sensor, len(SENSORES[sensor]), r4(np.mean(accs)), r4(np.std(accs)), r4(m['f1_macro'])]
                 + [r4(v) for v in m['f1']])
guardar_tab(5, 'sensores', ['sensores', 'canales', 'exactitud_fp32', 'desv', 'f1_macro']
            + [f'f1_{c}' for c in CLASES], filas)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ancho = 0.38
for k, sensor in enumerate(['acc', 'acc_gir']):
    medias = [np.mean([metricas(c)['exactitud'] for c in loso_de(sensor, 'cm_fp32', p)]) for p in PERSONAS]
    desv = [np.std([metricas(c)['exactitud'] for c in loso_de(sensor, 'cm_fp32', p)]) for p in PERSONAS]
    x = np.arange(len(PERSONAS)) + (k - 0.5) * ancho
    ax.bar(x, medias, ancho * 0.95, yerr=desv, color=COLOR_SENSOR[sensor], ecolor=TINTA_2, capsize=3,
           label=NOMBRE_SENSOR[sensor])
    for xi, mi in zip(x, medias):
        ax.text(xi, mi + 0.04, f'{100 * mi:.0f}', ha='center', fontsize=8, color=TINTA)
ax.set_xticks(range(len(PERSONAS)), [f'prueba con persona {p}' for p in PERSONAS], fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel('exactitud LOSO (FP32)')
ax.set_title(f'Exactitud por persona de prueba (media y desviacion de {len(SEMILLAS_LOSO)} semillas)',
             loc='left', fontsize=9)
ax.legend(loc='upper left', ncol=2, fontsize=8)
fig.tight_layout()
guardar_fig(5, 'sensores', fig)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
for ax, sensor in zip(axes, ['acc', 'acc_gir']):
    cm = sum(loso_de(sensor, 'cm_fp32'))
    dibujar_confusion(ax, cm, f'{NOMBRE_SENSOR[sensor]}\nexactitud {100 * metricas(cm)["exactitud"]:.1f} %')
axes[1].set_ylabel('')
fig.suptitle('Matriz de confusion LOSO, FP32 (% por fila)', x=0.01, ha='left', fontsize=10, color=TINTA)
fig.tight_layout()
guardar_fig(5, 'confusion_sensores', fig)


#%% 6 Modelo original (FP32) frente al cuantizado (INT8)
TFLITE = {}
for sensor, info in MODELOS_FINALES.items():
    rng = np.random.default_rng(SEED)
    X_rep = info['Xtr'][rng.choice(len(info['Xtr']), size=N_REP, replace=False)]
    for modo in ('fp32', 'int8'):
        blob = convertir(info['model'], modo, X_rep if modo == 'int8' else None)
        TFLITE[(sensor, modo)] = blob
        guardar_modelo(6, f'har_{sensor}_{modo}', blob)

filas = []
for sensor in SENSORES:
    for modo in ('fp32', 'int8'):
        cms = loso_de(sensor, f'cm_{modo}')
        accs = [metricas(c)['exactitud'] for c in cms]
        cm_tot = sum(cms)
        blob = TFLITE[(sensor, modo)]
        p_val = predecir_tflite(blob, MODELOS_FINALES[sensor]['Xva'])
        filas.append([sensor, modo, len(blob), MODELOS_FINALES[sensor]['model'].count_params(),
                      r4(np.mean(accs)), r4(np.std(accs)), r4(metricas(cm_tot)['f1_macro']),
                      r4((p_val.argmax(1) == yva_all).mean()),
                      r4(np.mean(loso_de(sensor, 'acuerdo_fp32_int8'))) if modo == 'int8' else ''])
guardar_tab(6, 'cuantizacion', ['sensores', 'modelo', 'bytes_tflite', 'parametros', 'exactitud_loso', 'desv_loso',
                                'f1_macro_loso', 'exactitud_val_modelo_final', 'acuerdo_con_fp32_loso'], filas)

filas = []
for sensor in SENSORES:
    b32, b8 = len(TFLITE[(sensor, 'fp32')]), len(TFLITE[(sensor, 'int8')])
    n_par = MODELOS_FINALES[sensor]['model'].count_params()
    filas.append([sensor, n_par, n_par * 4, b32, b8, r4(100 * (1 - b8 / b32)),
                  macs_modelo(MODELOS_FINALES[sensor]['model'])[1]])
guardar_tab(6, 'tamano', ['sensores', 'parametros', 'bytes_parametros_fp32', 'bytes_tflite_fp32',
                          'bytes_tflite_int8', 'reduccion_pct', 'macs_por_inferencia'], filas)

filas = []
for (sensor, modo), blob in TFLITE.items():
    it = interprete(blob)
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    filas.append([sensor, modo, np.dtype(i['dtype']).name, 'x'.join(map(str, i['shape'])),
                  '%.8g' % i['quantization'][0], i['quantization'][1],
                  np.dtype(o['dtype']).name, '%.8g' % o['quantization'][0], o['quantization'][1],
                  ' '.join(operadores(blob))])
guardar_tab(6, 'entrada_salida', ['sensores', 'modelo', 'tipo_entrada', 'forma_entrada', 'scale_entrada',
                                  'zero_point_entrada', 'tipo_salida', 'scale_salida', 'zero_point_salida',
                                  'operadores'], filas)
guardar_tab(6, 'tensores_int8', ['idx', 'nombre', 'tipo', 'forma', 'n_scales', 'scale_max', 'zero_point'],
            tensores(TFLITE[(SENSOR_PRINCIPAL, 'int8')]))

fig, axes = plt.subplots(1, len(SENSORES), figsize=(11, 3.3), sharey=True)
ancho = 0.38
for ax, sensor in zip(axes, [SENSOR_PRINCIPAL] + [s for s in SENSORES if s != SENSOR_PRINCIPAL]):
    for k, (modo, color) in enumerate([('fp32', COLOR_FP32), ('int8', COLOR_INT8)]):
        medias = [np.mean([metricas(c)['exactitud'] for c in loso_de(sensor, f'cm_{modo}', p)])
                  for p in PERSONAS]
        x = np.arange(len(PERSONAS)) + (k - 0.5) * ancho
        ax.bar(x, medias, ancho * 0.95, color=color, label=modo.upper())
        for xi, mi in zip(x, medias):
            ax.text(xi, mi + 0.02, f'{100 * mi:.1f}', ha='center', fontsize=7.5, color=TINTA)
    ax.set_xticks(range(len(PERSONAS)), [f'prueba con p{p}' for p in PERSONAS], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(NOMBRE_SENSOR[sensor], loc='left', fontsize=9)
axes[0].set_ylabel('exactitud LOSO')
axes[0].legend(loc='upper left', ncol=2, fontsize=8)
fig.suptitle(f'FP32 frente a INT8 (media de {len(SEMILLAS_LOSO)} semillas)', x=0.01, ha='left',
             fontsize=10, color=TINTA)
fig.tight_layout()
guardar_fig(6, 'cuantizacion', fig)


#%% 7 Arduino: model.h, har_config.h y sketches
OP_ADD = {
    'CONV_2D': 'AddConv2D',
    'DEPTHWISE_CONV_2D': 'AddDepthwiseConv2D',
    'EXPAND_DIMS': 'AddExpandDims',
    'RESHAPE': 'AddReshape',
    'SQUEEZE': 'AddSqueeze',
    'MEAN': 'AddMean',
    'MAX_POOL_2D': 'AddMaxPool2D',
    'AVERAGE_POOL_2D': 'AddAveragePool2D',
    'FULLY_CONNECTED': 'AddFullyConnected',
    'SOFTMAX': 'AddSoftmax',
    'RELU': 'AddRelu',
    'QUANTIZE': 'AddQuantize',
    'DEQUANTIZE': 'AddDequantize',
}


def tflite_to_header(blob, desc):
    body = '\n'.join('  ' + ', '.join(f'0x{b:02x}' for b in blob[i:i + 12]) + ','
                     for i in range(0, len(blob), 12))
    return (f'#pragma once\n#include <stdint.h>\n\n'
            f'// Modelo {desc} generado por Proyecto_3.py. No editar.\n'
            f'alignas(16) const unsigned char g_model[] = {{\n{body}\n}};\n'
            f'const unsigned int g_model_len = sizeof(g_model);\n')


def f32(v):
    """Literal float de C++ que reproduce exactamente el float32 de Python."""
    texto = '%.9g' % float(np.float32(v))
    if not any(c in texto for c in '.e'):
        texto += '.0'          # '0f' o '2f' no son literales validos
    return texto + 'f'


def ventanas_autoprueba():
    """Una ventana cruda de validacion por clase, de personas distintas."""
    Vva, yva, qva = armar(PERSONAS, 'val')
    elegidas = []
    for c in range(len(CLASES)):
        persona = PERSONAS[c % len(PERSONAS)]
        idx = np.where((yva == c) & (qva == persona))[0]
        elegidas.append(idx[len(idx) // 2])
    return Vva[elegidas], yva[elegidas], qva[elegidas]


def config_header(sensor, modo, blob):
    info = MODELOS_FINALES[sensor]
    canales = SENSORES[sensor]
    V, y, q = ventanas_autoprueba()
    X, _, _ = preprocesar(V, canales, info['mu'], info['sd'])
    esperado = predecir_tflite(blob, X)
    lineas = [
        '#pragma once',
        '#include <stdint.h>',
        '',
        f'// Configuracion del modelo har_{sensor}_{modo}, generada por Proyecto_3.py. No editar.',
        '// Debe coincidir con preprocesar() de Proyecto_3.py.',
        f'constexpr int kVentana = {VENTANA};               // muestras por ventana (2 s)',
        f'constexpr int kPaso = {PASO};                   // se clasifica cada {PASO} muestras',
        f'constexpr uint32_t kPeriodoUs = {1000000 // FS_HZ};       // {FS_HZ} Hz',
        f'constexpr int kCanales = {len(canales)};',
        f'constexpr int kNumClases = {len(CLASES)};',
        '// Columnas crudas en el orden del CSV: 0 ax, 1 ay, 2 az (g), 3 gx, 4 gy, 5 gz (grados/s)',
        'constexpr int kColumna[kCanales] = {' + ', '.join(map(str, canales)) + '};',
        '// Columnas crudas a las que se resta su media dentro de cada ventana',
        'constexpr bool kRestarMedia[6] = {' + ', '.join('true' if c in RESTAR_MEDIA else 'false'
                                                         for c in range(len(COLUMNAS))) + '};',
        'constexpr float kMedia[kCanales] = {' + ', '.join(f32(v) for v in info['mu']) + '};',
        'constexpr float kDesv[kCanales] = {' + ', '.join(f32(v) for v in info['sd']) + '};',
        'const char* const kClases[kNumClases] = {' + ', '.join(f'"{c}"' for c in CLASES) + '};',
        '',
        '// Autoprueba: ventanas crudas del dataset (bloque de validacion) y la salida que da',
        '// el mismo modelo en Python con TFLite (kernels de referencia).',
        f'constexpr int kNumPruebas = {len(V)};',
        'const int kPruebaClase[kNumPruebas] = {' + ', '.join(str(int(v)) for v in y) + '};',
        'const int kPruebaPersona[kNumPruebas] = {' + ', '.join(str(int(v)) for v in q) + '};',
        'const float kPruebaEsperada[kNumPruebas][kNumClases] = {',
    ]
    for fila in esperado:
        lineas.append('  {' + ', '.join(f32(v) for v in fila) + '},')
    lineas.append('};')
    lineas.append('const float kPruebaX[kNumPruebas][kVentana][6] = {')
    for v in V:
        lineas.append('  {')
        for muestra in v:
            lineas.append('    {' + ', '.join(f32(x) for x in muestra) + '},')
        lineas.append('  },')
    lineas.append('};')
    return '\n'.join(lineas) + '\n', esperado, y


SKETCH = r'''// TinyML 2026 | Reconocimiento de actividad fisica | modelo @@NAME@@
// Placa: Arduino Nano 33 BLE Sense (Rev1, IMU LSM9DS1) o Sense Rev2 (IMU BMI270).
// Bibliotecas: Chirale_TensorFlowLite 2.0.0 y Arduino_LSM9DS1 (Rev1) o Arduino_BMI270_BMM150 (Rev2).
// Generado por Proyecto_3.py junto con model.h y har_config.h. Todo se muestra en el Monitor Serie.
//
// Al encender: autoprueba con ventanas del dataset (debe dar lo mismo que Python) y
// cronometraje de @@NRUNS@@ inferencias. Despues: lee la IMU a 50 Hz y clasifica cada 0.5 s
// la ultima ventana de 2 s.
#include <Arduino.h>
#include <math.h>
#include <stdint.h>

// 0 = Nano 33 BLE Sense (Rev1, LSM9DS1); 1 = Nano 33 BLE Sense Rev2 (BMI270).
// Debe ser la misma revision de placa con la que se grabo el dataset.
#ifndef SENSE_REV2
#define SENSE_REV2 @@REV2@@
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

constexpr int kTensorArenaSize = @@ARENA@@;
constexpr int kNumInferencias = @@NRUNS@@;   // inferencias cronometradas en setup()
constexpr int kSuavizado = @@SUAVIZADO@@;          // promedio de las ultimas probabilidades
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
    Serial.print("[error] modelo=@@NAME@@ ");
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
  Serial.print("[autoprueba] modelo=@@NAME@@ coinciden="); Serial.print(autoprueba_ok);
  Serial.print("/"); Serial.print(kNumPruebas);
  Serial.print(" dif_max="); Serial.println(autoprueba_dif_max, 5);
  Serial.print("[tiempo] modelo=@@NAME@@ n="); Serial.print(kNumInferencias);
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

@@RESOLVER_DECL@@
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
  Serial.println("modelo: @@NAME@@");
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
'''


def build_sketch(sensor, modo):
    blob = TFLITE[(sensor, modo)]
    nombre = f'HAR_{sensor}_{modo}'
    folder = ARDUINO / nombre
    folder.mkdir(parents=True, exist_ok=True)
    ops = sorted(set(operadores(blob)))
    sin_mapeo = [op for op in ops if op not in OP_ADD]
    if sin_mapeo:
        raise ValueError(f'Operadores sin mapeo en OP_ADD: {sin_mapeo}')
    decl = '\n'.join([f'  static tflite::MicroMutableOpResolver<{len(ops)}> resolver;']
                     + [f'  resolver.{OP_ADD[op]}();' for op in ops])
    code = (SKETCH.replace('@@NAME@@', nombre)
                  .replace('@@REV2@@', str(SENSE_REV2))
                  .replace('@@ARENA@@', str(TENSOR_ARENA_BYTES))
                  .replace('@@NRUNS@@', str(N_TIMING_RUNS))
                  .replace('@@SUAVIZADO@@', str(SUAVIZADO))
                  .replace('@@RESOLVER_DECL@@', decl))
    (folder / f'{nombre}.ino').write_text(code, encoding='ascii')
    (folder / 'model.h').write_text(tflite_to_header(blob, f'har_{sensor}_{modo}'), encoding='ascii')
    cfg, esperado, y = config_header(sensor, modo, blob)
    (folder / 'har_config.h').write_text(cfg, encoding='ascii')
    return nombre, ops, esperado, y


filas_auto = []
for sensor, modo in SKETCHES:
    nombre, ops, esperado, y = build_sketch(sensor, modo)
    print('Sketch:', nombre, '| modelo:', len(TFLITE[(sensor, modo)]), 'bytes | operadores:', ' '.join(ops))
    for i, (fila, clase) in enumerate(zip(esperado, y)):
        filas_auto.append([nombre, i, CLASES[clase], CLASES[int(fila.argmax())], r4(fila.max())])
guardar_tab(7, 'autoprueba_python', ['sketch', 'i', 'clase_real', 'pred_python', 'p'], filas_auto)

# Tabla de la placa: la llenan compilar_sketches.py (Flash y RAM) y la sesion con la tarjeta.
# No se sobrescribe si ya existe.
TABLA_PLACA = carpeta(7) / '7_tab_placa.csv'
if not TABLA_PLACA.exists():
    guardar_tab(7, 'placa', ['sketch', 'sensores', 'modelo', 'bytes_tflite', 'compila', 'flash_bytes',
                             'ram_estatica_bytes', 'arena_usada_bytes', 'autoprueba', 't_min_us', 't_max_us',
                             't_prom_us', 't_desv_us', 'prep_prom_us'],
                [[f'HAR_{s}_{m}', s, m, len(TFLITE[(s, m)])] + [PENDIENTE] * 10 for s, m in SKETCHES])
else:
    print('Tabla:', TABLA_PLACA.relative_to(ROOT), '(ya existe, no se sobrescribe)')

print(f'Tiempo total: {(time.time() - T_INICIO) / 60:.1f} min')
