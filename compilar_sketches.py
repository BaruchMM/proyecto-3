"""Compila los sketches de arduino/ para la Nano 33 BLE Sense. No necesita la tarjeta.

Guarda la salida del compilador en resultados/7_arduino/evidencias/7_compilacion.txt y anota
compila, Flash y RAM estatica en resultados/7_arduino/7_tab_placa.csv.
Correr despues de Proyecto_3.py. Requiere las bibliotecas Chirale_TensorFlowLite 2.0.0 y
Arduino_LSM9DS1 o Arduino_BMI270_BMM150 (segun SENSE_REV2 del sketch).

COMPILAR_OTRA_REVISION = True tambien compila cada sketch con la otra revision de la placa
(solo para comprobar que compila; la tabla guarda la revision que trae el .ino).
"""
import csv, re, subprocess, unicodedata
from pathlib import Path

ARDUINO_CLI = r'C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe'
FQBN = 'arduino:mbed_nano:nano33ble'
COMPILAR_OTRA_REVISION = True

ROOT = Path(__file__).resolve().parent
SKETCHES = ROOT / 'arduino'
TABLA = ROOT / 'resultados' / '7_arduino' / '7_tab_placa.csv'
LOG = ROOT / 'resultados' / '7_arduino' / 'evidencias' / '7_compilacion.txt'


def to_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def compilar(carpeta, rev2=None):
    cmd = [ARDUINO_CLI, 'compile', '--fqbn', FQBN, str(carpeta)]
    if rev2 is not None:
        cmd += ['--build-property', f'compiler.cpp.extra_flags=-DSENSE_REV2={rev2}']
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    tam = [int(n) for n in re.findall(r'(\d+) bytes \(\d+%\)', p.stdout)]
    ok = p.returncode == 0 and len(tam) == 2
    return ok, tam, to_ascii(p.stdout + p.stderr)


with open(TABLA, newline='', encoding='utf-8-sig') as f:
    filas = list(csv.DictReader(f))

LOG.parent.mkdir(parents=True, exist_ok=True)
with open(LOG, 'w', newline='\n', encoding='ascii') as log:
    log.write(f'Compilacion para {FQBN} con arduino-cli\n')
    for fila in filas:
        nombre = fila['sketch']
        carpeta = SKETCHES / nombre
        ino = (carpeta / f'{nombre}.ino').read_text(encoding='ascii')
        rev2 = int(re.search(r'#define SENSE_REV2 (\d)', ino).group(1))
        ok, tam, salida = compilar(carpeta)
        fila['compila'] = 'si' if ok else 'no'
        fila['flash_bytes'], fila['ram_estatica_bytes'] = tam if ok else ('pendiente', 'pendiente')
        log.write(f'\n===== {nombre} (SENSE_REV2={rev2}, la del .ino) =====\n{salida}\n')
        log.flush()
        print(nombre, f'SENSE_REV2={rev2}', fila['compila'], fila['flash_bytes'], fila['ram_estatica_bytes'],
              flush=True)
        if COMPILAR_OTRA_REVISION:
            ok2, tam2, salida2 = compilar(carpeta, 1 - rev2)
            log.write(f'\n===== {nombre} (SENSE_REV2={1 - rev2}, solo comprobacion) =====\n{salida2}\n')
            log.flush()
            print(nombre, f'SENSE_REV2={1 - rev2}', 'si' if ok2 else 'no', *(tam2 if ok2 else []), flush=True)

with open(TABLA, 'w', newline='', encoding='ascii') as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
print('Salida del compilador:', LOG.relative_to(ROOT))
